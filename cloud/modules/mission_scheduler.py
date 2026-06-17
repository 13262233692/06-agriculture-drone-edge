import os
import sys
import time
import uuid
import json
import logging
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.heatmap_generator import HeatmapGenerator
from modules.path_planner import (
    GeneticPathPlanner,
    DroneState,
    TargetZone,
    PlannedPath,
    Waypoint as PlannerWaypoint,
)

logger = logging.getLogger(__name__)


@dataclass
class DroneMissionState:
    drone_id: str
    current_mission_id: str = ""
    last_heatmap_generation: float = 0.0
    last_path_planning: float = 0.0
    pending_commands: int = 0
    battery_level_pct: float = 100.0
    chemical_level_pct: float = 100.0
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 50.0
    home_latitude: float = 0.0
    home_longitude: float = 0.0
    cruise_speed: float = 8.0
    spray_rate: float = 0.5
    last_status_update: float = 0.0


class DynamicMissionScheduler:
    """
    动态任务调度器。
    核心流程：
      1. 汇总当日病害点位数据
      2. KDE 生成热力图
      3. 遗传算法规划补喷路径（考虑电量/药量约束 + 安全返航）
      4. 通过 gRPC 下发新航点指令到边缘设备

    调度策略：
      - 数据驱动：新数据积累到一定量后自动重规划
      - 定时触发：周期性检查并重规划
      - 手动触发：支持外部触发重规划
    """

    def __init__(
        self,
        config: Dict[str, Any],
        send_command_fn: Optional[Callable[[str, str, Dict[str, Any]], bool]] = None,
    ):
        self._config = config
        self._send_command_fn = send_command_fn

        sched_cfg = config.get("mission_scheduler", {}) if isinstance(config, dict) else {}
        self._field_id = sched_cfg.get("field_id", "FIELD-001")
        self._schedule_interval_s = float(sched_cfg.get("schedule_interval_s", 300.0))
        self._min_new_frames_for_replan = int(sched_cfg.get("min_new_frames_for_replan", 50))
        self._max_waypoints_per_mission = int(sched_cfg.get("max_waypoints_per_mission", 100))
        self._high_risk_threshold = float(sched_cfg.get("high_risk_threshold", 0.6))
        self._max_high_risk_zones = int(sched_cfg.get("max_high_risk_zones", 20))
        self._auto_schedule_enabled = bool(sched_cfg.get("auto_schedule_enabled", True))
        self._spray_width_m = float(sched_cfg.get("spray_width_m", 5.0))

        self._heatmap_gen = HeatmapGenerator(config)
        self._path_planner = GeneticPathPlanner(config)

        self._drone_states: Dict[str, DroneMissionState] = {}
        self._drone_states_lock = threading.Lock()

        self._frame_count_since_last_plan: Dict[str, int] = {}
        self._frame_count_lock = threading.Lock()

        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._missions_generated = 0
        self._commands_sent = 0
        self._stats_lock = threading.Lock()

        logger.info(
            f"DynamicMissionScheduler initialized: "
            f"field={self._field_id} "
            f"interval={self._schedule_interval_s}s "
            f"auto={self._auto_schedule_enabled}"
        )

    def start(self) -> bool:
        if self._running:
            return True

        self._running = True
        self._stop_event.clear()

        if self._auto_schedule_enabled:
            self._scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                daemon=True,
                name="MissionSchedulerThread",
            )
            self._scheduler_thread.start()

        logger.info("DynamicMissionScheduler started")
        return True

    def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=5.0)

        logger.info("DynamicMissionScheduler stopped")

    def update_drone_status(self, status: Dict[str, Any]) -> None:
        """更新无人机状态（从 gRPC 状态流接收）"""
        drone_id = status.get("drone_id", "UNKNOWN")
        if not drone_id:
            return

        with self._drone_states_lock:
            state = self._drone_states.get(drone_id)
            if state is None:
                state = DroneMissionState(drone_id=drone_id)
                self._drone_states[drone_id] = state

            state.battery_level_pct = float(status.get("battery_level", state.battery_level_pct))
            state.chemical_level_pct = float(status.get("chemical_level", state.chemical_level_pct))

            pos = status.get("current_position", {})
            if pos:
                state.latitude = float(pos.get("latitude", state.latitude))
                state.longitude = float(pos.get("longitude", state.longitude))
                state.altitude = float(pos.get("altitude", state.altitude))

            home = status.get("home_position", {})
            if home:
                state.home_latitude = float(home.get("latitude", state.home_latitude))
                state.home_longitude = float(home.get("longitude", state.home_longitude))

            state.cruise_speed = float(status.get("cruise_speed", state.cruise_speed))
            state.spray_rate = float(status.get("spray_rate", state.spray_rate))
            state.current_mission_id = str(status.get("current_mission_id", state.current_mission_id))
            state.last_status_update = time.time()

        logger.debug(f"Updated status for drone {drone_id}: battery={state.battery_level_pct:.1f}%")

    def notify_frame_received(self, drone_id: str, frame_id: int, detection_count: int) -> None:
        """通知接收到新帧，累计到一定数量触发重规划"""
        if detection_count == 0:
            return

        with self._frame_count_lock:
            self._frame_count_since_last_plan[drone_id] = (
                self._frame_count_since_last_plan.get(drone_id, 0) + 1
            )

            count = self._frame_count_since_last_plan[drone_id]
            if count >= self._min_new_frames_for_replan:
                logger.info(
                    f"New frame threshold reached ({count}) for drone {drone_id}, "
                    f"triggering replan..."
                )
                self._frame_count_since_last_plan[drone_id] = 0
                threading.Thread(
                    target=lambda: self.trigger_mission_for_drone(drone_id),
                    daemon=True,
                ).start()

    def trigger_mission_for_drone(self, drone_id: str) -> Dict[str, Any]:
        """
        为指定无人机触发一次任务规划。
        生成热力图 → 提取高风险区 → 规划路径 → 下发航点
        """
        logger.info(f"Triggering mission for drone {drone_id}...")

        try:
            with self._drone_states_lock:
                state = self._drone_states.get(drone_id)
                if state is None:
                    logger.warning(f"Drone {drone_id} not registered for mission scheduling")
                    return {"success": False, "reason": "drone_not_registered"}

                drone_state = DroneState(
                    drone_id=drone_id,
                    latitude=state.latitude,
                    longitude=state.longitude,
                    altitude=state.altitude,
                    battery_level_pct=state.battery_level_pct,
                    chemical_level_pct=state.chemical_level_pct,
                    cruise_speed_m_s=state.cruise_speed,
                    spray_rate_l_per_s=state.spray_rate,
                    home_latitude=state.home_latitude,
                    home_longitude=state.home_longitude,
                    spray_width_m=self._spray_width_m,
                )

            heatmap = self._heatmap_gen.generate_heatmap(
                field_id=self._field_id,
                drone_id=drone_id,
                hours_back=24,
            )
            logger.info(
                f"Heatmap generated: {len(heatmap.get('cells', []))} cells, "
                f"{heatmap.get('total_points', 0)} points"
            )

            high_risk_zones = self._heatmap_gen.get_high_risk_zones(
                heatmap,
                threshold_pct=self._high_risk_threshold,
                max_zones=self._max_high_risk_zones,
            )
            logger.info(f"Found {len(high_risk_zones)} high-risk zones")

            if not high_risk_zones:
                logger.info("No high-risk zones found, no mission needed")
                return {
                    "success": True,
                    "mission_planned": False,
                    "reason": "no_high_risk_zones",
                }

            targets = []
            for i, zone in enumerate(high_risk_zones):
                target = TargetZone(
                    zone_id=i,
                    latitude=zone["latitude"],
                    longitude=zone["longitude"],
                    density=zone["density"],
                    severity_score=zone.get("severity_score", 0.5),
                    radius_m=30.0,
                    priority=zone["density"],
                )
                targets.append(target)

            planned_path = self._path_planner.plan(
                drone_state=drone_state,
                target_zones=targets,
                mission_type="SUPPLEMENTARY_SPRAY",
            )
            logger.info(
                f"Path planned: {len(planned_path.waypoints)} waypoints, "
                f"distance={planned_path.total_distance_m:.0f}m, "
                f"safe_rtl={planned_path.return_to_home_safe}"
            )

            if not planned_path.return_to_home_safe:
                logger.warning(
                    f"Planned path cannot return home safely for drone {drone_id}, "
                    f"truncated automatically"
                )

            mission_id = f"MISSION-{uuid.uuid4().hex[:12].upper()}"
            success = self._send_mission_to_drone(
                drone_id=drone_id,
                mission_id=mission_id,
                planned_path=planned_path,
                heatmap=heatmap,
            )

            with self._drone_states_lock:
                state.last_path_planning = time.time()
                state.last_heatmap_generation = time.time()
                if success:
                    state.current_mission_id = mission_id
                    state.pending_commands += 1

            with self._stats_lock:
                self._missions_generated += 1
                if success:
                    self._commands_sent += 1

            return {
                "success": success,
                "mission_planned": True,
                "mission_id": mission_id,
                "waypoints": len(planned_path.waypoints),
                "distance_m": planned_path.total_distance_m,
                "battery_used_pct": planned_path.estimated_battery_used_pct,
                "chemical_used_pct": planned_path.estimated_chemical_used_pct,
                "safe_return": planned_path.return_to_home_safe,
                "high_risk_zones": len(high_risk_zones),
            }

        except Exception as e:
            logger.error(f"Mission planning failed for drone {drone_id}: {e}")
            import traceback

            traceback.print_exc()
            return {"success": False, "reason": str(e)}

    def trigger_mission_all_drones(self) -> Dict[str, Any]:
        """为所有已注册的无人机触发任务规划"""
        results: Dict[str, Any] = {}

        with self._drone_states_lock:
            drone_ids = list(self._drone_states.keys())

        for drone_id in drone_ids:
            results[drone_id] = self.trigger_mission_for_drone(drone_id)

        return {
            "total_drones": len(drone_ids),
            "results": results,
        }

    def _send_mission_to_drone(
        self,
        drone_id: str,
        mission_id: str,
        planned_path: PlannedPath,
        heatmap: Dict[str, Any],
    ) -> bool:
        """将任务下发给无人机（通过 gRPC 命令队列）"""
        if self._send_command_fn is None:
            logger.warning("No send_command function provided, mission not sent")
            return False

        waypoints = []
        for wp in planned_path.waypoints[: self._max_waypoints_per_mission]:
            wp_dict = {
                "waypoint_id": wp.wp_id,
                "latitude": wp.latitude,
                "longitude": wp.longitude,
                "altitude": wp.altitude,
                "speed": wp.speed,
                "action": wp.action,
                "spray_density": wp.spray_density,
            }
            waypoints.append(wp_dict)

        mission_plan = {
            "mission_id": mission_id,
            "mission_type": "SUPPLEMENTARY_SPRAY",
            "description": "Dynamic supplementary spray mission based on disease heatmap",
            "created_at": int(time.time() * 1e9),
            "waypoints": waypoints,
            "estimated_distance_m": planned_path.total_distance_m,
            "estimated_duration_s": planned_path.estimated_duration_s,
            "estimated_battery_used_pct": planned_path.estimated_battery_used_pct,
            "estimated_chemical_used_pct": planned_path.estimated_chemical_used_pct,
            "priority": "high",
        }

        heatmap_summary = {
            "field_id": heatmap.get("field_id"),
            "grid_size": heatmap.get("grid_size", 0),
            "total_points": heatmap.get("total_points", 0),
            "max_density": heatmap.get("max_density", 0),
            "avg_severity": heatmap.get("avg_severity", 0),
            "cells_count": len(heatmap.get("cells", [])),
        }

        params = {
            "mission": mission_plan,
            "heatmap": heatmap_summary,
        }

        success = self._send_command_fn(drone_id, "UPDATE_MISSION", params)

        if success:
            logger.info(
                f"Mission {mission_id} sent to drone {drone_id} "
                f"({len(waypoints)} waypoints)"
            )
        else:
            logger.warning(f"Failed to send mission {mission_id} to drone {drone_id}")

        return success

    def _scheduler_loop(self) -> None:
        """定时调度循环"""
        logger.info("Mission scheduler loop started")

        while self._running and not self._stop_event.is_set():
            try:
                self._stop_event.wait(self._schedule_interval_s)
                if not self._running:
                    break

                with self._drone_states_lock:
                    drone_ids = list(self._drone_states.keys())

                for drone_id in drone_ids:
                    try:
                        self.trigger_mission_for_drone(drone_id)
                    except Exception as e:
                        logger.error(f"Scheduled mission failed for {drone_id}: {e}")

            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(5.0)

        logger.info("Mission scheduler loop exited")

    def get_heatmap(self, drone_id: Optional[str] = None) -> Dict[str, Any]:
        """获取当前热力图数据"""
        return self._heatmap_gen.generate_heatmap(
            field_id=self._field_id,
            drone_id=drone_id,
            hours_back=24,
        )

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            with self._drone_states_lock:
                drones = {}
                for did, state in self._drone_states.items():
                    drones[did] = {
                        "current_mission_id": state.current_mission_id,
                        "battery_level_pct": state.battery_level_pct,
                        "chemical_level_pct": state.chemical_level_pct,
                        "latitude": state.latitude,
                        "longitude": state.longitude,
                        "pending_commands": state.pending_commands,
                        "last_status_update": state.last_status_update,
                    }

            return {
                "missions_generated": self._missions_generated,
                "commands_sent": self._commands_sent,
                "registered_drones": len(drones),
                "drones": drones,
                "field_id": self._field_id,
                "auto_schedule_enabled": self._auto_schedule_enabled,
                "schedule_interval_s": self._schedule_interval_s,
            }

    def register_drone(self, drone_id: str, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """注册无人机到调度器"""
        with self._drone_states_lock:
            if drone_id not in self._drone_states:
                state = DroneMissionState(drone_id=drone_id)
                if initial_state:
                    state.battery_level_pct = float(initial_state.get("battery_level", 100.0))
                    state.chemical_level_pct = float(initial_state.get("chemical_level", 100.0))
                    pos = initial_state.get("current_position", {})
                    if pos:
                        state.latitude = float(pos.get("latitude", 0.0))
                        state.longitude = float(pos.get("longitude", 0.0))
                self._drone_states[drone_id] = state
                logger.info(f"Drone {drone_id} registered with mission scheduler")

    def unregister_drone(self, drone_id: str) -> None:
        """注销无人机"""
        with self._drone_states_lock:
            self._drone_states.pop(drone_id, None)
        with self._frame_count_lock:
            self._frame_count_since_last_plan.pop(drone_id, None)
        logger.info(f"Drone {drone_id} unregistered from mission scheduler")

    def invalidate_heatmap_cache(self) -> None:
        """使热力图缓存失效"""
        self._heatmap_gen.invalidate_cache(self._field_id)
        logger.info("Heatmap cache invalidated")
