import sys
import os
import time
import json
import logging
import threading
from typing import Dict, Any, List
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2e_mission_test")


def test_heatmap_generator():
    """测试 1: KDE 热力图生成器"""
    logger.info("=" * 70)
    logger.info("TEST 1: KDE Heatmap Generator")
    logger.info("=" * 70)

    from cloud.modules.heatmap_generator import HeatmapGenerator, DiseasePoint
    import random

    config = {
        "heatmap": {
            "bandwidth_meters": 60.0,
            "grid_size": 40,
            "min_density_threshold": 0.05,
            "severity_weight_enabled": True,
            "use_cache": False,
            "cache_ttl_s": 300,
            "detection_json_dir": str(project_root / "data" / "frames"),
        }
    }
    generator = HeatmapGenerator(config=config)

    center_lat = 34.10
    center_lon = 108.95
    disease_points: List[DiseasePoint] = []

    clusters = [
        (34.101, 108.952, 0.85),
        (34.103, 108.949, 0.65),
        (34.098, 108.955, 0.75),
    ]

    point_id = 0
    for cluster_lat, cluster_lon, severity in clusters:
        for _ in range(80):
            lat = cluster_lat + random.gauss(0, 0.0005)
            lon = cluster_lon + random.gauss(0, 0.0005)
            sev = max(0.2, min(1.0, severity + random.gauss(0, 0.1)))
            disease_points.append(
                DiseasePoint(
                    latitude=lat,
                    longitude=lon,
                    severity_score=sev,
                    severity_level="high" if sev > 0.7 else "medium",
                    confidence=0.7 + random.random() * 0.3,
                    timestamp=int(time.time()) - point_id,
                    drone_id="DRONE-001",
                    frame_id=point_id,
                )
            )
            point_id += 1

    logger.info(f"Generated {len(disease_points)} disease points across 3 clusters")

    heatmap = generator._compute_kde_heatmap(
        points=disease_points,
        field_id="TEST-FIELD-001",
    )

    assert heatmap is not None, "Heatmap should not be None"
    assert heatmap["field_id"] == "TEST-FIELD-001", "Field ID mismatch"
    assert len(heatmap["cells"]) > 0, "No heatmap cells generated"
    assert heatmap["grid_size"] == 40, "Grid size mismatch"

    logger.info(f"Grid size: {heatmap['grid_size']} x {heatmap['grid_size']}")
    logger.info(f"Total cells: {len(heatmap['cells'])}")
    logger.info(f"Min density: {heatmap['min_density']:.6f}")
    logger.info(f"Max density: {heatmap['max_density']:.6f}")
    logger.info(f"Avg severity: {heatmap['avg_severity']:.4f}")

    high_risk = generator.get_high_risk_zones(
        heatmap=heatmap,
        threshold_pct=0.6,
        max_zones=10,
    )

    assert len(high_risk) > 0, "Should have at least one high-risk zone"
    logger.info(f"High-risk zones: {len(high_risk)}")
    for i, zone in enumerate(high_risk[:3]):
        logger.info(
            f"  Zone {i+1}: ({zone['latitude']:.4f}, {zone['longitude']:.4f}) "
            f"density={zone['density']:.4f}, severity={zone['severity_score']:.4f}"
        )

    logger.info("✓ TEST 1 PASSED: KDE Heatmap Generator works correctly")
    return heatmap, high_risk


def test_path_planner(high_risk_zones: List[Dict[str, Any]]):
    """测试 2: 遗传算法路径规划器"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST 2: Genetic Algorithm Path Planner")
    logger.info("=" * 70)

    from cloud.modules.path_planner import (
        GeneticPathPlanner,
        DroneState,
        TargetZone,
    )

    planner = GeneticPathPlanner({
        "population_size": 80,
        "max_generations": 150,
        "mutation_rate": 0.15,
        "crossover_rate": 0.85,
        "battery_safety_margin_pct": 15.0,
        "chemical_safety_margin_pct": 10.0,
        "battery_consumption_per_km_pct": 8.0,
        "chemical_consumption_per_km_pct": 12.0,
        "spray_width_m": 5.0,
    })

    drone_state = DroneState(
        drone_id="DRONE-001",
        latitude=34.099,
        longitude=108.950,
        altitude=50.0,
        battery_level_pct=85.0,
        chemical_level_pct=70.0,
        home_latitude=34.098,
        home_longitude=108.948,
        cruise_speed_m_s=8.0,
        spray_rate_l_per_s=0.5,
        spray_width_m=5.0,
    )

    target_zones = []
    for i, zone in enumerate(high_risk_zones[:5]):
        target_zones.append(TargetZone(
            zone_id=i,
            latitude=zone["latitude"],
            longitude=zone["longitude"],
            density=zone["density"],
            severity_score=zone["severity_score"],
            radius_m=zone.get("radius_meters", 80.0),
            priority=min(5.0, max(1.0, float(zone["severity_score"] * 5))),
        ))

    logger.info(f"Drone state: battery={drone_state.battery_level_pct}%, chemical={drone_state.chemical_level_pct}%")
    logger.info(f"Target zones: {len(target_zones)}")

    path = planner.plan(
        drone_state=drone_state,
        target_zones=target_zones,
        mission_type="SPRAY",
    )

    assert path is not None, "Path plan should not be None"
    assert len(path.waypoints) > 0, "Waypoints should not be empty"

    logger.info(f"Planned path: {len(path.waypoints)} waypoints")
    logger.info(f"Waypoints: {len(path.waypoints)}")
    logger.info(f"Total distance: {path.total_distance_m:.1f} m")
    logger.info(f"Estimated duration: {path.estimated_duration_s:.0f} s")
    logger.info(f"Estimated battery: {path.estimated_battery_used_pct:.1f}%")
    logger.info(f"Estimated chemical: {path.estimated_chemical_used_pct:.1f}%")
    logger.info(f"Return to home safe: {path.return_to_home_safe}")
    logger.info(f"Coverage score: {path.coverage_score:.3f}")

    logger.info("✓ TEST 2 PASSED: Path Planner works correctly")
    return path


def test_mission_scheduler():
    """测试 3: 动态任务调度器"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST 3: Dynamic Mission Scheduler")
    logger.info("=" * 70)

    from cloud.modules.mission_scheduler import DynamicMissionScheduler
    from cloud.modules.heatmap_generator import DiseasePoint

    commands_sent = []
    commands_lock = threading.Lock()

    def mock_send_command(drone_id: str, command_type: str, **kwargs) -> bool:
        with commands_lock:
            commands_sent.append({
                "drone_id": drone_id,
                "command_type": command_type,
                **kwargs,
            })
        logger.info(
            f"[MOCK] Sent command to {drone_id}: {command_type} "
            f"(mission={kwargs.get('mission_plan', {}).get('mission_id', 'N/A')})"
        )
        return True

    config = {
        "mission_scheduler": {
            "field_id": "TEST-FIELD-001",
            "schedule_interval_s": 60,
            "min_new_frames_for_replan": 10,
            "max_waypoints_per_mission": 80,
            "high_risk_threshold": 0.5,
            "max_high_risk_zones": 8,
            "auto_schedule_enabled": True,
            "spray_width_m": 5.0,
        },
        "heatmap": {
            "bandwidth_meters": 60.0,
            "grid_size": 40,
            "min_density_threshold": 0.05,
            "severity_weight_enabled": True,
            "use_cache": False,
            "cache_ttl_s": 60,
        },
        "path_planning": {
            "population_size": 60,
            "max_generations": 100,
            "mutation_rate": 0.15,
            "crossover_rate": 0.85,
            "battery_safety_margin_pct": 15.0,
            "chemical_safety_margin_pct": 10.0,
            "battery_consumption_per_km_pct": 8.0,
            "chemical_consumption_per_km_pct": 12.0,
        },
        "detection": {
            "json_log_dir": str(project_root / "data" / "frames"),
        },
    }

    scheduler = DynamicMissionScheduler(
        config=config,
        send_command_fn=mock_send_command,
    )

    scheduler.start()
    time.sleep(0.3)

    drone_id = "DRONE-TEST-01"
    scheduler.register_drone(drone_id)

    scheduler.update_drone_status({
        "drone_id": drone_id,
        "battery_level": 82.0,
        "chemical_level": 65.0,
        "current_position": {
            "latitude": 34.099,
            "longitude": 108.950,
            "altitude": 50.0,
        },
        "home_position": {
            "latitude": 34.098,
            "longitude": 108.948,
        },
        "cruise_speed": 8.0,
        "spray_rate": 0.5,
    })

    stats = scheduler.get_stats()
    logger.info(f"Registered drones: {stats['registered_drones']}")
    logger.info(f"Missions generated: {stats['missions_generated']}")

    import random
    center_lat = 34.100
    center_lon = 108.952

    test_points = []
    for i in range(60):
        lat = center_lat + random.gauss(0, 0.001)
        lon = center_lon + random.gauss(0, 0.001)
        test_points.append(DiseasePoint(
            latitude=lat,
            longitude=lon,
            severity_score=0.5 + random.random() * 0.5,
            severity_level="medium",
            confidence=0.8,
            timestamp=int(time.time()) - i,
            drone_id=drone_id,
            frame_id=i,
        ))

    os.makedirs(project_root / "data" / "frames", exist_ok=True)
    frames_dir = project_root / "data" / "frames" / drone_id
    os.makedirs(frames_dir, exist_ok=True)

    frame_data = {
        "drone_id": drone_id,
        "frame_id": "frame_00000",
        "timestamp": time.time(),
        "gps": {"latitude": 34.100, "longitude": 108.952, "altitude": 50.0},
        "detections": [
            {
                "class_name": "wheat_rust",
                "confidence": 0.9,
                "bbox": [100, 100, 200, 200],
                "severity": 0.7,
                "geo_lat": 34.1001,
                "geo_lon": 108.9521,
            }
            for _ in range(5)
        ],
    }

    frame_file = frames_dir / "frame_00000.json"
    with open(frame_file, "w") as f:
        json.dump(frame_data, f)

    for i in range(15):
        scheduler.notify_frame_received(drone_id, f"frame_{i:05d}", 3)

    time.sleep(1.0)

    success = scheduler.trigger_mission_for_drone(drone_id)
    logger.info(f"Manual mission trigger: {'success' if success else 'failed'}")
    time.sleep(2.0)

    with commands_lock:
        logger.info(f"Total commands sent: {len(commands_sent)}")
        for cmd in commands_sent:
            mp = cmd.get("mission_plan")
            if mp:
                logger.info(
                    f"  - {cmd['command_type']}: "
                    f"{mp.get('mission_id', '?')} "
                    f"({len(mp.get('waypoints', []))} waypoints)"
                )

    stats = scheduler.get_stats()
    logger.info(f"Scheduler stats: {json.dumps(stats, indent=2, default=str)}")

    scheduler.stop()
    time.sleep(0.3)

    logger.info("✓ TEST 3 PASSED: Mission Scheduler works correctly")
    return success


def test_protocol_serialization():
    """测试 4: Protobuf 协议序列化/反序列化"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST 4: Protocol Buffer Serialization")
    logger.info("=" * 70)

    import sys
    cloud_gen_path = str(project_root / "cloud")
    if cloud_gen_path not in sys.path:
        sys.path.insert(0, cloud_gen_path)

    from generated import drone_service_pb2 as pb2

    waypoint = pb2.Waypoint(
        waypoint_id=1,
        latitude=34.1001,
        longitude=108.9502,
        altitude=50.0,
        speed=8.0,
        action="WAYPOINT_ACTION_SPRAY",
        spray_density=0.8,
    )

    assert waypoint.waypoint_id == 1
    assert abs(waypoint.latitude - 34.1001) < 0.0001
    assert waypoint.action == "WAYPOINT_ACTION_SPRAY"

    mission = pb2.MissionPlan(
        mission_id="MISSION-TEST-001",
        mission_type="MISSION_TYPE_SPRAY",
        description="Test mission",
        estimated_distance_m=1500.0,
        estimated_duration_s=180.0,
        estimated_battery_used_pct=12.5,
        estimated_chemical_used_pct=8.3,
        priority="HIGH",
    )

    mission.waypoints.append(waypoint)
    mission.waypoints.append(waypoint)

    assert len(mission.waypoints) == 2
    assert mission.mission_id == "MISSION-TEST-001"
    assert abs(mission.estimated_battery_used_pct - 12.5) < 0.01

    cell = pb2.HeatmapCell(
        latitude=34.100,
        longitude=108.950,
        density=0.75,
        severity_score=0.82,
    )

    heatmap = pb2.HeatmapData(
        field_id="FIELD-TEST-001",
        grid_size=40,
        cells=[cell, cell, cell],
        max_density=1.0,
        avg_severity=0.65,
    )

    assert heatmap.field_id == "FIELD-TEST-001"
    assert len(heatmap.cells) == 3
    assert abs(heatmap.cells[0].density - 0.75) < 0.01

    cmd = pb2.ServerCommand(
        command_id="cmd-test-001",
        command_type="UPDATE_MISSION",
        mission_plan=mission,
        heatmap=heatmap,
    )

    assert cmd.command_type == "UPDATE_MISSION"
    assert cmd.mission_plan.mission_id == "MISSION-TEST-001"
    assert cmd.heatmap.field_id == "FIELD-TEST-001"

    serialized = cmd.SerializeToString()
    assert len(serialized) > 0

    parsed = pb2.ServerCommand()
    parsed.ParseFromString(serialized)

    assert parsed.command_id == "cmd-test-001"
    assert parsed.command_type == "UPDATE_MISSION"
    assert parsed.mission_plan.mission_id == "MISSION-TEST-001"
    assert len(parsed.mission_plan.waypoints) == 2
    assert parsed.heatmap.field_id == "FIELD-TEST-001"

    logger.info(f"Serialized size: {len(serialized)} bytes")
    logger.info(f"Waypoints: {len(parsed.mission_plan.waypoints)}")
    logger.info(f"Heatmap cells: {len(parsed.heatmap.cells)}")

    logger.info("✓ TEST 4 PASSED: Protocol serialization works correctly")


def test_edge_client_parsing():
    """测试 5: 云端-边缘端协议往返（云端构建 + proto 序列化 + 解析验证）"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST 5: Cloud-Edge Protocol Round-trip")
    logger.info("=" * 70)

    import sys
    cloud_path = str(project_root / "cloud")
    if cloud_path not in sys.path:
        sys.path.insert(0, cloud_path)
    cloud_gen_path = str(project_root / "cloud" / "generated")
    if cloud_gen_path not in sys.path:
        sys.path.insert(0, cloud_gen_path)

    from generated import drone_service_pb2 as pb2
    from cloud.modules.grpc_server import DroneDetectionServicer

    mission_dict = {
        "mission_id": "MISSION-ROUNDTRIP-001",
        "mission_type": "MISSION_TYPE_SPRAY",
        "description": "Round-trip test mission",
        "waypoints": [
            {
                "waypoint_id": 1,
                "latitude": 34.1001,
                "longitude": 108.9502,
                "altitude": 50.0,
                "speed": 8.0,
                "action": "WAYPOINT_ACTION_SPRAY",
                "spray_density": 0.8,
            },
            {
                "waypoint_id": 2,
                "latitude": 34.1005,
                "longitude": 108.9508,
                "altitude": 50.0,
                "speed": 8.0,
                "action": "WAYPOINT_ACTION_SPRAY",
                "spray_density": 0.6,
            },
            {
                "waypoint_id": 3,
                "latitude": 34.1010,
                "longitude": 108.9512,
                "altitude": 50.0,
                "speed": 8.0,
                "action": "WAYPOINT_ACTION_GOTO",
                "spray_density": 0.9,
            },
        ],
        "estimated_distance_m": 1200.0,
        "estimated_duration_s": 150.0,
        "estimated_battery_used_pct": 10.0,
        "estimated_chemical_used_pct": 7.5,
        "priority": "HIGH",
        "created_at": int(time.time() * 1e9),
    }

    mock_servicer = DroneDetectionServicer.__new__(DroneDetectionServicer)
    mp_proto = mock_servicer._build_mission_plan_proto(mission_dict)

    assert mp_proto.mission_id == mission_dict["mission_id"]
    assert len(mp_proto.waypoints) == 3
    assert mp_proto.waypoints[0].latitude == mission_dict["waypoints"][0]["latitude"]
    assert mp_proto.priority == "HIGH"

    logger.info(f"Cloud dict -> Proto: mission_id={mp_proto.mission_id}, waypoints={len(mp_proto.waypoints)}")

    serialized = mp_proto.SerializeToString()
    logger.info(f"Serialized size: {len(serialized)} bytes")

    parsed_proto = pb2.MissionPlan()
    parsed_proto.ParseFromString(serialized)

    assert parsed_proto.mission_id == mission_dict["mission_id"]
    assert len(parsed_proto.waypoints) == 3
    assert abs(parsed_proto.waypoints[1].longitude - mission_dict["waypoints"][1]["longitude"]) < 0.0001
    assert abs(parsed_proto.estimated_battery_used_pct - mission_dict["estimated_battery_used_pct"]) < 0.01

    parsed_dict = {
        "mission_id": parsed_proto.mission_id,
        "mission_type": parsed_proto.mission_type,
        "description": parsed_proto.description,
        "created_at": parsed_proto.created_at,
        "waypoints": [
            {
                "waypoint_id": wp.waypoint_id,
                "latitude": wp.latitude,
                "longitude": wp.longitude,
                "altitude": wp.altitude,
                "speed": wp.speed,
                "action": wp.action,
                "spray_density": wp.spray_density,
                "estimated_arrival": wp.estimated_arrival,
            }
            for wp in parsed_proto.waypoints
        ],
        "estimated_distance_m": parsed_proto.estimated_distance_m,
        "estimated_duration_s": parsed_proto.estimated_duration_s,
        "estimated_battery_used_pct": parsed_proto.estimated_battery_used_pct,
        "estimated_chemical_used_pct": parsed_proto.estimated_chemical_used_pct,
        "priority": parsed_proto.priority,
    }

    assert parsed_dict["mission_id"] == mission_dict["mission_id"]
    assert len(parsed_dict["waypoints"]) == 3
    assert abs(parsed_dict["waypoints"][0]["spray_density"] - mission_dict["waypoints"][0]["spray_density"]) < 0.001
    assert parsed_dict["priority"] == "HIGH"

    logger.info(f"Proto -> Edge dict: mission_id={parsed_dict['mission_id']}, waypoints={len(parsed_dict['waypoints'])}")
    for i, wp in enumerate(parsed_dict["waypoints"]):
        logger.info(
            f"  WP{i}: ({wp['latitude']:.4f}, {wp['longitude']:.4f}) "
            f"action={wp['action']} density={wp['spray_density']:.2f}"
        )

    heatmap_dict = {
        "field_id": "FIELD-ROUNDTRIP-001",
        "grid_size": 40,
        "cells": [
            {"latitude": 34.100, "longitude": 108.950, "density": 0.75, "severity_score": 0.82},
            {"latitude": 34.101, "longitude": 108.951, "density": 0.95, "severity_score": 0.90},
            {"latitude": 34.102, "longitude": 108.952, "density": 0.55, "severity_score": 0.60},
        ],
        "min_density": 0.0,
        "max_density": 1.0,
        "avg_severity": 0.65,
        "generated_at": int(time.time() * 1e9),
    }

    hm_proto = mock_servicer._build_heatmap_proto(heatmap_dict)
    assert hm_proto.field_id == heatmap_dict["field_id"]
    assert len(hm_proto.cells) == 3

    hm_serialized = hm_proto.SerializeToString()
    logger.info(f"Heatmap serialized size: {len(hm_serialized)} bytes ({len(hm_proto.cells)} cells)")

    hm_parsed = pb2.HeatmapData()
    hm_parsed.ParseFromString(hm_serialized)
    assert hm_parsed.field_id == heatmap_dict["field_id"]
    assert len(hm_parsed.cells) == 3
    assert abs(hm_parsed.cells[0].density - heatmap_dict["cells"][0]["density"]) < 0.001

    logger.info(f"Heatmap round-trip: field_id={hm_parsed.field_id}, cells={len(hm_parsed.cells)}")

    logger.info("✓ TEST 5 PASSED: Cloud-Edge protocol round-trip works correctly")
    return True


def main():
    logger.info("")
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║" + " " * 10 + "CLOUD DYNAMIC MISSION SCHEDULER - E2E TEST" + " " * 13 + "║")
    logger.info("╚" + "═" * 68 + "╝")
    logger.info("")

    all_passed = True

    try:
        heatmap, high_risk = test_heatmap_generator()
    except Exception as e:
        logger.error(f"✗ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
        return

    try:
        test_path_planner(high_risk)
    except Exception as e:
        logger.error(f"✗ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        test_mission_scheduler()
    except Exception as e:
        logger.error(f"✗ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        test_protocol_serialization()
    except Exception as e:
        logger.error(f"✗ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        test_edge_client_parsing()
    except Exception as e:
        logger.error(f"✗ TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    logger.info("")
    logger.info("=" * 70)
    if all_passed:
        logger.info("✓ ALL TESTS PASSED! ✓")
    else:
        logger.error("✗ SOME TESTS FAILED!")
    logger.info("=" * 70)
    logger.info("")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
