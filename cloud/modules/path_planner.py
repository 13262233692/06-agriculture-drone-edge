import os
import sys
import time
import math
import random
import logging
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from copy import deepcopy

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


@dataclass
class Waypoint:
    wp_id: int
    latitude: float
    longitude: float
    altitude: float = 50.0
    speed: float = 8.0
    action: str = "SPRAY"
    spray_density: float = 1.0
    estimated_arrival: int = 0


@dataclass
class DroneState:
    drone_id: str
    latitude: float
    longitude: float
    altitude: float = 50.0
    battery_level_pct: float = 100.0
    chemical_level_pct: float = 100.0
    cruise_speed_m_s: float = 8.0
    spray_rate_l_per_s: float = 0.5
    home_latitude: float = 0.0
    home_longitude: float = 0.0
    spray_width_m: float = 5.0


@dataclass
class TargetZone:
    zone_id: int
    latitude: float
    longitude: float
    density: float
    severity_score: float
    radius_m: float = 30.0
    priority: float = 1.0


@dataclass
class PlannedPath:
    waypoints: List[Waypoint] = field(default_factory=list)
    total_distance_m: float = 0.0
    estimated_duration_s: float = 0.0
    estimated_battery_used_pct: float = 0.0
    estimated_chemical_used_pct: float = 0.0
    coverage_score: float = 0.0
    return_to_home_safe: bool = False
    fitness: float = 0.0


class GeneticPathPlanner:
    """
    基于遗传算法的补喷路径规划器。
    目标：在电量和药量约束下，最大化高风险病害区覆盖度，保证安全返航。

    遗传算法要素：
    - 编码：访问目标区域的顺序（排列编码）
    - 适应度：覆盖权重 × 覆盖度 - 距离惩罚 - 约束违反惩罚
    - 选择：锦标赛选择
    - 交叉：顺序交叉 (OX)
    - 变异：交换变异 + 逆序变异
    - 约束处理：电量不足 / 药量不足时截断路径，强制返航
    """

    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("path_planning", {}) if isinstance(config, dict) and "path_planning" in config else {}

        self._population_size = int(cfg.get("population_size", 100))
        self._max_generations = int(cfg.get("max_generations", 200))
        self._mutation_rate = float(cfg.get("mutation_rate", 0.15))
        self._crossover_rate = float(cfg.get("crossover_rate", 0.85))
        self._tournament_size = int(cfg.get("tournament_size", 5))
        self._elitism_count = int(cfg.get("elitism_count", 3))

        self._battery_safety_margin_pct = float(cfg.get("battery_safety_margin_pct", 15.0))
        self._chemical_safety_margin_pct = float(cfg.get("chemical_safety_margin_pct", 10.0))

        self._battery_consumption_per_km = float(cfg.get("battery_consumption_per_km_pct", 8.0))
        self._chemical_consumption_per_km = float(cfg.get("chemical_consumption_per_km_pct", 12.0))

        self._waypoint_spacing_m = float(cfg.get("waypoint_spacing_m", 20.0))

        self._coverage_weight = float(cfg.get("coverage_weight", 100.0))
        self._distance_weight = float(cfg.get("distance_weight", 0.01))
        self._constraint_penalty = float(cfg.get("constraint_penalty", 1000.0))

        self._min_targets_for_ga = int(cfg.get("min_targets_for_ga", 4))

        self._random = random.Random(42)

        logger.info(
            f"GeneticPathPlanner initialized: "
            f"pop={self._population_size} gens={self._max_generations} "
            f"mut_rate={self._mutation_rate} "
            f"battery_margin={self._battery_safety_margin_pct}% "
            f"chemical_margin={self._chemical_safety_margin_pct}%"
        )

    def plan(
        self,
        drone_state: DroneState,
        target_zones: List[TargetZone],
        mission_type: str = "SUPPLEMENTARY_SPRAY",
    ) -> PlannedPath:
        """
        规划补喷路径。
        输入：无人机状态 + 目标区域列表
        输出：规划好的路径（含航点、资源消耗估计、是否安全）
        """
        n_targets = len(target_zones)
        if n_targets == 0:
            return self._create_empty_path(drone_state)

        logger.info(
            f"Planning path: {n_targets} target zones, "
            f"battery={drone_state.battery_level_pct:.1f}%, "
            f"chemical={drone_state.chemical_level_pct:.1f}%"
        )

        if n_targets <= self._min_targets_for_ga:
            best_path = self._greedy_nearest_neighbor(drone_state, target_zones)
        else:
            best_path = self._run_genetic_algorithm(drone_state, target_zones)

        best_path = self._enforce_safety_constraints(best_path, drone_state, target_zones)

        detailed_path = self._generate_detailed_waypoints(
            best_path, drone_state, target_zones
        )

        logger.info(
            f"Path planned: {len(detailed_path.waypoints)} waypoints, "
            f"distance={detailed_path.total_distance_m:.0f}m, "
            f"battery={detailed_path.estimated_battery_used_pct:.1f}%, "
            f"chemical={detailed_path.estimated_chemical_used_pct:.1f}%, "
            f"safe_rtl={detailed_path.return_to_home_safe}"
        )

        return detailed_path

    def _run_genetic_algorithm(
        self, drone_state: DroneState, targets: List[TargetZone]
    ) -> PlannedPath:
        """运行遗传算法寻找最优路径"""
        n = len(targets)

        population = self._initialize_population(n, drone_state, targets)

        best_overall = None
        best_fitness = float("-inf")

        for gen in range(self._max_generations):
            fitnesses = [
                self._evaluate_fitness(ind, drone_state, targets) for ind in population
            ]

            max_idx = int(np.argmax(fitnesses))
            if fitnesses[max_idx] > best_fitness:
                best_fitness = fitnesses[max_idx]
                best_overall = deepcopy(population[max_idx])

            new_population = []

            sorted_indices = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
            for i in range(self._elitism_count):
                new_population.append(deepcopy(population[sorted_indices[i]]))

            while len(new_population) < self._population_size:
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)

                if self._random.random() < self._crossover_rate:
                    child1, child2 = self._order_crossover(parent1, parent2)
                else:
                    child1, child2 = deepcopy(parent1), deepcopy(parent2)

                if self._random.random() < self._mutation_rate:
                    child1 = self._swap_mutation(child1)
                if self._random.random() < self._mutation_rate:
                    child2 = self._swap_mutation(child2)

                new_population.append(child1)
                if len(new_population) < self._population_size:
                    new_population.append(child2)

            population = new_population

            if gen % 50 == 0:
                avg_fitness = sum(fitnesses) / len(fitnesses)
                logger.debug(f"  Gen {gen}: best_fitness={best_fitness:.2f}, avg_fitness={avg_fitness:.2f}")

        if best_overall is None:
            return self._greedy_nearest_neighbor(drone_state, targets)

        return self._decode_path(best_overall, drone_state, targets)

    def _initialize_population(
        self, n: int, drone_state: DroneState, targets: List[TargetZone]
    ) -> List[List[int]]:
        """初始化种群，包含贪心解和随机解"""
        population: List[List[int]] = []

        greedy = self._greedy_solution(targets, drone_state)
        population.append(greedy)

        for _ in range(self._population_size - 1):
            perm = list(range(n))
            self._random.shuffle(perm)
            population.append(perm)

        return population

    def _greedy_solution(
        self, targets: List[TargetZone], drone_state: DroneState
    ) -> List[int]:
        """最近邻贪心解作为初始种群的一部分"""
        n = len(targets)
        unvisited = list(range(n))
        path: List[int] = []
        current_lat, current_lon = drone_state.latitude, drone_state.longitude

        while unvisited:
            nearest_idx = None
            nearest_dist = float("inf")

            for idx in unvisited:
                t = targets[idx]
                d = self._haversine_distance_m(current_lat, current_lon, t.latitude, t.longitude)
                priority_weight = 1.0 / max(0.1, t.priority)
                weighted_dist = d * priority_weight

                if weighted_dist < nearest_dist:
                    nearest_dist = weighted_dist
                    nearest_idx = idx

            path.append(nearest_idx)
            unvisited.remove(nearest_idx)
            current_lat = targets[nearest_idx].latitude
            current_lon = targets[nearest_idx].longitude

        return path

    def _greedy_nearest_neighbor(
        self, drone_state: DroneState, targets: List[TargetZone]
    ) -> PlannedPath:
        """贪心最近邻算法（小规模问题直接用）"""
        order = self._greedy_solution(targets, drone_state)
        return self._decode_path(order, drone_state, targets)

    def _tournament_selection(
        self, population: List[List[int]], fitnesses: List[float]
    ) -> List[int]:
        """锦标赛选择"""
        candidates = self._random.sample(range(len(population)), self._tournament_size)
        best = max(candidates, key=lambda i: fitnesses[i])
        return deepcopy(population[best])

    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """顺序交叉 (Order Crossover, OX)"""
        n = len(parent1)
        if n <= 1:
            return deepcopy(parent1), deepcopy(parent2)

        start = self._random.randint(0, n - 2)
        end = self._random.randint(start + 1, n - 1)

        def ox_crossover(p1: List[int], p2: List[int]) -> List[int]:
            child = [-1] * n
            child[start : end + 1] = p1[start : end + 1]

            remaining = [x for x in p2 if x not in child]
            ptr = 0
            for i in range(n):
                if child[i] == -1:
                    child[i] = remaining[ptr]
                    ptr += 1
            return child

        return ox_crossover(parent1, parent2), ox_crossover(parent2, parent1)

    def _swap_mutation(self, individual: List[int]) -> List[int]:
        """交换变异（两个随机位置互换）"""
        n = len(individual)
        if n <= 1:
            return individual

        i, j = self._random.sample(range(n), 2)
        individual[i], individual[j] = individual[j], individual[i]

        if self._random.random() < 0.3:
            start = self._random.randint(0, n - 2)
            end = self._random.randint(start + 1, n - 1)
            individual[start : end + 1] = reversed(individual[start : end + 1])

        return individual

    def _evaluate_fitness(
        self,
        individual: List[int],
        drone_state: DroneState,
        targets: List[TargetZone],
    ) -> float:
        """计算适应度：覆盖度 - 距离惩罚 - 约束违反惩罚"""
        path = self._decode_path(individual, drone_state, targets)

        coverage_score = path.coverage_score
        total_distance = path.total_distance_m

        battery_penalty = 0.0
        chemical_penalty = 0.0
        if not path.return_to_home_safe:
            battery_penalty = self._constraint_penalty
            chemical_penalty = self._constraint_penalty

        battery_shortage = max(
            0,
            path.estimated_battery_used_pct
            - (drone_state.battery_level_pct - self._battery_safety_margin_pct),
        )
        chemical_shortage = max(
            0,
            path.estimated_chemical_used_pct
            - (drone_state.chemical_level_pct - self._chemical_safety_margin_pct),
        )

        if battery_shortage > 0:
            battery_penalty += battery_shortage * self._constraint_penalty
        if chemical_shortage > 0:
            chemical_penalty += chemical_shortage * self._constraint_penalty

        fitness = (
            self._coverage_weight * coverage_score
            - self._distance_weight * total_distance
            - battery_penalty
            - chemical_penalty
        )

        return fitness

    def _decode_path(
        self,
        order: List[int],
        drone_state: DroneState,
        targets: List[TargetZone],
    ) -> PlannedPath:
        """
        将排列编码解码为完整路径（含返航），并计算各指标。
        路径：当前位置 → 目标1 → 目标2 → ... → 目标N → 返航点
        """
        if not order:
            return self._create_empty_path(drone_state)

        waypoints: List[Waypoint] = []
        total_distance = 0.0
        coverage_score = 0.0

        current_lat, current_lon = drone_state.latitude, drone_state.longitude

        for idx, target_idx in enumerate(order):
            target = targets[target_idx]

            dist = self._haversine_distance_m(
                current_lat, current_lon, target.latitude, target.longitude
            )
            total_distance += dist

            wp = Waypoint(
                wp_id=idx,
                latitude=target.latitude,
                longitude=target.longitude,
                altitude=drone_state.altitude,
                speed=drone_state.cruise_speed_m_s,
                action="SPRAY",
                spray_density=target.severity_score,
            )
            waypoints.append(wp)

            coverage_score += target.density * target.priority * target.severity_score

            current_lat = target.latitude
            current_lon = target.longitude

        home_lat = drone_state.home_latitude or drone_state.latitude
        home_lon = drone_state.home_longitude or drone_state.longitude

        return_distance = self._haversine_distance_m(
            current_lat, current_lon, home_lat, home_lon
        )
        total_distance += return_distance

        total_distance_km = total_distance / 1000.0
        battery_used_pct = total_distance_km * self._battery_consumption_per_km
        chemical_used_pct = total_distance_km * self._chemical_consumption_per_km

        battery_available = drone_state.battery_level_pct - self._battery_safety_margin_pct
        chemical_available = drone_state.chemical_level_pct - self._chemical_safety_margin_pct

        return_to_home_safe = (
            battery_used_pct <= battery_available
            and chemical_used_pct <= chemical_available
        )

        return PlannedPath(
            waypoints=waypoints,
            total_distance_m=total_distance,
            estimated_duration_s=total_distance / max(0.1, drone_state.cruise_speed_m_s),
            estimated_battery_used_pct=battery_used_pct,
            estimated_chemical_used_pct=chemical_used_pct,
            coverage_score=coverage_score,
            return_to_home_safe=return_to_home_safe,
            fitness=0.0,
        )

    def _enforce_safety_constraints(
        self,
        path: PlannedPath,
        drone_state: DroneState,
        targets: List[TargetZone],
    ) -> PlannedPath:
        """
        强制安全约束：如果电量/药量不足，截断路径，保证能安全返航。
        从路径末端向前逐步移除目标，直到约束满足。
        """
        if path.return_to_home_safe:
            return path

        n = len(path.waypoints)
        if n == 0:
            return self._create_empty_path(drone_state)

        battery_available = drone_state.battery_level_pct - self._battery_safety_margin_pct
        chemical_available = drone_state.chemical_level_pct - self._chemical_safety_margin_pct

        best_truncated = None
        best_coverage = 0.0

        current_lat, current_lon = drone_state.latitude, drone_state.longitude
        cumulative_dist = 0.0
        cumulative_coverage = 0.0

        for i in range(n):
            wp = path.waypoints[i]

            dist = self._haversine_distance_m(current_lat, current_lon, wp.latitude, wp.longitude)
            cumulative_dist += dist

            target_idx = next(
                (j for j, t in enumerate(targets)
                 if abs(t.latitude - wp.latitude) < 1e-7 and abs(t.longitude - wp.longitude) < 1e-7),
                0,
            )
            if target_idx < len(targets):
                cumulative_coverage += targets[target_idx].density * targets[target_idx].priority

            home_lat = drone_state.home_latitude or drone_state.latitude
            home_lon = drone_state.home_longitude or drone_state.longitude
            return_dist = self._haversine_distance_m(wp.latitude, wp.longitude, home_lat, home_lon)
            total_dist = cumulative_dist + return_dist

            total_dist_km = total_dist / 1000.0
            battery_used = total_dist_km * self._battery_consumption_per_km
            chemical_used = total_dist_km * self._chemical_consumption_per_km

            if battery_used <= battery_available and chemical_used <= chemical_available:
                if cumulative_coverage > best_coverage:
                    best_coverage = cumulative_coverage
                    best_truncated = PlannedPath(
                        waypoints=path.waypoints[: i + 1],
                        total_distance_m=total_dist,
                        estimated_duration_s=total_dist / max(0.1, drone_state.cruise_speed_m_s),
                        estimated_battery_used_pct=battery_used,
                        estimated_chemical_used_pct=chemical_used,
                        coverage_score=cumulative_coverage,
                        return_to_home_safe=True,
                        fitness=0.0,
                    )

            current_lat = wp.latitude
            current_lon = wp.longitude

        if best_truncated is not None:
            logger.debug(
                f"Truncated path for safety: {len(best_truncated.waypoints)}/{n} waypoints retained"
            )
            return best_truncated

        return self._create_empty_path(drone_state)

    def _generate_detailed_waypoints(
        self,
        path: PlannedPath,
        drone_state: DroneState,
        targets: List[TargetZone],
    ) -> PlannedPath:
        """
        将目标区域级的路径细化为密集的航点路径（用于实际飞行）。
        在每个目标区域周围生成喷洒覆盖航点（之字形/环绕式）。
        """
        if not path.waypoints:
            return self._create_empty_path(drone_state)

        detailed_wps: List[Waypoint] = []
        wp_id = 0
        total_distance = 0.0
        total_chemical = 0.0
        total_time = 0.0

        current_lat, current_lon = drone_state.latitude, drone_state.longitude

        for target_wp in path.waypoints:
            travel_dist = self._haversine_distance_m(
                current_lat, current_lon, target_wp.latitude, target_wp.longitude
            )
            total_distance += travel_dist
            total_time += travel_dist / max(0.1, drone_state.cruise_speed_m_s)

            target = next(
                (t for t in targets
                 if abs(t.latitude - target_wp.latitude) < 1e-7
                 and abs(t.longitude - target_wp.longitude) < 1e-7),
                None,
            )

            radius = target.radius_m if target else 30.0
            severity = target.severity_score if target else 0.5

            spray_wps = self._generate_spray_pattern(
                target_wp.latitude,
                target_wp.longitude,
                radius,
                drone_state.spray_width_m,
                drone_state.altitude,
                drone_state.cruise_speed_m_s,
                severity,
                wp_id,
            )

            for swp in spray_wps:
                d = self._haversine_distance_m(current_lat, current_lon, swp.latitude, swp.longitude)
                total_distance += d
                total_time += d / max(0.1, swp.speed)
                total_chemical += d * drone_state.spray_rate_l_per_s / max(0.1, swp.speed)
                detailed_wps.append(swp)
                current_lat, current_lon = swp.latitude, swp.longitude
                wp_id = swp.wp_id + 1

        home_lat = drone_state.home_latitude or drone_state.latitude
        home_lon = drone_state.home_longitude or drone_state.longitude

        return_dist = self._haversine_distance_m(current_lat, current_lon, home_lat, home_lon)
        total_distance += return_dist
        total_time += return_dist / max(0.1, drone_state.cruise_speed_m_s)

        total_dist_km = total_distance / 1000.0
        battery_used_pct = total_dist_km * self._battery_consumption_per_km
        total_chemical_pct = (
            total_chemical / max(1.0, drone_state.spray_rate_l_per_s * 3600)
        ) * 100.0

        home_wp = Waypoint(
            wp_id=wp_id,
            latitude=home_lat,
            longitude=home_lon,
            altitude=drone_state.altitude,
            speed=drone_state.cruise_speed_m_s,
            action="RTL",
            spray_density=0.0,
        )
        detailed_wps.append(home_wp)

        return PlannedPath(
            waypoints=detailed_wps,
            total_distance_m=total_distance,
            estimated_duration_s=total_time,
            estimated_battery_used_pct=battery_used_pct,
            estimated_chemical_used_pct=total_chemical_pct,
            coverage_score=path.coverage_score,
            return_to_home_safe=path.return_to_home_safe,
            fitness=path.fitness,
        )

    def _generate_spray_pattern(
        self,
        center_lat: float,
        center_lon: float,
        radius_m: float,
        spray_width_m: float,
        altitude: float,
        speed: float,
        severity: float,
        start_wp_id: int,
    ) -> List[Waypoint]:
        """
        在目标区域生成之字形喷洒覆盖路径。
        简单实现：两条垂直的穿越线（十字形），覆盖核心区域。
        严重程度越高，喷洒密度越高（增加环绕圈数）。
        """
        waypoints: List[Waypoint] = []
        wp_id = start_wp_id

        R = 6371000.0
        d_lat = (radius_m / R) * (180.0 / math.pi)
        d_lon = (radius_m / R) * (180.0 / math.pi) / math.cos(math.radians(center_lat))

        num_passes = max(1, int(severity * 4))
        pass_spacing = (2 * radius_m) / max(1, num_passes)

        for i in range(num_passes + 1):
            offset_m = -radius_m + i * pass_spacing
            offset_lat = (offset_m / R) * (180.0 / math.pi)
            offset_lon = (offset_m / R) * (180.0 / math.pi) / math.cos(math.radians(center_lat))

            if i % 2 == 0:
                wp1 = Waypoint(
                    wp_id=wp_id,
                    latitude=center_lat + d_lat,
                    longitude=center_lon + offset_lon,
                    altitude=altitude,
                    speed=speed,
                    action="SPRAY",
                    spray_density=severity,
                )
                wp_id += 1
                wp2 = Waypoint(
                    wp_id=wp_id,
                    latitude=center_lat - d_lat,
                    longitude=center_lon + offset_lon,
                    altitude=altitude,
                    speed=speed,
                    action="SPRAY",
                    spray_density=severity,
                )
                wp_id += 1
            else:
                wp1 = Waypoint(
                    wp_id=wp_id,
                    latitude=center_lat - d_lat,
                    longitude=center_lon + offset_lon,
                    altitude=altitude,
                    speed=speed,
                    action="SPRAY",
                    spray_density=severity,
                )
                wp_id += 1
                wp2 = Waypoint(
                    wp_id=wp_id,
                    latitude=center_lat + d_lat,
                    longitude=center_lon + offset_lon,
                    altitude=altitude,
                    speed=speed,
                    action="SPRAY",
                    spray_density=severity,
                )
                wp_id += 1

            waypoints.append(wp1)
            waypoints.append(wp2)

        return waypoints

    def _create_empty_path(self, drone_state: DroneState) -> PlannedPath:
        """创建空路径（直接返航）"""
        home_lat = drone_state.home_latitude or drone_state.latitude
        home_lon = drone_state.home_longitude or drone_state.longitude

        rtl_wp = Waypoint(
            wp_id=0,
            latitude=home_lat,
            longitude=home_lon,
            altitude=drone_state.altitude,
            speed=drone_state.cruise_speed_m_s,
            action="RTL",
            spray_density=0.0,
        )

        distance = self._haversine_distance_m(
            drone_state.latitude, drone_state.longitude, home_lat, home_lon
        )

        return PlannedPath(
            waypoints=[rtl_wp],
            total_distance_m=distance,
            estimated_duration_s=distance / max(0.1, drone_state.cruise_speed_m_s),
            estimated_battery_used_pct=(distance / 1000.0) * self._battery_consumption_per_km,
            estimated_chemical_used_pct=0.0,
            coverage_score=0.0,
            return_to_home_safe=True,
            fitness=0.0,
        )

    def _haversine_distance_m(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Haversine 公式计算两点间距离（米）"""
        R = 6371000.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def set_seed(self, seed: int) -> None:
        self._random = random.Random(seed)
