import os
import sys
import time
import json
import math
import logging
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


@dataclass
class DiseasePoint:
    latitude: float
    longitude: float
    severity_score: float
    severity_level: str
    confidence: float
    timestamp: int
    drone_id: str
    frame_id: int


class HeatmapGenerator:
    """
    病害热力图生成器。
    基于核密度估计（KDE）算法，将离散的病害点位转换为连续的密度热力图。
    使用高斯核函数，支持按严重程度加权。
    """

    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("heatmap", {}) if isinstance(config, dict) and "heatmap" in config else {}
        self._bandwidth_meters = float(cfg.get("bandwidth_meters", 50.0))
        self._grid_size = int(cfg.get("grid_size", 50))
        self._min_density_threshold = float(cfg.get("min_density_threshold", 0.05))
        self._severity_weight_enabled = bool(cfg.get("severity_weight_enabled", True))
        self._use_cache = bool(cfg.get("use_cache", True))
        self._cache_ttl_s = int(cfg.get("cache_ttl_s", 300))

        self._frame_log_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "frames",
        )

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()

        logger.info(
            f"HeatmapGenerator initialized: bandwidth={self._bandwidth_meters}m "
            f"grid={self._grid_size}x{self._grid_size} "
            f"severity_weight={self._severity_weight_enabled}"
        )

    def generate_heatmap(
        self,
        field_id: str,
        drone_id: Optional[str] = None,
        date_str: Optional[str] = None,
        hours_back: int = 24,
    ) -> Dict[str, Any]:
        """
        生成指定地块的病害热力图。
        返回包含网格、密度范围、平均严重度等信息的字典。
        """
        cache_key = f"{field_id}_{drone_id or 'all'}_{date_str or 'today'}_{hours_back}h"

        if self._use_cache:
            with self._cache_lock:
                cached = self._cache.get(cache_key)
                if cached and time.time() - cached["generated_at"] < self._cache_ttl_s:
                    logger.debug(f"Cache hit for heatmap {cache_key}")
                    return dict(cached["data"])

        points = self._load_disease_points(drone_id, date_str, hours_back)
        logger.info(
            f"Loaded {len(points)} disease points for heatmap generation"
        )

        if len(points) == 0:
            result = {
                "field_id": field_id,
                "generated_at": int(time.time() * 1e9),
                "grid_size": 0,
                "cells": [],
                "min_density": 0.0,
                "max_density": 0.0,
                "avg_severity": 0.0,
                "total_points": 0,
                "bounds": {"min_lat": 0, "max_lat": 0, "min_lon": 0, "max_lon": 0},
            }
            return result

        result = self._compute_kde_heatmap(points, field_id)

        if self._use_cache:
            with self._cache_lock:
                self._cache[cache_key] = {
                    "generated_at": time.time(),
                    "data": dict(result),
                }
                if len(self._cache) > 50:
                    oldest = min(self._cache.keys(), key=lambda k: self._cache[k]["generated_at"])
                    self._cache.pop(oldest, None)

        return result

    def _load_disease_points(
        self,
        drone_id: Optional[str] = None,
        date_str: Optional[str] = None,
        hours_back: int = 24,
    ) -> List[DiseasePoint]:
        """从本地 JSON 日志加载病害点位数据"""
        points: List[DiseasePoint] = []
        now = time.time()
        cutoff = now - hours_back * 3600

        if date_str:
            dates_to_scan = [date_str]
        else:
            dates_to_scan = []
            for i in range(max(1, hours_back // 24 + 1)):
                d = datetime.now() - timedelta(days=i)
                dates_to_scan.append(d.strftime("%Y%m%d"))

        for date_dir_name in dates_to_scan:
            date_dir = os.path.join(self._frame_log_dir, date_dir_name)
            if not os.path.isdir(date_dir):
                continue

            for fname in os.listdir(date_dir):
                if not fname.endswith(".json"):
                    continue

                if drone_id and not fname.startswith(drone_id):
                    continue

                fpath = os.path.join(date_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        frame_data = json.load(f)

                    ts = frame_data.get("timestamp", 0) / 1e9
                    if ts < cutoff:
                        continue

                    gps = frame_data.get("gps", {})
                    lat = gps.get("latitude", 0.0)
                    lon = gps.get("longitude", 0.0)
                    if lat == 0 and lon == 0:
                        continue

                    detections = frame_data.get("detections", [])
                    for det in detections:
                        sev_score = det.get("severity_score", 0.5)
                        sev_level = det.get("severity_level", "unknown")
                        conf = det.get("confidence", 0.5)

                        points.append(
                            DiseasePoint(
                                latitude=lat,
                                longitude=lon,
                                severity_score=sev_score,
                                severity_level=sev_level,
                                confidence=conf,
                                timestamp=frame_data.get("timestamp", 0),
                                drone_id=frame_data.get("drone_id", "unknown"),
                                frame_id=frame_data.get("frame_id", 0),
                            )
                        )
                except Exception as e:
                    logger.debug(f"Failed to load frame {fname}: {e}")
                    continue

        return points

    def _compute_kde_heatmap(
        self, points: List[DiseasePoint], field_id: str
    ) -> Dict[str, Any]:
        """
        使用高斯核函数计算 KDE 热力图。
        为了性能和准确性，将经纬度转换为米制坐标系进行计算。
        """
        n = len(points)
        if n == 0:
            return {
                "field_id": field_id,
                "generated_at": int(time.time() * 1e9),
                "grid_size": 0,
                "cells": [],
                "min_density": 0.0,
                "max_density": 0.0,
                "avg_severity": 0.0,
                "total_points": 0,
                "bounds": {"min_lat": 0, "max_lat": 0, "min_lon": 0, "max_lon": 0},
            }

        lats = np.array([p.latitude for p in points])
        lons = np.array([p.longitude for p in points])
        severities = np.array([p.severity_score for p in points])
        confidences = np.array([p.confidence for p in points])

        min_lat, max_lat = float(lats.min()), float(lats.max())
        min_lon, max_lon = float(lons.min()), float(lons.max())

        lat_range = max(max_lat - min_lat, 0.001)
        lon_range = max(max_lon - min_lon, 0.001)

        padding = 0.1
        min_lat -= lat_range * padding
        max_lat += lat_range * padding
        min_lon -= lon_range * padding
        max_lon += lon_range * padding

        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2

        def latlon_to_meters(lat: float, lon: float) -> Tuple[float, float]:
            """以中心点为原点，将经纬度转换为米（近似，小范围可用）"""
            lat_m = (lat - center_lat) * 111320.0
            lon_m = (lon - center_lon) * 111320.0 * math.cos(math.radians(center_lat))
            return lat_m, lon_m

        def meters_to_latlon(x_m: float, y_m: float) -> Tuple[float, float]:
            lat = center_lat + y_m / 111320.0
            lon = center_lon + x_m / (111320.0 * math.cos(math.radians(center_lat)))
            return lat, lon

        points_xy = np.array([latlon_to_meters(p.latitude, p.longitude) for p in points])
        points_x = points_xy[:, 1]
        points_y = points_xy[:, 0]

        min_x, max_x = float(points_x.min()), float(points_x.max())
        min_y, max_y = float(points_y.min()), float(points_y.max())
        range_x = max(max_x - min_x, self._bandwidth_meters * 4)
        range_y = max(max_y - min_y, self._bandwidth_meters * 4)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        half_range = max(range_x, range_y) / 2
        min_x = center_x - half_range
        max_x = center_x + half_range
        min_y = center_y - half_range
        max_y = center_y + half_range

        grid_size = self._grid_size
        x_grid = np.linspace(min_x, max_x, grid_size)
        y_grid = np.linspace(min_y, max_y, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)

        if self._severity_weight_enabled:
            weights = severities * confidences
        else:
            weights = np.ones(n)

        bandwidth = self._bandwidth_meters
        bandwidth_sq = bandwidth * bandwidth

        density = np.zeros((grid_size, grid_size))
        severity_weighted = np.zeros((grid_size, grid_size))

        for i in range(n):
            px, py = points_x[i], points_y[i]
            w = weights[i]
            sev = severities[i]

            dx = X - px
            dy = Y - py
            dist_sq = dx * dx + dy * dy

            kernel = np.exp(-dist_sq / (2 * bandwidth_sq))

            density += kernel * w
            severity_weighted += kernel * w * sev

        total_weight = np.sum(weights)
        if total_weight > 0:
            density /= total_weight * math.pi * bandwidth_sq

        max_density = float(np.max(density))
        min_density = float(np.min(density))

        if max_density > 0:
            density_norm = density / max_density
        else:
            density_norm = density

        avg_severity = float(np.mean(severities))

        cells = []
        for i in range(grid_size):
            for j in range(grid_size):
                d = float(density_norm[i, j])
                if d < self._min_density_threshold:
                    continue

                y_m = y_grid[i]
                x_m = x_grid[j]
                lat, lon = meters_to_latlon(x_m, y_m)

                sev_val = (
                    float(severity_weighted[i, j] / max(1e-9, density[i, j]))
                    if density[i, j] > 1e-9
                    else 0.0
                )

                cells.append(
                    {
                        "latitude": lat,
                        "longitude": lon,
                        "density": d,
                        "severity_score": min(1.0, max(0.0, sev_val)),
                    }
                )

        return {
            "field_id": field_id,
            "generated_at": int(time.time() * 1e9),
            "grid_size": grid_size,
            "cells": cells,
            "min_density": min_density,
            "max_density": max_density,
            "avg_severity": avg_severity,
            "total_points": n,
            "bounds": {
                "min_lat": min_lat,
                "max_lat": max_lat,
                "min_lon": min_lon,
                "max_lon": max_lon,
            },
        }

    def get_high_risk_zones(
        self,
        heatmap: Dict[str, Any],
        threshold_pct: float = 0.7,
        max_zones: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        从热力图中提取高风险区域（密度高于阈值的局部极大值点）。
        返回按密度降序排列的区域列表。
        """
        cells = heatmap.get("cells", [])
        if not cells:
            return []

        sorted_cells = sorted(cells, key=lambda c: c["density"], reverse=True)

        threshold = threshold_pct
        high_risk = [c for c in sorted_cells if c["density"] >= threshold]

        if not high_risk:
            top_n = min(max_zones, len(sorted_cells))
            high_risk = sorted_cells[:top_n]

        zones: List[Dict[str, Any]] = []
        min_distance_m = self._bandwidth_meters

        for cell in high_risk:
            too_close = False
            lat1, lon1 = cell["latitude"], cell["longitude"]

            for zone in zones:
                lat2, lon2 = zone["latitude"], zone["longitude"]
                dist = self._haversine_distance_m(lat1, lon1, lat2, lon2)
                if dist < min_distance_m:
                    if cell["density"] > zone["density"]:
                        zone.update(cell)
                    too_close = True
                    break

            if not too_close:
                zones.append(dict(cell))

            if len(zones) >= max_zones:
                break

        return zones

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

    def invalidate_cache(self, field_id: Optional[str] = None) -> None:
        with self._cache_lock:
            if field_id:
                keys_to_remove = [k for k in self._cache if k.startswith(field_id)]
                for k in keys_to_remove:
                    self._cache.pop(k, None)
            else:
                self._cache.clear()
            logger.info(f"Heatmap cache invalidated (field={field_id or 'all'})")
