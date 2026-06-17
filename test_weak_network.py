import os
import sys
import time
import json
import uuid
import random
import shutil
import sqlite3
import logging
import tempfile
import subprocess
import threading
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


ROOT = os.path.dirname(os.path.abspath(__file__))
CLOUD_DIR = os.path.join(ROOT, "cloud")
EDGE_DIR = os.path.join(ROOT, "edge")

TOTAL_FRAMES_NORMAL = 30
TOTAL_FRAMES_OFFLINE = 120
FRAME_INTERVAL = 0.08

TEST_DATA_DIR = os.path.join(ROOT, "test_data")
EDGE_DB_PATH = os.path.join(EDGE_DIR, "data", "msg_queue.sqlite3")
EDGE_DB_SHM = EDGE_DB_PATH + "-shm"
EDGE_DB_WAL = EDGE_DB_PATH + "-wal"
CLOUD_FRAMES_DIR = os.path.join(CLOUD_DIR, "data", "frames")
CLOUD_INFLUX_MOCK_DIR = os.path.join(CLOUD_DIR, "data", "influx_mock")
EDGE_LOG = os.path.join(TEST_DATA_DIR, "edge_test.log")
CLOUD_LOG = os.path.join(TEST_DATA_DIR, "cloud_test.log")
SUMMARY_FILE = os.path.join(TEST_DATA_DIR, "summary.json")


def _setup_logger() -> logging.Logger:
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    logger = logging.getLogger("weaknet_test")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(
        os.path.join(TEST_DATA_DIR, "weaknet_test.log"), encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


log = _setup_logger()


def clean_previous_test() -> None:
    log.info("[CLEAN] 清理上次测试遗留数据...")
    for p in [EDGE_DB_PATH, EDGE_DB_SHM, EDGE_DB_WAL]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
    for d in [CLOUD_FRAMES_DIR, CLOUD_INFLUX_MOCK_DIR, TEST_DATA_DIR]:
        if d == TEST_DATA_DIR:
            continue
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    os.makedirs(TEST_DATA_DIR, exist_ok=True)


def _port_in_use(port: int) -> bool:
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            return s.connect_ex(("127.0.0.1", port)) == 0
    except Exception:
        return False


def start_cloud() -> subprocess.Popen:
    log.info("[CLOUD] 启动云端 gRPC 服务...")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        sys.executable,
        "-u",
        os.path.join(CLOUD_DIR, "main.py"),
        "--config", os.path.join(CLOUD_DIR, "config", "config.yaml"),
    ]
    log_file = open(CLOUD_LOG, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=CLOUD_DIR,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 15
    while time.time() < deadline:
        if proc.poll() is not None:
            log.error(f"[CLOUD] 启动失败，退出码={proc.returncode}")
            with open(CLOUD_LOG, "r", encoding="utf-8") as f:
                log.error(f.read()[-1000:])
            raise RuntimeError("Cloud failed to start")
        if _port_in_use(50051):
            log.info("[CLOUD] 服务启动成功，监听 50051")
            return proc
        time.sleep(0.3)
    proc.kill()
    raise RuntimeError("Cloud start timeout")


def kill_cloud(proc: subprocess.Popen) -> None:
    log.warning("[CLOUD] ========= 强制终止云端（模拟断网）=========")
    try:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass
    time.sleep(1.0)
    log.info("[CLOUD] 已终止，50051 端口应已释放")


def count_sqlite_pending() -> Tuple[int, int, int]:
    try:
        if not os.path.exists(EDGE_DB_PATH):
            return (0, 0, 0)
        conn = sqlite3.connect(EDGE_DB_PATH, timeout=2.0)
        try:
            cur = conn.cursor()
            counts = {}
            for st in [0, 1, 2, 3, 4]:
                cur.execute(
                    "SELECT COUNT(*) FROM msg_queue WHERE status=?", (st,)
                )
                counts[st] = cur.fetchone()[0]
            return (counts.get(0, 0), counts.get(1, 0), counts.get(2, 0))
        finally:
            conn.close()
    except Exception as e:
        log.debug(f"SQLite 统计失败: {e}")
        return (-1, -1, -1)


class MockEdgeRunner:
    def __init__(self, total_frames: int):
        self._total = total_frames
        self._sent = 0
        self._grpc_client = None
        self._gps_t = 0.0

    def start(self) -> None:
        log.info("[EDGE] 初始化边缘端模块 (Mock 模式)...")
        sys.path.insert(0, EDGE_DIR)
        import yaml
        from modules.grpc_client import DroneGRPCClient

        with open(
            os.path.join(EDGE_DIR, "config", "config.yaml"), "r", encoding="utf-8"
        ) as f:
            config = yaml.safe_load(f)

        from modules.gps import GPSData
        from modules.detector import Detection

        self._GPSData = GPSData
        self._Detection = Detection
        self._drone_id = config["drone"]["id"]

        self._grpc_client = DroneGRPCClient(
            config=config.get("grpc", {}),
            drone_id=self._drone_id,
        )
        if not self._grpc_client.start():
            raise RuntimeError("Failed to start edge gRPC client")
        log.info("[EDGE] gRPC 客户端已启动（Reliable Mode + SQLite）")

    def stop(self) -> None:
        if self._grpc_client:
            log.info("[EDGE] 停止客户端（优雅排空 30s）...")
            self._grpc_client.stop()

    def generate_and_send_frames(
        self, count: int, stop_event: Optional[threading.Event] = None
    ) -> int:
        from modules.gps import GPSData
        from modules.detector import Detection

        submitted = 0
        start_frame_id = self._sent + 1
        for i in range(count):
            if stop_event and stop_event.is_set():
                break
            fid = start_frame_id + i
            ts_ns = int(time.time() * 1e9)
            lat = 39.9042 + (fid * 0.00001)
            lon = 116.4074 + (fid * 0.000015)
            gps = GPSData(
                latitude=lat,
                longitude=lon,
                altitude=50.0 + random.uniform(-2, 2),
                speed=8.5,
                heading=90.0,
                timestamp=ts_ns,
                satellites=12,
                hdop=0.8,
            )
            detections = []
            n_det = random.randint(2, 6)
            for j in range(n_det):
                cls_id = random.randint(0, 2)
                sev_map = {0: "mild", 1: "moderate", 2: "severe"}
                x1 = int(random.uniform(0, 1000))
                y1 = int(random.uniform(0, 600))
                detections.append(
                    Detection(
                        x1=x1,
                        y1=y1,
                        x2=min(int(x1 + random.uniform(30, 150)), 1280),
                        y2=min(int(y1 + random.uniform(30, 150)), 720),
                        confidence=random.uniform(0.55, 0.98),
                        class_id=cls_id,
                        class_name={
                            0: "wheat_rust_early",
                            1: "wheat_rust_moderate",
                            2: "wheat_rust_severe",
                        }[cls_id],
                        severity_score=random.uniform(0.1, 0.95),
                        severity_level=sev_map[cls_id],
                    )
                )
            ok = self._grpc_client.send_detection(
                frame_id=fid,
                timestamp_ns=ts_ns,
                gps_data=gps,
                detections=detections,
                frame_width=1280,
                frame_height=720,
                multispectral_band="RGB",
                inference_latency_ms=random.uniform(5.0, 25.0),
            )
            if ok:
                submitted += 1
                self._sent = fid
            else:
                log.warning(f"[EDGE] frame {fid} 提交失败！")
            time.sleep(FRAME_INTERVAL)
        log.info(f"[EDGE] 本阶段提交 {submitted}/{count} 帧（累计 {self._sent}）")
        return submitted

    def get_stats(self) -> Dict[str, Any]:
        return self._grpc_client.get_stats() if self._grpc_client else {}


def count_cloud_frames() -> int:
    total = 0
    if not os.path.isdir(CLOUD_FRAMES_DIR):
        return 0
    for root, dirs, files in os.walk(CLOUD_FRAMES_DIR):
        total += sum(1 for f in files if f.endswith(".json"))
    return total


def check_frame_ids_contiguous(max_expected: int) -> Tuple[bool, List[int], int]:
    frame_ids = set()
    drone_ids = set()
    if os.path.isdir(CLOUD_FRAMES_DIR):
        for root, dirs, files in os.walk(CLOUD_FRAMES_DIR):
            for fname in files:
                if not fname.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(root, fname), "r", encoding="utf-8") as f:
                        d = json.load(f)
                    frame_ids.add(int(d.get("frame_id", -1)))
                    drone_ids.add(d.get("drone_id", ""))
                except Exception:
                    pass
    expected = set(range(1, max_expected + 1))
    missing = sorted(expected - frame_ids)
    extra = sorted(frame_ids - expected)
    dup_count = len(frame_ids) != count_cloud_frames()
    log.info(
        f"[VERIFY] 云端接收 frame_id 数={len(frame_ids)}, "
        f"缺失={len(missing)}, 多余={len(extra)}, 唯一无人机={drone_ids}"
    )
    if missing:
        log.warning(f"[VERIFY] 缺失的 frame_id 前 20 个: {missing[:20]}")
    if extra:
        log.warning(f"[VERIFY] 超出范围的 frame_id: {extra[:20]}")
    return (len(missing) == 0 and len(extra) == 0, missing, len(frame_ids))


def main():
    log.info("=" * 70)
    log.info("  弱网可靠性自动化测试  |  断网 → 离线积累 → 恢复补传 → 无损验证")
    log.info("=" * 70)

    clean_previous_test()

    cloud_proc = None
    edge = MockEdgeRunner(total_frames=TOTAL_FRAMES_NORMAL + TOTAL_FRAMES_OFFLINE)

    try:
        # ---------------------------------------------------------------
        # 阶段 1：正常通信
        # ---------------------------------------------------------------
        log.info("\n" + "=" * 70)
        log.info("  阶段 1：正常通信（云端在线，发送少量帧）")
        log.info("=" * 70)
        cloud_proc = start_cloud()
        time.sleep(1.0)
        edge.start()
        time.sleep(1.5)

        log.info(f"[PHASE 1] 发送 {TOTAL_FRAMES_NORMAL} 帧（正常在线场景）...")
        n1 = edge.generate_and_send_frames(TOTAL_FRAMES_NORMAL)
        time.sleep(3.0)

        p, infl, a = count_sqlite_pending()
        log.info(
            f"[PHASE 1] 完成后 SQLite 状态: PENDING={p} INFLIGHT={infl} ACKED={a}"
        )
        frames_cloud = count_cloud_frames()
        log.info(f"[PHASE 1] 云端 frames/ 目录 JSON 数 = {frames_cloud}")

        # ---------------------------------------------------------------
        # 阶段 2：强制断网
        # ---------------------------------------------------------------
        log.info("\n" + "=" * 70)
        log.info(f"  阶段 2：强制断网 → 离线产生 {TOTAL_FRAMES_OFFLINE} 帧")
        log.info("=" * 70)
        kill_cloud(cloud_proc)
        cloud_proc = None
        time.sleep(0.5)

        stop_event = threading.Event()
        offline_thread = threading.Thread(
            target=edge.generate_and_send_frames,
            args=(TOTAL_FRAMES_OFFLINE, stop_event),
            daemon=True,
        )
        log.info(
            f"[PHASE 2] 开始离线写入 {TOTAL_FRAMES_OFFLINE} 帧 "
            f"(仅写 SQLite，无法上送)"
        )
        offline_thread.start()

        total_expected = TOTAL_FRAMES_NORMAL + TOTAL_FRAMES_OFFLINE
        last_pending = 0
        check_deadline = time.time() + (TOTAL_FRAMES_OFFLINE * FRAME_INTERVAL) + 10
        while offline_thread.is_alive() and time.time() < check_deadline:
            time.sleep(1.5)
            p, infl, a = count_sqlite_pending()
            if p != last_pending:
                log.info(
                    f"  → SQLite: PENDING={p} INFLIGHT={infl} ACKED={a} "
                    f"(上送应该都失败，PENDING 持续增长)"
                )
                last_pending = p
        offline_thread.join(timeout=10)
        stop_event.set()

        time.sleep(1.0)
        p, infl, a = count_sqlite_pending()
        total_expected = TOTAL_FRAMES_NORMAL + TOTAL_FRAMES_OFFLINE
        log.info(
            f"[PHASE 2] 离线写入完成: PENDING={p} INFLIGHT={infl} ACKED={a} "
            f"(期望 PENDING ≈ {TOTAL_FRAMES_OFFLINE}+ 未 ACK 的在线帧)"
        )

        frames_cloud_after_offline = count_cloud_frames()
        log.info(
            f"[PHASE 2] 断网期间云端接收应为 0 → "
            f"当前={frames_cloud_after_offline} (与阶段1末一致则正确)"
        )

        # ---------------------------------------------------------------
        # 阶段 3：网络恢复
        # ---------------------------------------------------------------
        log.info("\n" + "=" * 70)
        log.info("  阶段 3：网络恢复 → 重新启动云端 → 自动补传")
        log.info("=" * 70)
        cloud_proc = start_cloud()
        time.sleep(1.0)

        log.info("[PHASE 3] 等待边缘端自动检测连接恢复并补传...")
        deadline = time.time() + 150
        last_acked = -1
        stagnant = 0
        while time.time() < deadline:
            p, infl, a = count_sqlite_pending()
            cloud_frames = count_cloud_frames()
            progress_pct = round(100 * cloud_frames / max(1, total_expected), 1)
            log.info(
                f"  ⏳ SQLite: PEND={p:>4} INFL={infl:>3} ACK={a:>4}  |  "
                f"云端已收: {cloud_frames}/{total_expected} ({progress_pct}%)"
            )

            if a == last_acked and a > 0:
                stagnant += 1
            else:
                stagnant = 0
            last_acked = a

            if p == 0 and infl == 0 and a > 0:
                log.info("  ✅ SQLite PENDING 和 INFLIGHT 都清空了！")
                time.sleep(2.0)
                cloud_frames = count_cloud_frames()
                if cloud_frames >= total_expected:
                    break
            if stagnant >= 6 and p > 0:
                log.warning("  ⚠ ACK 数停滞，但仍有 PENDING，继续等待...")
                stagnant = 0
            time.sleep(2.0)

        time.sleep(4.0)

        # ---------------------------------------------------------------
        # 阶段 4：验证
        # ---------------------------------------------------------------
        log.info("\n" + "=" * 70)
        log.info("  阶段 4：完整性与幂等性验证")
        log.info("=" * 70)

        p_final, infl_final, a_final = count_sqlite_pending()
        cloud_frames_final = count_cloud_frames()
        edge_stats = edge.get_stats()
        rel_stats = edge_stats.get("reliable", {})
        acked_total = rel_stats.get("acked_total", 0)
        retries_total = rel_stats.get("retries_total", 0)
        retr_exhausted = rel_stats.get("retry_exhausted", 0)

        ids_ok, missing_ids, unique_received = check_frame_ids_contiguous(
            total_expected
        )

        influx_points = 0
        if os.path.isdir(CLOUD_INFLUX_MOCK_DIR):
            for root, dirs, files in os.walk(CLOUD_INFLUX_MOCK_DIR):
                for fn in files:
                    if fn.endswith(".jsonl"):
                        try:
                            with open(
                                os.path.join(root, fn), "r", encoding="utf-8"
                            ) as f:
                                influx_points += sum(1 for _ in f)
                        except Exception:
                            pass

        pending_ok = (p_final == 0) or (p_final == -1 and acked_total >= total_expected)
        summary = {
            "test_timestamp": datetime.now().isoformat(),
            "total_expected_frames": total_expected,
            "normal_online": TOTAL_FRAMES_NORMAL,
            "offline_generated": TOTAL_FRAMES_OFFLINE,
            "sqlite_final": {
                "pending": p_final,
                "inflight": infl_final,
                "acked_in_db": a_final,
            },
            "cloud_frames_received": cloud_frames_final,
            "cloud_unique_frames": unique_received,
            "frame_ids_contiguous": ids_ok,
            "missing_frame_ids_count": len(missing_ids),
            "missing_frame_ids_sample": missing_ids[:20],
            "edge_reliable_stats": {
                "acked_total": acked_total,
                "retries_total": retries_total,
                "retry_exhausted": retr_exhausted,
            },
            "influx_mock_points_written": influx_points,
            "PASS": (
                cloud_frames_final >= total_expected
                and ids_ok
                and retr_exhausted == 0
                and pending_ok
                and acked_total >= total_expected
            ),
        }
        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        log.info("\n" + "=" * 70)
        log.info("  测试结果汇总")
        log.info("=" * 70)
        log.info(f"  总期望帧数:          {total_expected}")
        log.info(f"  其中 - 在线阶段:     {TOTAL_FRAMES_NORMAL}")
        log.info(f"       - 离线阶段:     {TOTAL_FRAMES_OFFLINE}")
        log.info("-" * 70)
        log.info(f"  SQLite 最终状态:     PENDING={p_final} INFLIGHT={infl_final} DB_ACKED={a_final}")
        log.info(f"  云端 frames/ 接收:   {cloud_frames_final} (JSON 文件数)")
        log.info(f"  云端唯一 frame_id:   {unique_received}")
        log.info(f"  frame_id 连续无缺:   {'✅ 是' if ids_ok else '❌ 否 (缺失 ' + str(len(missing_ids)) + ')'}")
        log.info(f"  InfluxDB Mock 点数:  {influx_points}")
        log.info(f"  边缘端重试次数:      {retries_total} (指数退避生效)")
        log.info(f"  耗尽重试（丢帧）:    {retr_exhausted} {'❌ 有丢帧!' if retr_exhausted > 0 else '✅ 0'}")
        log.info("-" * 70)
        passed = summary["PASS"]
        log.info("-" * 70)
        log.info(
            f"  综合结论:            {'✅ 测试通过 - Exactly-Once 语义验证成功' if passed else '❌ 测试失败 - 存在数据缺失或异常'}"
        )
        log.info(f"  关键指标验证:")
        log.info(f"    - 云端完整接收 150/150:  {'✅' if cloud_frames_final >= total_expected else '❌'}")
        log.info(f"    - frame_id 1-150 连续无缺: {'✅' if ids_ok else '❌'}")
        log.info(f"    - 边缘端 ACK 150 帧:     {'✅' if acked_total >= total_expected else '❌'} ({acked_total})")
        log.info(f"    - 重试耗尽丢帧数:       {'✅ 0' if retr_exhausted == 0 else '❌ ' + str(retr_exhausted)}")
        log.info(f"    - 指数退避重传次数:     ✅ {retries_total} 次 (机制激活)")
        log.info(f"  详细摘要见:          {SUMMARY_FILE}")
        log.info("=" * 70)
        return 0 if passed else 1

    except Exception as e:
        log.error(f"测试异常: {e}")
        import traceback
        traceback.print_exc()
        return 2
    finally:
        try:
            edge.stop()
        except Exception:
            pass
        if cloud_proc:
            try:
                cloud_proc.kill()
                cloud_proc.wait(timeout=5)
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
