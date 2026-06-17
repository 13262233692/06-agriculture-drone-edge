import os
import sys
import time
import json
import uuid
import threading
import logging
import traceback
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from queue import Queue, Empty


logger = logging.getLogger(__name__)


@dataclass
class InflightEntry:
    msg_id: str
    ack_nonce: str
    frame_id: int
    sent_at: int
    deadline_at: int
    proto_bytes: bytes
    payload_json: str


class ReliableTransportConfig:
    """可靠传输参数配置（兼容两套键名：标准名 + config.yaml 短名）"""

    def __init__(self, cfg: Dict[str, Any]):
        ack_timeout_s = cfg.get(
            "ack_timeout_s", cfg.get("ack_timeout_ms", 3000) / 1000
        )
        self.ack_timeout_ns = int(ack_timeout_s * 1_000_000_000)

        self.backoff_base_ms = int(
            cfg.get("retry_backoff_base_ms", cfg.get("backoff_base_ms", 500))
        )
        self.max_backoff_ms = int(
            cfg.get("retry_backoff_max_ms", cfg.get("max_backoff_ms", 60000))
        )
        self.max_retries = int(cfg.get("max_retries", 50))
        self.max_inflight = int(
            cfg.get("inflight_window_size", cfg.get("max_inflight", 64))
        )
        self.batch_size = int(
            cfg.get("dequeue_batch_size", cfg.get("batch_size", 16))
        )
        self.flush_interval_ms = int(cfg.get("flush_interval_ms", 50))
        self.resend_stuck_interval_ms = int(
            cfg.get("resend_stuck_interval_ms", 10000)
        )
        self.expire_interval_s = int(
            cfg.get("maintenance_scan_interval_s", cfg.get("expire_interval_s", 300))
        )
        self.heartbeat_interval_s = int(cfg.get("heartbeat_interval_s", 15))


class ReliableGRPCTransport:
    """
    可靠 gRPC 双向流传输层。
    职责：
      1) 拦截上层 send 调用 → 写入持久化队列
      2) 从队列批量拉取 → 发送到 gRPC → 加入 inflight
      3) 监听 ACK 回包 → 命中 inflight → 标记 ACKED
      4) 超时扫描 → 未收到 ACK 的 → 标记重试（指数退避）
      5) 周期清理 + 心跳
    """

    def __init__(
        self,
        persistence_queue,
        config: ReliableTransportConfig,
        drone_id: str,
        proto_frame_class,
        proto_ack_class,
        send_fn: Callable[[bytes], bool],
        stats_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self._queue = persistence_queue
        self._cfg = config
        self._drone_id = drone_id
        self._proto_frame_cls = proto_frame_class
        self._proto_ack_cls = proto_ack_class
        self._send_fn = send_fn
        self._stats_cb = stats_callback

        self._inflight: Dict[str, InflightEntry] = {}
        self._inflight_lock = threading.RLock()
        self._frame_to_msg: Dict[int, str] = {}

        self._running = False
        self._send_thread: Optional[threading.Thread] = None
        self._ack_thread: Optional[threading.Thread] = None
        self._timeout_thread: Optional[threading.Thread] = None
        self._house_thread: Optional[threading.Thread] = None

        self._ack_input: "Queue[Tuple[int, str, bool]]" = Queue(maxsize=5000)
        self._stop_event = threading.Event()

        self._stats_lock = threading.Lock()
        self._stats = {
            "enqueue_ok": 0,
            "enqueue_fail": 0,
            "send_attempt": 0,
            "send_success": 0,
            "send_fail": 0,
            "ack_received": 0,
            "ack_mismatch": 0,
            "retry_triggered": 0,
            "retry_exhausted": 0,
            "duplicate_acks": 0,
            "heartbeats_sent": 0,
        }

        self._last_heartbeat = 0

    # =================================================================
    # 生命周期
    # =================================================================

    def start(self) -> bool:
        if self._running:
            return True

        self._running = True
        self._stop_event.clear()

        self._send_thread = threading.Thread(
            target=self._send_loop, daemon=True, name="ReliableSendThread"
        )
        self._ack_thread = threading.Thread(
            target=self._ack_loop, daemon=True, name="ReliableAckThread"
        )
        self._timeout_thread = threading.Thread(
            target=self._timeout_loop, daemon=True, name="ReliableTimeoutThread"
        )
        self._house_thread = threading.Thread(
            target=self._housekeeping_loop, daemon=True, name="ReliableHouseThread"
        )

        for t in [self._send_thread, self._ack_thread, self._timeout_thread, self._house_thread]:
            t.start()

        logger.info(
            f"Reliable transport started for {self._drone_id}: "
            f"ack_timeout={self._cfg.ack_timeout_ns/1e6:.0f}ms "
            f"max_retries={self._cfg.max_retries}"
        )
        return True

    def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        for t in [
            self._send_thread,
            self._ack_thread,
            self._timeout_thread,
            self._house_thread,
        ]:
            if t and t.is_alive():
                t.join(timeout=5.0)

        self._flush_pending_as_failed()
        logger.info("Reliable transport stopped")

    # =================================================================
    # 对外接口：上层 send 调用入口
    # =================================================================

    def submit_frame(self, frame_msg) -> Tuple[bool, str]:
        """
        拦截上层 send_detection 调用，返回 (是否成功入队, msg_id)
        注意：此调用同步持久化到 SQLite，保证不丢。
        """
        try:
            msg_id = f"{self._drone_id}-{frame_msg.frame_id}-{uuid.uuid4().hex[:8]}"
            payload_dict = {
                "drone_id": frame_msg.drone_id,
                "frame_id": frame_msg.frame_id,
                "timestamp": frame_msg.timestamp,
                "detection_count": len(frame_msg.detections),
            }
            payload_json = json.dumps(payload_dict, ensure_ascii=False)
            proto_bytes = frame_msg.SerializeToString()

            ok = self._queue.enqueue(
                msg_id=msg_id,
                frame_id=frame_msg.frame_id,
                payload_json=payload_json,
                proto_bytes=proto_bytes,
            )

            with self._stats_lock:
                if ok:
                    self._stats["enqueue_ok"] += 1
                else:
                    self._stats["enqueue_fail"] += 1

            return ok, msg_id

        except Exception as e:
            logger.error(f"Submit frame failed: {e}")
            traceback.print_exc()
            with self._stats_lock:
                self._stats["enqueue_fail"] += 1
            return False, ""

    # =================================================================
    # 对外接口：云端 ACK 回包入口
    # =================================================================

    def on_ack_received(self, ack) -> None:
        """收到一个 ServerAck 消息，投递给处理线程"""
        try:
            frame_id = int(ack.received_frame_id)
            nonce = getattr(ack, "message", "") or ""
            success = bool(getattr(ack, "success", True))
            self._ack_input.put_nowait((frame_id, nonce, success))
        except Exception as e:
            logger.error(f"ACK dispatch failed: {e}")

    # =================================================================
    # 发送循环：拉队列表 → 发 gRPC → 登记 inflight
    # =================================================================

    def _send_loop(self) -> None:
        logger.info("Reliable send loop started")
        flush_ns = self._cfg.flush_interval_ms * 1_000_000
        last_flush = 0

        while self._running and not self._stop_event.is_set():
            try:
                with self._inflight_lock:
                    inflight_count = len(self._inflight)

                if inflight_count >= self._cfg.max_inflight:
                    self._stop_event.wait(min(0.01, self._cfg.flush_interval_ms / 1000))
                    continue

                now_ns = time.time() * 1e9
                need_flush = (now_ns - last_flush) >= flush_ns

                stuck_timeout = (
                    self._cfg.resend_stuck_interval_ms * 1_000_000
                    if need_flush
                    else 0
                )
                batch_limit = min(
                    self._cfg.batch_size,
                    self._cfg.max_inflight - inflight_count,
                )

                msgs = self._queue.dequeue_batch(
                    batch_size=batch_limit,
                    include_inflight_older_than_ns=stuck_timeout,
                )

                if not msgs and need_flush:
                    self._maybe_send_heartbeat()
                    last_flush = time.time() * 1e9
                    self._stop_event.wait(self._cfg.flush_interval_ms / 1000)
                    continue

                sent_any = False
                now_ns = time.time() * 1e9
                for msg in msgs:
                    if not self._running:
                        break

                    ok = self._try_send_proto(msg.proto_bytes)

                    if ok:
                        new_deadline = int(now_ns) + self._cfg.ack_timeout_ns
                        with self._inflight_lock:
                            if msg.msg_id in self._inflight:
                                old = self._inflight[msg.msg_id]
                                old.ack_nonce = msg.ack_nonce
                                old.sent_at = int(now_ns)
                                old.deadline_at = new_deadline
                                old.proto_bytes = msg.proto_bytes
                                logger.debug(
                                    f"Re-sent stuck msg {msg.msg_id} "
                                    f"(frame={msg.frame_id}), refreshed nonce"
                                )
                            else:
                                entry = InflightEntry(
                                    msg_id=msg.msg_id,
                                    ack_nonce=msg.ack_nonce,
                                    frame_id=msg.frame_id,
                                    sent_at=int(now_ns),
                                    deadline_at=new_deadline,
                                    proto_bytes=msg.proto_bytes,
                                    payload_json=msg.payload_json,
                                )
                                self._inflight[msg.msg_id] = entry
                                self._frame_to_msg[msg.frame_id] = msg.msg_id
                        sent_any = True
                        with self._stats_lock:
                            self._stats["send_success"] += 1
                            self._stats["send_attempt"] += 1
                    else:
                        self._queue.mark_for_retry(
                            msg_id=msg.msg_id,
                            ack_nonce=msg.ack_nonce,
                            backoff_base_ms=self._cfg.backoff_base_ms,
                            max_backoff_ms=self._cfg.max_backoff_ms,
                            max_retries=self._cfg.max_retries,
                        )
                        with self._inflight_lock:
                            self._inflight.pop(msg.msg_id, None)
                            self._frame_to_msg.pop(msg.frame_id, None)
                        with self._stats_lock:
                            self._stats["send_fail"] += 1
                            self._stats["send_attempt"] += 1
                        break

                if sent_any:
                    last_flush = time.time() * 1e9
                else:
                    self._stop_event.wait(self._cfg.flush_interval_ms / 1000)

            except Exception as e:
                logger.error(f"Reliable send loop error: {e}")
                traceback.print_exc()
                self._stop_event.wait(0.5)

        logger.info("Reliable send loop exited")

    def _try_send_proto(self, proto_bytes: bytes) -> bool:
        try:
            return bool(self._send_fn(proto_bytes))
        except Exception as e:
            logger.debug(f"Proto send error: {e}")
            return False

    def _maybe_send_heartbeat(self) -> None:
        now = time.time()
        if now - self._last_heartbeat < self._cfg.heartbeat_interval_s:
            return

        try:
            hb_msg = self._proto_frame_cls(
                drone_id=self._drone_id,
                frame_id=-1,
                timestamp=int(now * 1e9),
            )
            self._send_fn(hb_msg.SerializeToString())
            self._last_heartbeat = now
            with self._stats_lock:
                self._stats["heartbeats_sent"] += 1
        except Exception:
            pass

    # =================================================================
    # ACK 处理循环：命中 inflight → 标记 ACKED
    # =================================================================

    def _ack_loop(self) -> None:
        logger.info("Reliable ACK loop started")

        while self._running and not self._stop_event.is_set():
            try:
                try:
                    frame_id, _nonce, success = self._ack_input.get(timeout=0.2)
                except Empty:
                    continue

                if frame_id == -1:
                    continue

                with self._inflight_lock:
                    msg_id = self._frame_to_msg.get(frame_id)
                    if not msg_id:
                        with self._stats_lock:
                            self._stats["ack_mismatch"] += 1
                        continue

                    entry = self._inflight.pop(msg_id, None)
                    self._frame_to_msg.pop(frame_id, None)

                if entry is None:
                    with self._stats_lock:
                        self._stats["duplicate_acks"] += 1
                    continue

                if success:
                    self._queue.mark_acked(entry.msg_id, entry.ack_nonce)
                    with self._stats_lock:
                        self._stats["ack_received"] += 1
                else:
                    _, exhausted = self._queue.mark_for_retry(
                        msg_id=entry.msg_id,
                        ack_nonce=entry.ack_nonce,
                        backoff_base_ms=self._cfg.backoff_base_ms,
                        max_backoff_ms=self._cfg.max_backoff_ms,
                        max_retries=self._cfg.max_retries,
                    )
                    with self._stats_lock:
                        if exhausted:
                            self._stats["retry_exhausted"] += 1
                        else:
                            self._stats["retry_triggered"] += 1

            except Exception as e:
                logger.error(f"Reliable ACK loop error: {e}")
                traceback.print_exc()

        logger.info("Reliable ACK loop exited")

    # =================================================================
    # 超时扫描循环：未收到 ACK 且超过 deadline → 触发重试
    # =================================================================

    def _timeout_loop(self) -> None:
        logger.info("Reliable timeout loop started")

        while self._running and not self._stop_event.is_set():
            self._stop_event.wait(0.5)
            if not self._running:
                break

            try:
                now_ns = int(time.time() * 1e9)
                expired_entries: List[InflightEntry] = []

                with self._inflight_lock:
                    to_remove: List[Tuple[str, InflightEntry]] = []
                    for mid, entry in self._inflight.items():
                        if now_ns >= entry.deadline_at:
                            to_remove.append((mid, entry))

                    for mid, entry in to_remove:
                        self._inflight.pop(mid, None)
                        self._frame_to_msg.pop(entry.frame_id, None)
                        expired_entries.append(entry)

                for entry in expired_entries:
                    age_ms = (now_ns - entry.sent_at) / 1_000_000
                    logger.debug(
                        f"ACK timeout for msg={entry.msg_id} "
                        f"frame={entry.frame_id} age={age_ms:.0f}ms"
                    )

                    ok, exhausted = self._queue.mark_for_retry(
                        msg_id=entry.msg_id,
                        ack_nonce=entry.ack_nonce,
                        backoff_base_ms=self._cfg.backoff_base_ms,
                        max_backoff_ms=self._cfg.max_backoff_ms,
                        max_retries=self._cfg.max_retries,
                    )
                    if not ok and not exhausted:
                        logger.warning(
                            f"mark_for_retry failed for frame {entry.frame_id} "
                            f"(msg={entry.msg_id} nonce={entry.ack_nonce})"
                        )
                    with self._stats_lock:
                        if exhausted:
                            self._stats["retry_exhausted"] += 1
                        elif ok:
                            self._stats["retry_triggered"] += 1

            except Exception as e:
                logger.error(f"Timeout loop error: {e}")
                traceback.print_exc()

        logger.info("Reliable timeout loop exited")

    # =================================================================
    # 维护线程：过期清理、定期报告
    # =================================================================

    def _housekeeping_loop(self) -> None:
        logger.info("Reliable housekeeping loop started")
        last_expire = time.time()

        while self._running and not self._stop_event.is_set():
            self._stop_event.wait(10.0)
            if not self._running:
                break

            try:
                now = time.time()
                if now - last_expire >= self._cfg.expire_interval_s:
                    self._queue.expire_old_messages()
                    self._queue.purge_all_acked()
                    last_expire = now

                stats = self.get_stats()
                if self._stats_cb:
                    try:
                        self._stats_cb(stats)
                    except Exception:
                        pass

                pending_count = sum(
                    v for k, v in stats.get("queue_counts", {}).items()
                    if k in ("pending", "inflight")
                )
                if pending_count > 0 or stats.get("inflight_count", 0) > 0:
                    logger.info(
                        f"[ReliableTX] "
                        f"pending={stats.get('queue_counts', {}).get('pending', 0)} "
                        f"inflight={stats.get('inflight_count', 0)} "
                        f"acked={stats.get('acked_total', 0)} "
                        f"retries={stats.get('retries_total', 0)} "
                        f"oldest_age={stats.get('oldest_pending_age_s', 0)}s"
                    )

            except Exception as e:
                logger.error(f"Housekeeping error: {e}")

        logger.info("Reliable housekeeping loop exited")

    def _flush_pending_as_failed(self) -> None:
        try:
            with self._inflight_lock:
                for mid, entry in list(self._inflight.items()):
                    try:
                        self._queue.mark_for_retry(
                            msg_id=mid,
                            ack_nonce=entry.ack_nonce,
                            backoff_base_ms=self._cfg.backoff_base_ms,
                            max_backoff_ms=self._cfg.max_backoff_ms,
                            max_retries=0,
                        )
                    except Exception:
                        pass
                self._inflight.clear()
                self._frame_to_msg.clear()
        except Exception:
            pass

    # =================================================================
    # 状态查询
    # =================================================================

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            transport_stats = dict(self._stats)

        q_stats = self._queue.get_stats()
        with self._inflight_lock:
            inflight_count = len(self._inflight)

        oldest_ms = q_stats.get("oldest_pending_age_ms", 0)

        return {
            **transport_stats,
            "queue_counts": q_stats.get("counts", {}),
            "inflight_count": inflight_count,
            "acked_total": q_stats.get("acked", 0),
            "retries_total": q_stats.get("retried", 0),
            "enqueued_total": q_stats.get("enqueued", 0),
            "oldest_pending_age_s": round(oldest_ms / 1000, 1),
            "db_size_mb": q_stats.get("db_size_mb", 0),
        }

    def wait_until_empty(
        self, timeout_s: int = 60, poll_interval_s: float = 0.5
    ) -> bool:
        """阻塞直到队列清空，用于优雅关机等待补传"""
        deadline = time.time() + timeout_s
        while time.time() < deadline and self._running:
            counts = self._queue.get_counts()
            pending = counts.get("pending", 0) + counts.get("inflight", 0)
            with self._inflight_lock:
                inflight = len(self._inflight)
            if pending == 0 and inflight == 0:
                return True
            time.sleep(poll_interval_s)
        return False
