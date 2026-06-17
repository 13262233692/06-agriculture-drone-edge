import os
import sys
import json
import time
import sqlite3
import threading
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from enum import IntEnum


logger = logging.getLogger(__name__)


class MessageStatus(IntEnum):
    PENDING = 0
    INFLIGHT = 1
    ACKED = 2
    FAILED = 3
    EXPIRED = 4


@dataclass
class QueuedMessage:
    msg_id: str
    drone_id: str
    frame_id: int
    created_at: int
    updated_at: int
    status: MessageStatus
    retry_count: int
    next_retry_at: int
    payload_json: str
    proto_bytes: bytes = b""
    ack_nonce: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = int(self.status)
        d["proto_bytes"] = self.proto_bytes.hex()
        return d


class SQLitePersistenceQueue:
    """
    SQLite 持久化消息队列，保证断网场景下消息不丢失。
    支持：WAL 模式、多线程安全、状态机流转、批量读取。
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS message_queue (
        msg_id          TEXT PRIMARY KEY,
        drone_id        TEXT NOT NULL,
        frame_id        INTEGER NOT NULL,
        created_at      INTEGER NOT NULL,
        updated_at      INTEGER NOT NULL,
        status          INTEGER NOT NULL DEFAULT 0,
        retry_count     INTEGER NOT NULL DEFAULT 0,
        next_retry_at   INTEGER NOT NULL DEFAULT 0,
        payload_json    TEXT NOT NULL,
        proto_bytes     BLOB NOT NULL DEFAULT X'',
        ack_nonce       TEXT NOT NULL DEFAULT ''
    );

    CREATE INDEX IF NOT EXISTS idx_status_retry ON message_queue(status, next_retry_at);
    CREATE INDEX IF NOT EXISTS idx_drone_frame ON message_queue(drone_id, frame_id);
    CREATE INDEX IF NOT EXISTS idx_created ON message_queue(created_at);
    """

    def __init__(
        self,
        db_path: str,
        drone_id: str,
        max_size_mb: int = 512,
        max_age_days: int = 7,
        wal_autocheckpoint: int = 1000,
    ):
        self._db_path = db_path
        self._drone_id = drone_id
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._max_age_ns = max_age_days * 24 * 3600 * 10**9

        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self._lock = threading.RLock()
        self._conn = self._init_db(wal_autocheckpoint)

        self._stats_lock = threading.Lock()
        self._stats = {
            "enqueued": 0,
            "dequeued": 0,
            "acked": 0,
            "retried": 0,
            "expired": 0,
            "failed": 0,
            "evicted_size": 0,
            "evicted_age": 0,
        }

    def _init_db(self, wal_autocheckpoint: int) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._db_path,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.execute(f"PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA synchronous=NORMAL")
        conn.execute(f"PRAGMA busy_timeout=30000")
        conn.execute(f"PRAGMA wal_autocheckpoint={wal_autocheckpoint}")
        conn.execute(f"PRAGMA temp_store=MEMORY")
        conn.execute(f"PRAGMA mmap_size=268435456")

        conn.executescript(self.SCHEMA)
        logger.info(
            f"SQLite queue initialized: path={self._db_path} "
            f"drone_id={self._drone_id} mode=WAL"
        )
        return conn

    # =================================================================
    # 公共写入接口
    # =================================================================

    def enqueue(
        self,
        msg_id: str,
        frame_id: int,
        payload_json: str,
        proto_bytes: bytes,
    ) -> bool:
        now = int(time.time() * 1e9)

        try:
            with self._lock, self._transaction() as cur:
                cur.execute(
                    """
                    INSERT OR REPLACE INTO message_queue
                    (msg_id, drone_id, frame_id, created_at, updated_at,
                     status, retry_count, next_retry_at, payload_json, proto_bytes, ack_nonce)
                    VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?)
                    """,
                    (
                        msg_id,
                        self._drone_id,
                        frame_id,
                        now,
                        now,
                        int(MessageStatus.PENDING),
                        payload_json,
                        proto_bytes,
                        "",
                    ),
                )

            with self._stats_lock:
                self._stats["enqueued"] += 1

            self._evict_if_needed()
            return True

        except Exception as e:
            logger.error(f"Failed to enqueue msg {msg_id}: {e}")
            traceback.print_exc()
            return False

    # =================================================================
    # 公共读取接口
    # =================================================================

    def dequeue_batch(
        self,
        batch_size: int = 50,
        include_inflight_older_than_ns: int = 0,
    ) -> List[QueuedMessage]:
        """
        获取可发送的消息：
          1) PENDING 且 next_retry_at <= now
          2) INFLIGHT 且超时 (include_inflight_older_than_ns > 0)
        同时将状态流转为 INFLIGHT。
        """
        now = int(time.time() * 1e9)
        results: List[QueuedMessage] = []

        try:
            with self._lock:
                cur = self._conn.cursor()

                sql_parts = []
                params = [self._drone_id]

                sql_parts.append(
                    "(status = ? AND next_retry_at <= ?)"
                )
                params.append(int(MessageStatus.PENDING))
                params.append(now)

                if include_inflight_older_than_ns > 0:
                    cutoff = now - include_inflight_older_than_ns
                    sql_parts.append(
                        "(status = ? AND updated_at <= ?)"
                    )
                    params.append(int(MessageStatus.INFLIGHT))
                    params.append(cutoff)

                where = " OR ".join(sql_parts)

                query = f"""
                    SELECT msg_id, drone_id, frame_id, created_at, updated_at,
                           status, retry_count, next_retry_at, payload_json,
                           proto_bytes, ack_nonce
                    FROM message_queue
                    WHERE drone_id = ? AND ({where})
                    ORDER BY next_retry_at ASC, created_at ASC
                    LIMIT ?
                """
                params.append(batch_size)

                cur.execute(query, params)
                rows = cur.fetchall()

                if not rows:
                    return []

                nonce_map = {}
                for row in rows:
                    msg_id = row[0]
                    nonce = f"{msg_id}-{now}-{os.urandom(4).hex()}"
                    nonce_map[msg_id] = nonce

                case_sql = "CASE msg_id " + " ".join(
                    [f"WHEN ? THEN ?" for _ in nonce_map]
                ) + " END"
                case_params: List[Any] = []
                for mid, n in nonce_map.items():
                    case_params.extend([mid, n])

                update_sql = f"""
                    UPDATE message_queue
                    SET status = ?, updated_at = ?, ack_nonce = ({case_sql})
                    WHERE msg_id IN ({','.join('?' * len(nonce_map))})
                """
                update_params = [
                    int(MessageStatus.INFLIGHT),
                    now,
                    *case_params,
                    *nonce_map.keys(),
                ]

                cur.execute(update_sql, update_params)

                for row in rows:
                    msg = QueuedMessage(
                        msg_id=row[0],
                        drone_id=row[1],
                        frame_id=row[2],
                        created_at=row[3],
                        updated_at=row[4],
                        status=MessageStatus(row[5]),
                        retry_count=row[6],
                        next_retry_at=row[7],
                        payload_json=row[8],
                        proto_bytes=row[9] or b"",
                        ack_nonce=nonce_map.get(row[0], row[10] or ""),
                    )
                    results.append(msg)

            with self._stats_lock:
                self._stats["dequeued"] += len(results)

            return results

        except Exception as e:
            logger.error(f"Dequeue batch failed: {e}")
            traceback.print_exc()
            return []

    # =================================================================
    # 状态流转接口
    # =================================================================

    def mark_acked(self, msg_id: str, ack_nonce: str) -> bool:
        now = int(time.time() * 1e9)
        try:
            with self._lock, self._transaction() as cur:
                cur.execute(
                    """
                    UPDATE message_queue
                    SET status = ?, updated_at = ?
                    WHERE msg_id = ? AND ack_nonce = ? AND status = ?
                    """,
                    (
                        int(MessageStatus.ACKED),
                        now,
                        msg_id,
                        ack_nonce,
                        int(MessageStatus.INFLIGHT),
                    ),
                )
                ok = cur.rowcount > 0

            if ok:
                with self._stats_lock:
                    self._stats["acked"] += 1
            return ok

        except Exception as e:
            logger.error(f"Failed to mark acked {msg_id}: {e}")
            return False

    def mark_for_retry(
        self,
        msg_id: str,
        ack_nonce: str,
        backoff_base_ms: int = 500,
        max_backoff_ms: int = 60000,
        max_retries: int = 50,
    ) -> Tuple[bool, bool]:
        """
        返回 (是否成功, 是否已达最大重试次数)
        """
        now = int(time.time() * 1e9)

        try:
            with self._lock:
                cur = self._conn.cursor()

                cur.execute(
                    "SELECT retry_count FROM message_queue WHERE msg_id = ? AND ack_nonce = ? AND status = ?",
                    (msg_id, ack_nonce, int(MessageStatus.INFLIGHT)),
                )
                row = cur.fetchone()
                if not row:
                    return False, False

                old_retry = row[0]
                new_retry = old_retry + 1

                if new_retry > max_retries:
                    cur.execute(
                        """
                        UPDATE message_queue
                        SET status = ?, updated_at = ?
                        WHERE msg_id = ? AND ack_nonce = ?
                        """,
                        (
                            int(MessageStatus.FAILED),
                            now,
                            msg_id,
                            ack_nonce,
                        ),
                    )
                    with self._stats_lock:
                        self._stats["failed"] += 1
                    return True, True

                delay_ms = min(
                    backoff_base_ms * (2 ** min(old_retry, 10)),
                    max_backoff_ms,
                )
                delay_ns = delay_ms * 1_000_000
                jitter_ns = int(
                    delay_ns * 0.1 * (2 * (time.time() % 1) - 1)
                )
                next_retry = now + delay_ns + jitter_ns

                cur.execute(
                    """
                    UPDATE message_queue
                    SET status = ?, updated_at = ?, retry_count = ?, next_retry_at = ?, ack_nonce = ''
                    WHERE msg_id = ? AND ack_nonce = ? AND status = ?
                    """,
                    (
                        int(MessageStatus.PENDING),
                        now,
                        new_retry,
                        next_retry,
                        msg_id,
                        ack_nonce,
                        int(MessageStatus.INFLIGHT),
                    ),
                )
                ok = cur.rowcount > 0

            if ok:
                with self._stats_lock:
                    self._stats["retried"] += 1
            return ok, False

        except Exception as e:
            logger.error(f"Failed to mark retry {msg_id}: {e}")
            return False, False

    def expire_old_messages(self) -> int:
        cutoff = int(time.time() * 1e9) - self._max_age_ns
        try:
            with self._lock, self._transaction() as cur:
                cur.execute(
                    """
                    UPDATE message_queue
                    SET status = ?, updated_at = ?
                    WHERE drone_id = ? AND status IN (?, ?) AND created_at <= ?
                    """,
                    (
                        int(MessageStatus.EXPIRED),
                        int(time.time() * 1e9),
                        self._drone_id,
                        int(MessageStatus.ACKED),
                        int(MessageStatus.FAILED),
                        cutoff,
                    ),
                )
                count = cur.rowcount

                cur.execute(
                    """
                    DELETE FROM message_queue
                    WHERE drone_id = ? AND status = ?
                    """,
                    (self._drone_id, int(MessageStatus.EXPIRED)),
                )
                count += cur.rowcount

            if count > 0:
                with self._stats_lock:
                    self._stats["expired"] += count
                logger.info(f"Expired {count} old messages")
            return count

        except Exception as e:
            logger.error(f"Expire error: {e}")
            return 0

    def _evict_if_needed(self) -> None:
        try:
            size = os.path.getsize(self._db_path)
            wal_path = self._db_path + "-wal"
            if os.path.exists(wal_path):
                size += os.path.getsize(wal_path)

            if size < self._max_size_bytes:
                return

            logger.warning(
                f"Queue size {size/1024/1024:.1f}MB exceeds limit, evicting old ACKED messages"
            )
            with self._lock, self._transaction() as cur:
                cur.execute(
                    """
                    DELETE FROM message_queue
                    WHERE rowid IN (
                        SELECT rowid FROM message_queue
                        WHERE drone_id = ? AND status = ?
                        ORDER BY created_at ASC
                        LIMIT 500
                    )
                    """,
                    (self._drone_id, int(MessageStatus.ACKED)),
                )
                evicted = cur.rowcount

            if evicted > 0:
                with self._stats_lock:
                    self._stats["evicted_size"] += evicted
                logger.info(f"Evicted {evicted} old acked messages")

                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        except Exception as e:
            logger.error(f"Evict error: {e}")

    def purge_all_acked(self) -> int:
        try:
            with self._lock, self._transaction() as cur:
                cur.execute(
                    "DELETE FROM message_queue WHERE drone_id = ? AND status = ?",
                    (self._drone_id, int(MessageStatus.ACKED)),
                )
                count = cur.rowcount
            return count
        except Exception as e:
            logger.error(f"Purge error: {e}")
            return 0

    # =================================================================
    # 状态查询接口
    # =================================================================

    def get_counts(self) -> Dict[str, int]:
        try:
            with self._lock:
                cur = self._conn.cursor()
                cur.execute(
                    "SELECT status, COUNT(*) FROM message_queue WHERE drone_id = ? GROUP BY status",
                    (self._drone_id,),
                )
                rows = cur.fetchall()

            result = {
                "pending": 0,
                "inflight": 0,
                "acked": 0,
                "failed": 0,
                "expired": 0,
            }
            for status_val, count in rows:
                try:
                    s = MessageStatus(status_val).name.lower()
                    result[s] = count
                except ValueError:
                    result[f"unknown_{status_val}"] = count

            return result
        except Exception as e:
            logger.error(f"Count error: {e}")
            return {}

    def get_oldest_pending_age_ms(self) -> int:
        try:
            with self._lock:
                cur = self._conn.cursor()
                cur.execute(
                    """
                    SELECT MIN(created_at) FROM message_queue
                    WHERE drone_id = ? AND status IN (?, ?)
                    """,
                    (
                        self._drone_id,
                        int(MessageStatus.PENDING),
                        int(MessageStatus.INFLIGHT),
                    ),
                )
                row = cur.fetchone()

            if not row or row[0] is None:
                return 0
            return max(0, int((time.time() * 1e9 - row[0]) / 1_000_000))

        except Exception as e:
            logger.error(f"Age query error: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            stats = dict(self._stats)
        stats["counts"] = self.get_counts()
        stats["oldest_pending_age_ms"] = self.get_oldest_pending_age_ms()
        try:
            stats["db_size_mb"] = round(
                os.path.getsize(self._db_path) / (1024 * 1024), 2
            )
        except Exception:
            stats["db_size_mb"] = 0
        return stats

    def pending_ids_snapshot(self, limit: int = 1000) -> List[Tuple[str, int, int]]:
        """返回 [(msg_id, frame_id, created_at)]，用于调试"""
        try:
            with self._lock:
                cur = self._conn.cursor()
                cur.execute(
                    """
                    SELECT msg_id, frame_id, created_at FROM message_queue
                    WHERE drone_id = ? AND status IN (?, ?)
                    ORDER BY created_at ASC LIMIT ?
                    """,
                    (
                        self._drone_id,
                        int(MessageStatus.PENDING),
                        int(MessageStatus.INFLIGHT),
                        limit,
                    ),
                )
                return cur.fetchall()
        except Exception as e:
            logger.error(f"Snapshot error: {e}")
            return []

    # =================================================================
    # 辅助
    # =================================================================

    @contextmanager
    def _transaction(self):
        cur = self._conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            yield cur
            cur.execute("COMMIT")
        except Exception:
            try:
                cur.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            cur.close()

    def close(self) -> None:
        try:
            self.expire_old_messages()
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._conn.close()
            logger.info("SQLite queue closed")
        except Exception as e:
            logger.error(f"Close error: {e}")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
