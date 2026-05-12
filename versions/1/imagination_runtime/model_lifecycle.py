"""
Model lifecycle manager: load-on-demand, progress estimation, eviction timers,
and one-module-at-a-time enforcement to prevent OOM.
"""
from __future__ import annotations

from threading import Lock, Timer
from typing import Callable, Dict, List, Optional, Deque
from collections import deque

from .registry import TaskId


# Estimated load times in seconds for progress bar
KNOWN_LOAD_TIMES: Dict[str, int] = {
    "cad_coder": 75,
    "reasoning_llm": 60,
    "embeddings": 10,
    "reranker": 15,
    "tiny_sd": 45,
}

# TaskId -> list of cache keys required for that task
TASK_TO_KEYS: Dict[TaskId, List[str]] = {
    TaskId.CHAT_MAIN: [],
    TaskId.CAD_CODER: ["cad_coder"],
    TaskId.REASONING: ["reasoning_llm"],
    TaskId.DEEP_RESEARCH: ["embeddings", "reranker", "reasoning_llm"],
    TaskId.IMAGE_TINY: ["tiny_sd"],
}

# Human-readable names for progress display
KEY_TO_LABEL: Dict[str, str] = {
    "cad_coder": "Coder",
    "reasoning_llm": "Reasoning",
    "embeddings": "Embeddings",
    "reranker": "Reranker",
    "tiny_sd": "Image",
}


class ModelLifecycle:
    """
    Manages module load/unload lifecycle:
    - Only one module set loaded at a time (besides main)
    - Switching to a different module: unload current first, then load new
    - Switching to main: start eviction timer; unload after timeout
    - Main model is never unloaded
    """

    EVICTION_TIMEOUT_S = 180  # 3 minutes

    def __init__(
        self,
        unload_callback: Callable[[str], bool],
        eviction_timeout_s: int = EVICTION_TIMEOUT_S,
    ):
        self._unload = unload_callback
        self._eviction_timeout = eviction_timeout_s
        self._lock = Lock()
        self._active_task: Optional[TaskId] = None
        self._eviction_timer: Optional[Timer] = None
        self._pending_eviction_keys: List[str] = []
        self._warn_timers: List[Timer] = []
        self._toast_queue: Deque[str] = deque()
        self._toast_lock = Lock()

    def get_keys_for_task(self, task_id: TaskId) -> List[str]:
        """Return cache keys required for the given task."""
        return TASK_TO_KEYS.get(task_id, [])

    def get_keys_to_unload_before_loading(
        self,
        new_task_id: TaskId,
    ) -> List[str]:
        """
        When switching to new_task_id, return keys that must be unloaded first.
        Call this before loading the new module.
        """
        with self._lock:
            self._cancel_eviction_locked()
            if new_task_id == TaskId.CHAT_MAIN:
                return []
            new_keys = set(self.get_keys_for_task(new_task_id))
            old_keys = set(self.get_keys_for_task(self._active_task)) if self._active_task else set()
            to_unload = list(old_keys - new_keys)
            return to_unload

    def on_switch_to_task(
        self,
        new_task_id: TaskId,
        cache_has: Callable[[str], bool],
    ) -> None:
        """
        Called when user switches to a new task. Handles:
        - Unloading previous module if switching to a different module
        - Scheduling eviction if switching to main
        """
        with self._lock:
            self._cancel_eviction_locked()
            if new_task_id == TaskId.CHAT_MAIN:
                old_keys = self.get_keys_for_task(self._active_task) if self._active_task else []
                if old_keys:
                    self._pending_eviction_keys = [k for k in old_keys if cache_has(k)]
                    self._eviction_timer = Timer(
                        self._eviction_timeout,
                        self._eviction_callback,
                    )
                    self._eviction_timer.daemon = True
                    self._eviction_timer.start()
                    self._schedule_eviction_warning_timers_locked()
                self._active_task = TaskId.CHAT_MAIN
            else:
                self._active_task = new_task_id

    def on_module_loaded(self, task_id: TaskId) -> None:
        """Called after modules for task_id are successfully loaded."""
        with self._lock:
            self._cancel_eviction_locked()
            self._active_task = task_id

    def _cancel_eviction_locked(self) -> None:
        for wt in self._warn_timers:
            try:
                wt.cancel()
            except Exception:
                pass
        self._warn_timers.clear()
        if self._eviction_timer is not None:
            self._eviction_timer.cancel()
            self._eviction_timer = None
        self._pending_eviction_keys = []

    def _schedule_eviction_warning_timers_locked(self) -> None:
        """Fire UI toasts at 1 min and 15 s before eviction (wall-clock delays)."""
        for wt in self._warn_timers:
            try:
                wt.cancel()
            except Exception:
                pass
        self._warn_timers.clear()
        T = float(self._eviction_timeout)
        if T > 60.0:
            delay = T - 60.0
            self._warn_timers.append(self._make_warn_timer(delay, "unload_warn_1min"))
        if T > 15.0:
            delay = T - 15.0
            self._warn_timers.append(self._make_warn_timer(delay, "unload_warn_15s"))

    def _make_warn_timer(self, delay_s: float, toast_key: str) -> Timer:
        def fire() -> None:
            with self._toast_lock:
                self._toast_queue.append(toast_key)

        t = Timer(delay_s, fire)
        t.daemon = True
        t.start()
        return t

    def pop_unload_toast_key(self) -> Optional[str]:
        """Thread-safe: next toast key for the UI poll (e.g. unload_warn_1min)."""
        with self._toast_lock:
            if not self._toast_queue:
                return None
            return self._toast_queue.popleft()

    def _eviction_callback(self) -> None:
        with self._lock:
            keys = list(self._pending_eviction_keys)
            self._pending_eviction_keys = []
            self._eviction_timer = None
        for k in keys:
            self._unload(k)

    def get_load_time_estimate(self, keys: List[str]) -> int:
        """Total estimated load time in seconds for the given keys."""
        return sum(KNOWN_LOAD_TIMES.get(k, 30) for k in keys)

    def get_progress_label(self, key: str) -> str:
        """Human-readable label for progress display."""
        return KEY_TO_LABEL.get(key, key)
