from __future__ import annotations

import gc
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, Optional

import torch


@dataclass
class RuntimeCache:
    """
    Process-wide caches so the main LLM stays loaded and module models
    are lazy-loaded and reused.
    """

    lock: RLock = field(default_factory=RLock)

    main_tokenizer: Optional[Any] = None
    main_model: Optional[Any] = None
    # v1.3: multimodal main (processor + flag); tokenizer may equal processor.tokenizer
    main_processor: Optional[Any] = None
    main_is_vlm: bool = False

    module_objects: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str) -> Any:
        with self.lock:
            return self.module_objects.get(key)

    def set(self, key: str, value: Any) -> None:
        with self.lock:
            self.module_objects[key] = value

    def unload_module(self, key: str) -> bool:
        """
        Unload a single module by key, freeing GPU memory.
        Returns True if the module was found and unloaded, False otherwise.
        """
        with self.lock:
            obj = self.module_objects.pop(key, None)
            if obj is None:
                return False

        # Delete references to model/tokenizer inside the module dict
        if isinstance(obj, dict):
            for k in list(obj.keys()):
                obj[k] = None
        del obj

        self._free_memory()
        return True

    def clear_modules(self) -> None:
        with self.lock:
            for key in list(self.module_objects.keys()):
                obj = self.module_objects.pop(key, None)
                if obj is not None and isinstance(obj, dict):
                    for k in list(obj.keys()):
                        obj[k] = None
                del obj
        self._free_memory()

    def _free_memory(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
