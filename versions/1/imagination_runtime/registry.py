from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

from .paths import ModelPaths


class TaskId(str, Enum):
    CHAT_MAIN = "chat_main"
    CAD_CODER = "cad_coder"
    REASONING = "reasoning"
    DEEP_RESEARCH = "deep_research"
    IMAGE_TINY = "image_tiny"


@dataclass(frozen=True)
class TaskSpec:
    id: TaskId
    label: str
    description: str
    required_keys: List[str]


def get_task_specs() -> List[TaskSpec]:
    return [
        TaskSpec(
            id=TaskId.CHAT_MAIN,
            label="Imagination 1.3.0",
            description="Always-on main chat / VLM from the repo root (IMAGINATION_MAIN_VLM_PATH optional).",
            required_keys=[],
        ),
        TaskSpec(
            id=TaskId.CAD_CODER,
            label="Coder",
            description="7B Qwen coder finetune (Lua, Python, C++, C#). Weights under modules/cad/ or IMAGINATION_CAD_CODER_PATH.",
            required_keys=["cad_coder"],
        ),
        TaskSpec(
            id=TaskId.REASONING,
            label="Reasoning",
            description="Loads the reasoning model when selected.",
            required_keys=["reasoning_llm"],
        ),
        TaskSpec(
            id=TaskId.DEEP_RESEARCH,
            label="Research",
            description="Loads retrieval models when selected.",
            required_keys=["embeddings", "reranker", "reasoning_llm"],
        ),
        TaskSpec(
            id=TaskId.IMAGE_TINY,
            label="Image",
            description="Loads tiny image pipeline when selected.",
            required_keys=["tiny_sd"],
        ),
    ]
