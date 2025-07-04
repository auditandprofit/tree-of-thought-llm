"""Task registry and helper utilities."""

from importlib import import_module
from typing import Dict, Type

from .base import Task


_TASK_REGISTRY: Dict[str, str] = {
    "game24": "tot.tasks.game24.Game24Task",
    "text": "tot.tasks.text.TextTask",
    "crosswords": "tot.tasks.crosswords.MiniCrosswordsTask",
}


def register_task(name: str, cls: Type[Task]) -> None:
    """Register a new task class."""

    path = f"{cls.__module__}.{cls.__name__}"
    _TASK_REGISTRY[name] = path


def _load(path: str) -> Type[Task]:
    module_path, cls_name = path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, cls_name)


def get_task(name: str) -> Task:
    """Instantiate a task by ``name``."""

    if name not in _TASK_REGISTRY:
        raise NotImplementedError(f"Task {name!r} is not implemented")
    return _load(_TASK_REGISTRY[name])()


def available_tasks() -> Dict[str, str]:
    """Return a mapping of available task names to class paths."""

    return dict(_TASK_REGISTRY)

