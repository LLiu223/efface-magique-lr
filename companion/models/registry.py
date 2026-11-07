"""Name -> InpaintModel class registry. Lookups are exact-match only (names may come from HTTP)."""

from typing import Dict, List, Type

from companion.models.base import InpaintModel

_REGISTRY: Dict[str, Type[InpaintModel]] = {}


class UnknownModelError(KeyError):
    def __str__(self) -> str:  # KeyError's default str() wraps the message in quotes
        return str(self.args[0])


def register(cls: Type[InpaintModel]) -> Type[InpaintModel]:
    """Register a model class under cls.name (usable as a decorator)."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} has no name")
    _REGISTRY[cls.name] = cls
    return cls


def get(name: str) -> Type[InpaintModel]:
    try:
        return _REGISTRY[name]
    except (KeyError, TypeError):
        raise UnknownModelError(
            f"Unknown model {name!r}; known models: {', '.join(sorted(_REGISTRY)) or 'none'}"
        ) from None


def all() -> List[Type[InpaintModel]]:  # noqa: A001 - name fixed by the spec
    return list(_REGISTRY.values())


def available() -> List[Type[InpaintModel]]:
    return [cls for cls in _REGISTRY.values() if cls.is_available()]
