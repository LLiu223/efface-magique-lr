"""Inpainting model registry. Importing this package registers the built-in models."""

from companion.models import registry
from companion.models.base import InpaintModel
from companion.models import lama  # noqa: F401  (registers LamaModel)

DEFAULT_MODEL = "lama"

__all__ = ["registry", "InpaintModel", "DEFAULT_MODEL"]
