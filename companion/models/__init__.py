"""Inpainting model registry. Importing this package registers the built-in models."""

from companion.models import registry
from companion.models.base import InpaintModel
from companion.models import lama  # noqa: F401  (registers LamaModel)
from companion.models import migan  # noqa: F401  (registers MiganModel; optional weights)
from companion.models import sd_inpaint  # noqa: F401  (registers SdInpaintModel; optional weights)

DEFAULT_MODEL = "lama"

__all__ = ["registry", "InpaintModel", "DEFAULT_MODEL"]
