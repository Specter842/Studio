"""Clip generators, all behind one interface.

Importing this package registers every adapter by name. Neither adapter pulls
in its HTTP dependency at import time, so this is cheap even on a run that
generates nothing.
"""

from generators.base_adapter import (  # noqa: F401
    GenerationError,
    GenerationRequest,
    GeneratorAdapter,
    UnknownAdapter,
    adapter_class,
    available,
    get_adapter,
    register,
)

# Imported for the side effect of registering themselves. local_comfyui first:
# it is the default, and the free path should be the one that is always present.
from generators import local_comfyui  # noqa: F401,E402
from generators import fal_gateway  # noqa: F401,E402
from generators import heygen  # noqa: F401,E402

__all__ = [
    "GenerationError",
    "GenerationRequest",
    "GeneratorAdapter",
    "UnknownAdapter",
    "adapter_class",
    "available",
    "get_adapter",
    "register",
]
