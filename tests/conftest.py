"""Inject stub ``providers`` / ``providers.base`` modules before ``profile`` imports.

Hermes ships ``providers.base.ProviderProfile`` (a dataclass) and
``providers.register_provider`` at runtime; the plugin must import cleanly
without Hermes installed so its units run anywhere. When the real package IS
importable (tests running inside a Hermes checkout), the stubs are skipped.
"""

import sys
import types

try:
    import providers  # noqa: F401
    import providers.base  # noqa: F401
except ImportError:
    _providers = types.ModuleType("providers")
    _providers.REGISTRY = {}  # name/alias -> profile, mirrors register_provider()

    def _register_provider(profile):
        _providers.REGISTRY[profile.name] = profile
        for alias in getattr(profile, "aliases", ()):
            _providers.REGISTRY[alias] = profile

    _providers.register_provider = _register_provider

    _base = types.ModuleType("providers.base")

    class ProviderProfile:
        """Stand-in for the real dataclass: kwargs become attributes."""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        # No-op defaults matching providers/base.py's contract, so a subclass
        # that leans on super() still gets sensible values.
        def get_max_tokens(self, model):
            return getattr(self, "default_max_tokens", None)

        def get_model_context_length(self, model):
            return getattr(self, "model_context_length", None)

    _base.ProviderProfile = ProviderProfile
    _providers.base = _base

    sys.modules["providers"] = _providers
    sys.modules["providers.base"] = _base
