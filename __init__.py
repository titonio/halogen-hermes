"""Halogen model-provider plugin: registers the ``halogen`` provider route.

Hermes discovers this directory (``plugin.yaml`` + ``__init__.py``) under
``$HERMES_HOME/plugins/model-providers/`` and imports it; the import itself
performs the registration, matching Hermes' discovery contract.
"""

from providers import register_provider

try:  # Hermes loads the plugin directory as a package
    from .profile import halogen
except ImportError:  # standalone import (tests, scripts): repo root on sys.path
    from profile import halogen

register_provider(halogen)

__all__ = ["halogen"]
