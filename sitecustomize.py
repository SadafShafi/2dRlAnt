"""Compatibility shims loaded automatically by Python at startup.

This workspace uses Python 3.6, but newer imageio-ffmpeg releases import
importlib.resources from the standard library. The backport package provides
the same API for Python 3.6, so we alias it here before any video code runs.
"""

import sys

try:
    import importlib_resources as _importlib_resources
except Exception:
    _importlib_resources = None

if _importlib_resources is not None:
    sys.modules.setdefault("importlib.resources", _importlib_resources)
