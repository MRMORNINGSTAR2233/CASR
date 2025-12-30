"""
CASR Configuration Module

Provides centralized configuration management using Pydantic Settings.
"""

from .settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
