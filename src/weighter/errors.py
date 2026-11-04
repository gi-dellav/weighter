"""Exceptions raised by :mod:`weighter`."""

from __future__ import annotations


class WeighterError(Exception):
    """Base class for all harness errors."""


class ConfigurationError(WeighterError):
    """The harness was configured incorrectly."""


class ToolError(WeighterError):
    """A tool call could not be completed."""


class EditError(ToolError):
    """A weight edit was invalid or could not be applied."""


class LLMError(WeighterError):
    """The language-model request failed."""
