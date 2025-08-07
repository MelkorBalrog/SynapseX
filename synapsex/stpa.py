"""Minimal STPA support utilities.

This module provides lightweight classes to model a control flow diagram
and its associated STPA elements.  When creating a new ``STPAElement``
instance, the available control actions are derived directly from the
selected ``ControlFlowDiagram``.  This allows user interfaces to populate
an action combo box with the correct options.

The module also exposes a simple ``STPAAnalysis`` container which can be
used by higher level tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List


@dataclass
class ControlFlowDiagram:
    """Representation of a control flow diagram.

    Parameters
    ----------
    control_actions:
        Iterable of available control actions in the diagram.
    """

    control_actions: Iterable[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Materialise ``control_actions`` into a list.

        ``control_actions`` may be provided as any iterable, including
        generators.  Generators are exhausted after a single iteration which
        would previously result in subsequent calls to
        :meth:`get_control_actions` returning an empty list.  By eagerly
        converting the iterable to a list we ensure that the available control
        actions remain accessible for the lifetime of the diagram.
        """

        self.control_actions = list(self.control_actions)

    def get_control_actions(self) -> List[str]:
        """Return control actions defined by the diagram."""
        return list(self.control_actions)


@dataclass
class STPAElement:
    """STPA element bound to a control flow diagram."""

    diagram: ControlFlowDiagram
    action: str | None = None

    def available_actions(self) -> List[str]:
        """List actions from the associated control flow diagram.

        This method can be used by graphical interfaces to populate the
        Action combo box when a new element is created.
        """

        return self.diagram.get_control_actions()


@dataclass
class STPAAnalysis:
    """Container for an STPA analysis result."""

    diagram: ControlFlowDiagram
    elements: List[STPAElement] = field(default_factory=list)

