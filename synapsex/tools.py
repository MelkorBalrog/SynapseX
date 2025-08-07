"""Helper functions for SynapseX tools."""

from __future__ import annotations

from typing import Iterable

from .stpa import ControlFlowDiagram, STPAAnalysis, STPAElement


def create_stpa_analysis(
    control_actions: Iterable[str],
    elements: Iterable[STPAElement] | None = None,
) -> STPAAnalysis:
    """Create a new :class:`STPAAnalysis` object.

    Parameters
    ----------
    control_actions:
        Collection of control actions that define the control flow diagram.
    elements:
        Optional iterable of existing :class:`STPAElement` instances to include
        in the analysis.
    """

    diagram = ControlFlowDiagram(control_actions)
    return STPAAnalysis(diagram, list(elements) if elements else [])

