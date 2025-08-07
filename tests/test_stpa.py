import sys
from pathlib import Path

# Ensure project root is on sys.path for direct module imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from synapsex.stpa import ControlFlowDiagram, STPAElement
from synapsex.tools import create_stpa_analysis


def test_available_actions_lists_from_diagram():
    diagram = ControlFlowDiagram(["A", "B"])
    element = STPAElement(diagram)
    assert element.available_actions() == ["A", "B"]


def test_create_stpa_analysis_builds_analysis():
    elements = []
    analysis = create_stpa_analysis(["A"], elements)
    assert analysis.diagram.get_control_actions() == ["A"]
    assert analysis.elements == elements
