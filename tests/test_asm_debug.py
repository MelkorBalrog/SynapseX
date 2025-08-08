import io
from contextlib import redirect_stdout
import pytest
import os
import sys

sys.path.append(os.getcwd())

from synapse.soc import SoC


def test_line_enumeration_and_debug_output():
    lines = [
        "ADDI $t0, $zero, 1",
        "HALT",
    ]
    soc = SoC()
    soc.load_assembly(lines)
    assert soc.asm_program[0][0] == 1
    assert soc.asm_program[1][0] == 2
    buf = io.StringIO()
    with redirect_stdout(buf):
        soc.run(debug=True)
    output = buf.getvalue()
    assert "1: ADDI $t0, $zero, 1" in output
    assert "2: HALT" in output


def test_invalid_instruction_raises_error():
    lines = ["FOO $t0, $t1, $t2"]
    soc = SoC()
    soc.load_assembly(lines)
    with pytest.raises(ValueError):
        soc.run()
