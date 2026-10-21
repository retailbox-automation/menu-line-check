"""The README's "use the safety gate in another agent" snippet imports what it says, and works."""

from __future__ import annotations

import re
from pathlib import Path

from decider_kit import DeciderGate

README = Path(__file__).parents[1] / "README.md"


def snippet() -> str:
    text = README.read_text(encoding="utf-8")
    head = text.index("Using the pattern's safety gate in another agent")
    return re.search(r"```python\n(.*?)```", text[head:], re.S).group(1)


def test_the_snippet_imports_resolve_and_build_a_gate():
    code = snippet()
    imports = [ln for ln in code.splitlines() if ln.startswith(("from ", "import "))]
    assert imports == [
        "from strands import Agent",
        "from decider_kit import DeciderClient",
        "from menu_line_tool_selection import build_safety_gate",
    ]
    names: dict = {}
    exec("\n".join(imports), names)  # noqa: S102 - our own README, checked above line by line
    gate = names["build_safety_gate"](names["DeciderClient"]("http://decider-host:8099"))
    assert isinstance(gate, DeciderGate) and gate.lifecycle == "before_tool_call"
    assert "interventions=[build_safety_gate(DeciderClient(" in code


def test_the_guarded_tool_takes_line_text():
    from menu_line_check.tools import make_picker_tools

    (decline,) = [t for t in make_picker_tools(lambda *a: None) if t.tool_name == "decline_addon"]
    assert "line_text" in decline.tool_spec["inputSchema"]["json"]["properties"]
    assert "line_text: str" in README.read_text(encoding="utf-8")
