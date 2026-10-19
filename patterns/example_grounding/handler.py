"""EXAMPLE pattern (not our hackathon entry): is this tool call grounded and ready?

Same two questions and the same policy as the official weather example
(strands-labs/strands-decider, examples/strands/tool_call_intervention.py, Apache-2.0),
wired through ``DeciderGate``. ``YES = 0.45`` is the example's illustrative knob, not a
recommendation: fit it to your own traffic with ``decider-bench``.

    from decider_kit import DeciderClient
    from handler import build_gate
    agent = Agent(tools=[...], interventions=[build_gate(DeciderClient("http://127.0.0.1:8099"))])
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from strands.interventions import Guide, InterventionAction, Proceed

from decider_kit import (
    Answer,
    DeciderClient,
    DeciderGate,
    Ledger,
    NoulAnswer,
    load_questions,
    tool_call_state_builder,
)

QUESTIONS_PATH = Path(__file__).with_name("questions.yaml")
YES = 0.45


def decide(answers: Mapping[str, Answer]) -> InterventionAction:
    grounded, premature = answers["args_grounded"], answers["premature"]
    assert isinstance(grounded, NoulAnswer) and isinstance(premature, NoulAnswer)
    if grounded.probability < YES:
        return Guide(
            feedback=(
                "The arguments are not grounded in anything the user said -- they look guessed. "
                "Ask the user to confirm the values instead of inventing them."
            )
        )
    if premature.probability >= YES:
        return Guide(feedback="It is too early to call this tool. Clarify with the user first.")
    return Proceed()


def build_gate(client: DeciderClient, ledger: Ledger | None = None) -> DeciderGate:
    return DeciderGate(
        name="example-grounding",
        client=client,
        questions=load_questions(QUESTIONS_PATH),
        state_builder=tool_call_state_builder(),
        decide=decide,
        ledger=ledger,
    )
