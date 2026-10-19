"""decider_kit: a concept-agnostic toolkit around Strands Decider 2B.

Typed questions, an HTTP client, a threshold policy, a decision ledger, a generic
Strands ``InterventionHandler`` (``DeciderGate``) and a replay benchmark. No domain
logic lives here: the questions and the mapping from answers to an action are supplied
by the caller.
"""

from .client import (
    Answer,
    ChoiceAnswer,
    DeciderClient,
    DeciderError,
    DeciderRequestError,
    DeciderResponse,
    DeciderUnavailable,
    NoulAnswer,
    ScoreAnswer,
)
from .intervention import DeciderGate, GateInput, recent_conversation, tool_call_state_builder
from .ledger import Ledger, read_ledger
from .policy import Route, ThresholdPolicy, Thresholds
from .questions import (
    ChoiceQuestion,
    NoulQuestion,
    Question,
    QuestionError,
    ScoreQuestion,
    load_questions,
    parse_questions,
)

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "DeciderClient",
    "DeciderError",
    "DeciderGate",
    "DeciderRequestError",
    "DeciderResponse",
    "DeciderUnavailable",
    "GateInput",
    "Ledger",
    "NoulAnswer",
    "NoulQuestion",
    "Question",
    "QuestionError",
    "Route",
    "ScoreAnswer",
    "ScoreQuestion",
    "ThresholdPolicy",
    "Thresholds",
    "load_questions",
    "parse_questions",
    "read_ledger",
    "recent_conversation",
    "tool_call_state_builder",
]
