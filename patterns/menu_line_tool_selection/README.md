# menu_line_tool_selection (PATTERN)

The reusable decision pattern of the N2 Menu Line Check entry.

A car owner at a service counter holds a printed menu. For each line of that menu, Strands
Decider 2B answers three typed questions, and plain code turns the answers into one of four
routes. The confidence needed to act depends on what a wrong action costs. A general model is
asked only when Decider is not sure. A line Decider flags as a safety item is never declined
automatically.

| Part | File |
|---|---|
| Typed questions, two wordings (OCR text, image) | `questions.yaml` |
| Routing code `decide_line`, safety gate `build_safety_gate` | `handler.py` |

## The questions

| Name | Type | Question (OCR-text wording) |
|---|---|---|
| `readable` | noul | Can you read both the name of the service and its dollar price on this OCR text of one menu line? |
| `safety` | noul | Is this OCR text of one menu line about brakes, tires, wheel alignment, steering, or another part that affects driving safety? |
| `tool` | choice | A car owner at the dealer's service counter holds the menu. Which action fits this OCR text of one menu line? |

The image wording says "this menu line" instead of "this OCR text of one menu line". The state
for the OCR-text wording is `OCR text of one line of a car dealer's printed service menu: "<text>"`.

The five options of `tool`:

| Option | Criterion given to Decider |
|---|---|
| `decline_addon` | an optional add-on product or chemical treatment the maker's schedule does not require (conditioners, cleanings, additives, nitrogen, protection packages) |
| `check_schedule` | routine maintenance the owner's manual schedules by mileage (oil and filter, filters, rotation, fluid changes, spark plugs, inspection) |
| `check_recall_warranty` | a diagnostic fee or the repair of one specific failed part, which an open recall or the warranty may cover |
| `ask_about_fee` | a fee or surcharge rather than a service (shop supplies, disposal, handling, admin) |
| `owner_decides` | brakes, tires, alignment, steering or another part that affects driving safety |

## The routing code

```python
if p_readable < 0.5:
    route = "retake"
elif tool.choice == "owner_decides":                    # any confidence
    route = "owner_decides"
elif tool.choice == "decline_addon" and p_safety >= 0.5:
    route = "ask_llm"                                   # never declined here
elif tool.confidence >= thresholds[tool.choice]:
    route = "tool"
else:
    route = "ask_llm"
```

| Tool | Act threshold (default) | Why |
|---|---|---|
| `decline_addon` | 0.8 | the one action the owner cannot take back |
| `check_schedule` | 0.5 | a lookup: a wrong pick costs one wasted check |
| `check_recall_warranty` | 0.5 | a lookup |
| `ask_about_fee` | 0.5 | a question to the advisor |
| `owner_decides` | none | followed at any confidence |

The `safety` answer guards that one irreversible action; it does not override the lookups. On OCR
text it says yes on tire lines that are not safety work (see "Measured").

`decide_line(answers, thresholds=None)` returns a `LineDecision` with the route, the tool to run,
the confidence, the threshold used, P(readable), P(safety) and the reason in words. `thresholds`
overrides some of the defaults (`{"decline_addon": 0.9}`); `parse_thresholds("decline_addon=0.9")`
reads the same from text, and `act_threshold=` sets one threshold for every tool.

Which input feeds which question (from the spike, see "Measured"): `safety` and `tool` are asked
on the OCR text. `readable` is asked on the line image (name crop + price crop) when there is
one; without an image the caller decides readability in code ("OCR found a name and a price").

## The safety gate

Lines flagged by `safety` that Decider wanted to decline, and lines below their tool's
threshold, go to a general model (route `ask_llm`). There `build_safety_gate(client)` returns a
`DeciderGate` for Strands `before_tool_call`. If the model calls `decline_addon`, the gate asks
Decider the `safety` question about the `line_text` argument of that call. P(safety) >= 0.5 gives
`Deny`, and the model reads `DENIED: this line is a safety item; the owner decides`. Other tools
pass without a Decider call. If Decider is down, the gate denies as well.

## Using it in another agent

```python
from decider_kit import DeciderClient
from menu_line_tool_selection import build_safety_gate, decide_line, load_pattern_questions, ocr_state

decider = DeciderClient("http://decider-host:8099")
answers = decider.ask(ocr_state(line_text), load_pattern_questions("ocr_text")).answers
decision = decide_line(answers)            # retake | owner_decides | tool | ask_llm

# where a general model picks the tool instead
agent = Agent(tools=[...], interventions=[build_safety_gate(decider)])
```

The tools the gate guards must take `line_text: str` as an argument. `src/menu_line_check/` is
the full agent built on this pattern.

## Measured

Spike of 2026-10-20 on synthetic menus (see `spike/n2-menu/` in the campaign folder): 120
labelled menu lines, 103 of them readable, warm Decider 2B on a Mac mini.

* Tool choice accuracy: 0.90 from the line image, 0.88 from the OCR text.
* At confidence >= 0.9 the tool choice was right on every line (1.00), but only on about 20 to 26 of
  the 103 readable lines.
* At >= 0.8 the OCR-text path covers 59 of 103 lines at 0.86; the image path 54 of 103 at 0.93.
* At >= 0.7 the OCR-text path covers 81 of 103 lines at 0.90.
* The yes/no `safety` question: 0.90 on the OCR text, 0.67 to 0.76 on images. With the rule
  "safety yes OR tool owner_decides" no safety line was missed.
* Price over $100 was read correctly on 97 to 100% of lines, even from the whole line as one image.

Live run of 2026-10-20 on the demo menu (real Decider on the Mac mini): the OCR-text `safety`
question said yes on the brake line (0.72), but also on "Tire rotation" (0.68) and "Nitrogen tire
fill" (0.66). Under the earlier rule "safety yes OR tool owner_decides -> owner decides" those
became the owner's call, and with one act threshold of 0.8 the line "Replace fuel pump module"
(check_recall_warranty at 0.64) never reached the recall lookup.

Pipeline policies replayed on the recorded answers from the spike (synthetic menus; 103 readable
lines, Decider 2B on the OCR text, 2026-10-20; `menu-race --replay-recorded`). "Acted" means a
tool ran or the owner decides from Decider's answers alone; the rest go to the general model.

| Policy | Acted | Accuracy on acted | Safety lines auto-declined | False "your call" | Wrong declines |
|---|---|---|---|---|---|
| per-tool thresholds (this pattern) | 90/103 | 0.89 | 0/20 | 0 | 0 |
| previous: safety OR owner_decides, act >= 0.8 | 76/103 | 0.76 | 0/20 | 10 | 0 |

So the defaults: OCR text for `safety` and `tool`; act at 0.8 for `decline_addon` and 0.5 for the
lookups and the fee question; `safety` only blocks a decline. These numbers come from synthetic
menus; measure them again on real menus before relying on them.
