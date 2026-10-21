# A real service sheet, run live (2026-10-21)

The owner's own printed sheet from the 20,000-mile service of a 2024 SUV (17.10.2026), photographed and
redacted (VIN, dealer, service advisor, logos, product brands, lender and form code removed). Two sheets
from the same visit:

- the dealer's recommended sheet, TOTAL $801.05 (`tests/fixtures/sheet_801_vision.json`);
- the sheet printed after the owner declined four lines, TOTAL $391.38, with those four under a
  "Declined" header (`tests/fixtures/sheet_391_vision.json`).

Neither sheet prices a single line: nine bullet lines, then Menu Total, Shop Charges, Tax, Fees, TOTAL.
The OCR is Apple Vision on the Mac (free), written in the shape of an Amazon Textract
`DetectDocumentText` response by `scripts/vision_to_textract.swift`; no Textract call was made.

## The maker's schedule used

The maker's 2024 Warranty and Services Guide for this model,
https://assets.sia.toyota.com/publications/en/omms-s/L-MMS-24RX350/pdf/L-MMS-24RX350.pdf , printed page 50
(PDF page 52), box "24 Months or 20,000 Miles" (checked on the page itself, 2026-10-21). Items:

- replace engine oil with 0W-20; replace oil filter (footnote 1: every 12 months or 10,000 miles, or when
  the oil maintenance reminder appears, whichever comes first)
- replace cabin air filter
- reset oil maintenance reminder if oil was changed; reset scheduled maintenance reminder
- rotate tires
- inspect brake linings, drums, pads and discs; inspect wiper blades; check installation of driver's
  floor mat; road test vehicle

Not entered: the injector cleaner (one region only) and the items for special operating conditions
(off-road, towing, road salt). Data: `src/menu_line_check/data/maker_schedules.yaml`, with the URL and page.

How a `check_schedule` line is judged against it: listed at 20,000 mi -> keep ("in the maker's schedule
at 20,000 mi"); a listed item sold with an extra the schedule does not list -> ask for the item alone;
a line that names a listed item but also words the schedule does not have -> ask; not listed -> decline
("not in the maker's schedule at 20,000 mi"), unless Decider flags the line as a safety item
(P(safety) >= 0.5), in which case the owner decides. A dealer's "multi-point inspection" maps to the
schedule's inspect and check items.

## Free run (no Bedrock)

Two runs, 2026-10-21 11:35:31Z to 11:35:41Z, at commit 2c84e43: Decider 2B on our reference server (MPS,
one request at a time) through `127.0.0.1:18299`, NHTSA live on the first run and from the disk cache on
the second, no Bedrock (code-made advisor questions, no LLM picker), default thresholds. Cost $0.

```bash
menu-check --ocr-json tests/fixtures/sheet_801_vision.json \
  --after tests/fixtures/sheet_391_vision.json \
  --vehicle "2024 Lexus RX 350 20000" --decider-url http://127.0.0.1:18299 \
  --nhtsa-cache out/nhtsa-cache-sheet --ledger out/ledger-sheet-free-1.jsonl \
  --json out/report-sheet-free-1.json
```

The make and model are there because the NHTSA API and the schedule key need them. Printed tables:
`out/run-sheet-free-1.txt`, `out/run-sheet-free-2.txt` (not committed; `out/` is ignored).

**Deterministic:** both runs gave the same route, tool, Decider choice, confidence, P(safety), verdict,
reason and question on all 11 lines; only the milliseconds differ.

| # | Line | Decider tool, conf | P(safety) | Route | Verdict | Reason | Label | Match | Owner |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Engine oil and filter | check_schedule 0.96 | 0.33 | tool | **keep** | in the maker's schedule at 20,000 mi | check_schedule | yes | kept |
| 2 | Premium engine oil conditioner (add-on) | decline_addon 0.87 | 0.13 | tool | **decline** | optional add-on | decline_addon | yes | declined |
| 3 | Tire rotation | check_schedule 0.79 | 0.78 | tool | **keep** | in the maker's schedule at 20,000 mi | check_schedule | yes | kept |
| 4 | Cabin filter + sanitizer | check_schedule 0.52 | 0.16 | tool | **ask** | the maker lists a cabin air filter at 20,000 mi; the sanitizer bundled with it is not listed | check_schedule or decline_addon | yes | declined |
| 5 | Multi-point inspection | check_schedule 0.79 | 0.66 | tool | **keep** | in the maker's schedule at 20,000 mi (the inspect items) | check_schedule | yes | kept |
| 6 | Lifetime protection plan | decline_addon 0.68 | 0.17 | ask_advisor | **ask** | below the 0.8 decline threshold | decline_addon or ask_about_fee | yes | kept |
| 7 | Wheel alignment | owner_decides 0.87 | 0.89 | owner_decides | **your call** | driving-safety item | owner_decides | yes | kept |
| 8 | Crankcase cleaning service | check_schedule 0.39 | 0.13 | ask_advisor | **ask** | below every threshold | decline_addon | no | declined |
| 9 | A/C evaporator cleaning | decline_addon 0.62 | 0.16 | ask_advisor | **ask** | below the 0.8 decline threshold | decline_addon | yes | declined |
| 10 | Shop charges $78.29 | ask_about_fee 0.98 | 0.07 | tool | **ask** | ask what it covers and whether it is optional | ask_about_fee | yes | (charge) |
| 11 | Fees $1.99 | ask_about_fee 0.97 | 0.08 | tool | **ask** | ask what it covers and whether it is optional | ask_about_fee | yes | (charge) |

Line texts as OCR read them: "REPLACE Engine Oil and Oil Filter", "INSTALL Premium Engine Oil
Conditioner", "PERFORM Tire Rotation", "REPLACE Cabin Filter, Incl. Sanitizer (EPA Registered Virucide,
Furgicide, Kills Odors)", "PERFORM Multi-Point Inspection", "ASK Your Service Advisor about the FREE
Lifetime Protection Planf", "PERFORM Wheel Alignment", "PERFORM Crankcase Cleaning Service", "PERFORM A/C
Evaporator System Cleaning Service with Climate control disinfectant" ("Furgicide" and "Planf" are OCR
misreads, passed to Decider as read). Confidences rounded from the report (`0.9608`, `0.8744`, `0.7887`,
`0.518`, `0.7898`, `0.6803`, `0.8652`, `0.3912`, `0.6209`, `0.9781`, `0.9705`). Labels:
`tests/fixtures/sheet_labels.json`.

Advisor questions (only `ask` and `your call` lines get one; all from code, source `template`):

- 4: "The maker's schedule lists a cabin air filter at 20,000 mi: can you do just that, without the
  sanitizer?"
- 6: "What is the lifetime protection planf for, and is it required by the maker's schedule?"
- 7: "What did the inspection find on the wheel alignment, and how soon does it need doing?"
- 8: "What is the crankcase cleaning service for, and is it required by the maker's schedule?"
- 9: "What is the a/c evaporator system cleaning service with climate control disinfectant for, and is it
  required by the maker's schedule?"
- 10: "What does the shop charges line of $78.29 cover, and can it be removed?"
- 11: "What does the fees line of $1.99 cover, and can it be removed?"

Summary (all from code): "Keep 3 · Decline 1 · Ask 6 · Your call 1. Sheet total $801.05. After the
owner's declines: $391.38, $409.67 less. 1 safety line is your call."

- Decider's top tool is one the label accepts on **10 of 11** lines; the miss is the crankcase cleaning
  (check_schedule at 0.39, below every threshold, so nothing was acted on: "ask the advisor").
- The pipeline acted on 8 lines without a general model (route `tool` or `owner_decides`); all 8 match
  the labels. Lines 6, 8 and 9 were below their threshold and, with no LLM picker, became "ask the
  advisor".
- Safety: wheel alignment (P(safety) 0.89) went to the owner. Tire rotation is flagged too (0.78); it is
  kept because the maker lists it. Had a schedule said "not listed" there, the decline would not be taken
  (`tests/test_maker_schedule.py`, the tire-rotation case at 0.78 becomes "your call").
- Time: the server answers one request at a time, so the 11 concurrent lines queue: about 345 to 356 ms
  per line after the first. Whole Decider phase: **4,150 ms** and **4,234 ms**.
- Against the owner: decline on 1 of the 4 lines the owner declined (the conditioner), ask on the other
  three; nothing the owner kept was declined.

## Bedrock run (paid, by the orchestrator)

One run with `--allow-bedrock --max-usd 0.10` (Nova Lite as picker and writer), at commit c3fb981, same
fixtures and flags otherwise: `out/run-rx350-nova-2.txt`, `out/report-rx350-nova-2.json`,
`out/ledger-rx350-nova-2.jsonl`. Printed: `Bedrock spend: $0.0006 of $0.10 cap, 9 calls`;
`11 lines decided in 4012 ms`. (That run predates the neutral fee reason, so it printed "A fee rather
than a service: ask what it covers and whether it is optional." on lines 6, 10 and 11.)

Lines 1 to 5, 7, 10 and 11 were routed by Decider exactly as in the free run. The three lines Decider was
unsure about went to the Nova picker:

| # | Line | Route | Picker's tool | Verdict | Reason as printed |
|---|---|---|---|---|---|
| 6 | Lifetime protection plan | ask_llm | ask_about_fee | ask | A fee rather than a service: ask what it covers and whether it is optional. |
| 8 | Crankcase cleaning service | ask_llm | decline_addon | decline | Optional add-on: the maintenance schedule does not require it. |
| 9 | A/C evaporator cleaning | ask_llm | check_schedule | decline | Not in the maker's schedule at 20,000 mi. |

Card: `Keep 3 · Decline 3 · Ask 4 · Your call 1`.

Against the owner's real declines, as printed:

```
Owner declined 4 lines; the agent said decline on 3 of them:
  line 2: decline (decline_addon) · INSTALL Premium Engine Oil Conditioner
  line 4: ask (check_schedule) · REPLACE Cabin Filter, Incl. Sanitizer (EPA Registered Virucide, Fungicide, Kills Odors)
  line 8: decline (decline_addon) · PERFORM Crankcase Cleaning Service
  line 9: decline (check_schedule) · PERFORM A/C Evaporator System Cleaning Service with Climate control disinfectant
```

Nothing the owner kept was declined (`agent_declined_owner_kept` is empty in the report).

Advisor questions, all with `question_source: "bedrock"` (no rephrasing was rejected in this run):

- 4: "Could you please replace the cabin air filter as per the manufacturer's schedule at 20,000 miles,
  without the sanitizer?"
- 6: "Could you please explain what the lifetime protection plan covers and if it's possible to remove it?"
- 7: "What did the inspection find regarding the wheel alignment, and how soon should it be addressed?"
- 10: "Could you please explain what the $78.29 shop charges cover and if it's optional?"
- 11: "Could you please explain what the $1.99 fee covers and if it's possible to opt out of it?"

Summary, `summary_source: "bedrock"` (the last sentence is Nova's, the rest is code): "Keep 3 · Decline 3 ·
Ask 4 · Your call 1. Sheet total $801.05. After the owner's declines: $391.38, $409.67 less. 1 safety line
is your call. Consider discussing the lifetime protection plan to understand its coverage and optional
status."

### Why the writer only rephrases: the first paid run

The first paid run (`out/run-rx350-nova-1.txt`, before commit 4e0a5c8) let Nova write each question and
the summary from the facts. Its text, as printed:

- on a keep line (oil and filter): "Does the engine oil and oil filter change service correspond to any
  recall campaign for my 2024 Lexus RX 350?" (a recall topic nobody raised, and the make)
- on the cabin filter: "Does the "REPLACE Cabin Filter, Incl. Sanitizer" service include any recall
  components?"
- on the wheel alignment: "Does the Lexus RX 350's wheel alignment service need to be performed at this
  time?"
- the summary: "The total cost of the menu is 80.28 USD, and the total cost of the declined services is
  0.0 USD." ($80.28 is the code's sum of the only two priced lines, the shop charges and the fees; the
  sheet total is $801.05.)

Since 4e0a5c8 code drafts every question (only for `ask` and `your call` lines), the writer gets the draft
and the facts without the make, and code rejects any rephrasing that mentions a recall without a campaign
number, names the make or model, carries a number that is not in the facts, is not a question or is over
160 characters; the draft is used instead and the reason recorded. Every number in the summary is
composed in code; the writer may add one sentence with no digits. `tests/test_writing.py` feeds these
exact strings through a fake Converse client: all rejected.

## Totals (from the sheet, not from line prices)

| | Recommended sheet | After declines | Difference |
|---|---|---|---|
| Menu Total | $672.00 | $309.85 | $362.15 |
| Shop Charges | $78.29 | $55.77 | $22.52 |
| Tax | $48.77 | $23.77 | $25.00 |
| Fees | $1.99 | $1.99 | $0.00 |
| **TOTAL** | **$801.05** | **$391.38** | **$409.67** |

The "up to $6,000" on the protection badge and the "$ 63.21 / month" financing offer are neither lines
nor totals.

## Recalls card and the NHTSA finding

As printed in the free run (the vehicle label comes from the make and model given for NHTSA):

```
Recalls · 2024 Lexus RX 350 · NHTSA, by model, not by VIN
  NHTSA returns no campaign for model "RX 350"; model family "RX" has 4, of which 3 name the RX 350 in their summary (by model, not by VIN).
  family "RX": kept 24V482000, 25V059000, 26V162000; not naming the RX 350: 25V744000
  24V482000  2024-06-27  SEATS:FRONT ASSEMBLY:HEAD RESTRAINT
  25V059000  2025-02-06  SEAT BELTS:REAR/OTHER:WEBBING
  26V162000  2026-03-18  BACK OVER PREVENTION:SOFTWARE
  Ask: "Are there open recalls on my VIN?"
```

(One-line summaries printed under each campaign are left out here.) 25V744000 (rearview camera freeze)
is not kept because its summary says "Please see the recall report for a complete list of models"
instead of naming the model.

For model "RX 350" the recalls API answers **HTTP 400** with an ordinary body
`{"Count":0,"Message":"Results returned successfully","results":[]}` (also for a made-up model), while
"RX" answers 200 with four campaigns. The client reads that 400 as "no campaigns". A naive agent would
have told the owner either "no recalls" or "lookup failed"; three campaigns name this car's model.

## What this does not show

- Eleven lines from one visit are an example, not an accuracy measure. The measure stays the 103
  labelled synthetic lines in the README.
- The labels are our judgement from the line text (two lines accept two tools); the owner's declines
  are one owner's choices on one day, not ground truth for what a line "is".
- The maker's schedule is entered for this one car and this one service (20,000 mi), without the
  special-operating-conditions items. Other cars and mileages get the illustrative table, which only keeps
  or asks.
- The recalls card lists campaigns for the model and year. Whether any is open on this car needs the
  VIN, which is why the card only asks.
- OCR is Apple Vision on redacted photos, not Textract on the original sheet; redaction split some
  printed rows into two OCR blocks, which the parser joins back.
- One paid run with the current writer; its picker choices on lines 6, 8 and 9 are one sample of a
  general model, not a measure (the race in the README measures the picker on 103 lines).
