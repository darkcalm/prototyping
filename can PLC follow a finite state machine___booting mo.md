<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# can PLC follow a finite state machine?

booting mode: just the booting and startup sequence, goes to detection mode

detection mode: every 3 seconds, a decision is made on what is approximately on the left side of the image, and what is approximately on the right side of the image. if the decision comes to CUP(L:False,R:False), CLAM(L:False,R:False), goes to detection mode. if any of the 4 values are true, then goes to action mode.

action mode: stop detecting. assuming cup should go to R, and clam should go to L, and blocker is placed at BLOCKSIDE, follow the cases below, then go to verification mode:
{CUP(L:False,R:True), CLAM(L:True,R:False)} or {CUP(L:False,R:False), CLAM(L:True,R:False)} or {CUP(L:False,R:True), CLAM(L:False,R:False)}: Case 1

* t=0-1.5s assure separator points to the center
* t=1.5-3s assure tray down
* t=3-6s tray stays down
* t=6-7.5s assure tray up
{CUP(L:False,R:False), CLAM(L:False,R:True)} or {CUP(L:False,R:False), CLAM(L:True,R:True)}: Case 2 (redirect clam)
* t=0-3s assure separator points to R (blocking clam to R)
* t=3-4.5s assure tray down
* t=4.5-7.5s tray stays down
* t=7.5-9s assure tray up
* t=9-10.5s assure separator points to the center
{CUP(L:True,R:False), CLAM(L:False,R:False)} or {CUP(L:True,R:True), CLAM(L:False,R:False)}: Case 3 (redirect cup)
* t=0-3s assure separator points to L (blocking cup to L)
* t=3-4.5s assure tray down
* t=4.5-7.5s tray stays down
* t=7.5-9s assure tray up
* t=9-10.5s assure separator points to the center
{CUP(L:True,R:False), CLAM(L:False,R:True)}: Case 4 (redirect two times to achieve a swap effect)
* t=0-3s assure blocker on
* t=3-6s assure separator points opposite to BLOCKSIDE
* t=6-7.5s assure tray down
* t=7.5-10.5s tray stays down
* t=10.5-12s assure tray up
* t=12-15s assure blocker off
* t=15-18s assure separator points to BLOCKSIDE
* t=18-19.5s assure tray down
* t=19.5-21s tray stays down
* t=21-22.5s assure tray up
* t=22.5-24s assure separator points to the center
IMPORTANT: create a data structure to allow quick programming of time and action

verification mode: no time limit, wait until the human verifies the result in the terminal running this python. once there is verification, goes back to detection mode.

Yes, a PLC is actually a perfect platform for a finite state machine, and what you describe (booting → detection → action → verification) maps very cleanly to an FSM implemented in Structured Text, SFC, or even Ladder using a state variable.[^1_1][^1_2]

Below is a compact proposal of:

- State definitions and transitions for your four modes
- A time‑driven “recipe” data structure for the action sequences (the “IMPORTANT” requirement)
- How to index that structure from your CUP/CLAM detection

I will assume Structured Text, but the idea is identical in any IEC 61131‑3 language.[^1_3][^1_4]

***

## High‑level FSM states

Define an enum (or constants) for your main states:

- `STATE_BOOT`
- `STATE_DETECT`
- `STATE_ACTION`
- `STATE_VERIFY`

Then one state variable and one “next state” variable:

- `State : INT;`
- `NextState : INT;`

Run a CASE on `State` every PLC cycle; only the active state’s code runs.[^1_5][^1_1]

***

## Encoding the cases (1–4)

First normalize the vision result into:

- `CupL, CupR : BOOL`
- `ClamL, ClamR : BOOL`
- `BlockSide : (LEFT, RIGHT)` or simple BOOL

Then map to a symbolic case:

- `CASE_NONE`          (no object → stay in DETECT)
- `CASE_1`, `CASE_2`, `CASE_3`, `CASE_4`

Example mapping in pseudocode:

```text
IF NOT CupL AND CupR AND ClamL AND NOT ClamR
   OR NOT CupL AND NOT CupR AND ClamL AND NOT ClamR
   OR NOT CupL AND CupR AND NOT ClamL AND NOT ClamR
THEN CaseID := CASE_1;
ELSIF NOT CupL AND NOT CupR AND NOT ClamL AND ClamR
   OR NOT CupL AND NOT CupR AND ClamL AND ClamR
THEN CaseID := CASE_2;
ELSIF CupL AND NOT CupR AND NOT ClamL AND NOT ClamR
   OR CupL AND CupR AND NOT ClamL AND NOT ClamR
THEN CaseID := CASE_3;
ELSIF CupL AND NOT CupR AND NOT ClamL AND ClamR
THEN CaseID := CASE_4;
ELSE
   CaseID := CASE_NONE;
END_IF;
```

If `CaseID = CASE_NONE`, stay in `STATE_DETECT`. Otherwise, move to `STATE_ACTION` and freeze the decision (store `ActiveCase := CaseID;`).[^1_6][^1_1]

***

## Core: time–action “recipe” data structure

Use a generic structure for each timed step:

```pascal
TYPE T_ActionStep :
STRUCT
    tStart  : REAL;   // seconds from start of action mode
    tEnd    : REAL;   // seconds from start of action mode
    SepCmd  : INT;    // 0=center, 1=L, 2=R, -1=ignore
    TrayCmd : INT;    // 0=up, 1=down, -1=hold/ignore
    Blocker : INT;    // 0=off, 1=on, -1=ignore
END_STRUCT
END_TYPE;
```

Then a table per case (fixed‑size arrays; unused entries can have `tStart=tEnd=0`):

- `Case1Steps : ARRAY[1..4] OF T_ActionStep;`
- `Case2Steps : ARRAY[1..5] OF T_ActionStep;`
- `Case3Steps : ARRAY[1..5] OF T_ActionStep;`
- `Case4Steps : ARRAY[1..9] OF T_ActionStep;`

Populate them once in initialization (e.g., in `STATE_BOOT` or via retained constants):

**Case 1** (times from your description):

1. 0–1.5 s: separator center, tray up, blocker ignore
2. 1.5–3 s: tray down
3. 3–6 s: tray stays down (you can encode as `TrayCmd=1` again or `-1` if an outer latch holds it)
4. 6–7.5 s: tray up

Similarly define Case 2–4, translating your timing table literally into `tStart/tEnd` and command codes. This is the “quick programming” layer: to tune timings, you only edit the arrays, not the logic.[^1_1][^1_6]

If `BLOCKSIDE` affects direction, do not encode “left/right” directly in the recipe; encode logical commands like “TO_BLOCKSIDE” vs “OPPOSITE_BLOCKSIDE” and translate once:

```pascal
SepCmd : INT;  // 0=center, 1=toBlockSide, 2=oppositeBlockSide
```

In the execution layer, resolve to actual outputs depending on `BlockSide`.

***

## Executing the recipe in ACTION state

In `STATE_ACTION`:

- On entry: latch `ActionStartTime := Now_s` (a REAL seconds counter) and choose a pointer to the correct step array for `ActiveCase`.
- Every PLC cycle:
    - `t := Now_s - ActionStartTime`
    - Loop through the steps for the active case:
        - If `t >= tStart` AND `t < tEnd`, apply that step’s commands:
            - If `SepCmd <> -1` then set separator target.
            - If `TrayCmd <> -1` then set tray up/down.
            - If `Blocker <> -1` then set blocker on/off.
    - When `t` is greater than the last `tEnd` in that case’s table, transition `State := STATE_VERIFY`.

This way:

- Timings are pure data.
- Output logic is central and generic.
- Adding new cases is just adding new step arrays.[^1_6][^1_1]

***

## Detection and verification states

- **BOOT**: Initialize arrays, zero timers, then `State := STATE_DETECT`.
- **DETECT**:
    - Every 3 s (simple timer) run detection and update CUP/CLAM booleans.
    - Compute `CaseID`. If `CASE_NONE`, keep waiting. If any case 1–4, set `ActiveCase` and go to `STATE_ACTION`.
- **VERIFY**:
    - Keep outputs in their last commanded state; no timeout.
    - Poll a BOOL or INT coming from your Python terminal (“verification received”, maybe with “OK/NOT OK” code).
    - When verified, reset small things as needed and go back to `STATE_DETECT`.

All of the above is exactly the typical PLC FSM pattern (state int + CASE + per‑state logic), and matches common guidance for structured state‑based control in IEC 61131‑3.[^1_2][^1_7][^1_1]

If you want, the next step can be concrete: a full IEC 61131‑3 Structured Text skeleton with type/VAR/CASE blocks that you can paste into TwinCAT/CodeSys and then hook to your Python side.
<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_8][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: https://www.linkedin.com/pulse/efficient-plc-programming-using-finite-state-machine-concept-terzija

[^1_2]: https://stefanhenneken.net/2018/11/17/iec-61131-3-the-state-pattern/

[^1_3]: https://www.plcacademy.com/structured-text-tutorial/

[^1_4]: https://en.wikipedia.org/wiki/IEC_61131-3

[^1_5]: https://www.reddit.com/r/PLC/comments/pfbw9p/plc_state_machine/

[^1_6]: https://www.hemelix.com/plc/state-machine-in-structured-text/

[^1_7]: http://www.inf.ufsc.br/~joao.dovicchi/pos-ed/pos/exerc/machines2.pdf

[^1_8]: https://www.linkedin.com/pulse/art-finite-state-machines-plc-programming-testing-episode-nouh-05bic

[^1_9]: https://www.solisplc.com/tutorials/programming-a-state-machine-in-ladder-logic

[^1_10]: https://www.splatco.com/tips/fsm/fsmarticl.pdf

[^1_11]: https://peer.asee.org/teaching-finite-state-machines-fsms-as-part-of-a-programmable-logic-control-plc-course.pdf

[^1_12]: https://stefanhenneken.net/2018/09/16/iec-61131-3-das-state-pattern/

[^1_13]: https://www.youtube.com/watch?v=XmcXRZXPRWs

[^1_14]: https://www.youtube.com/watch?v=nJ1vrhm1_Zo

[^1_15]: https://forum.digikey.com/t/plc-finite-state-machine-control-using-a-sequence-cam-block/37783

[^1_16]: https://www.stateworks.com/technology/overcoming-limitations-plc-programming/

[^1_17]: https://stackoverflow.com/questions/78447504/clean-code-plc-programming-in-structured-text-hsm-engine

[^1_18]: https://www.youtube.com/watch?v=DT-xznz-sOQ

[^1_19]: https://www.reddit.com/r/PLC/comments/yrj52m/hierarchical_state_machine_hsm_using_structured/

[^1_20]: https://www.linkedin.com/pulse/finite-state-machine-practical-example-bojan-terzija

