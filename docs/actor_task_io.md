# Actor task: proposed inputs and outputs

Status: design proposal, nothing implemented yet. Every item below is
**proposed, to confirm with team**.

The actor is the second network in the two-network model. The planner
(`StackTask`) remembers the target sequence and says where to reach; the actor
drives the MotorNet arm there and reports when it has arrived. The actor is
based on the existing `RandomTarget` task.

## Context inputs (3 per timestep)

- `TargetX`, `TargetY`: target direction in unit-circle coordinates, identical
  to the planner's outputs. *(proposed, to confirm with team)*
- `TargetValid`: on while the planner is presenting a target. It also acts as
  the go signal; there is no separate go cue. *(proposed, to confirm with team)*
- The target stays on for as long as `TargetValid` is on. `RandomTarget`
  removes the target at go, which forces the network to remember it; here that
  memory lives in the planner, not the actor. *(proposed, to confirm with team)*

## Observation (the "?" in the schematic, 14 per timestep)

- MotorNet arm feedback, unchanged from `RandomTarget`: fingertip x, y (vision,
  50 ms delay) plus 6 muscle lengths and 6 muscle velocities (proprioception,
  20 ms delay). *(proposed, to confirm with team)*
- Total network input is therefore 17 per timestep: 14 observation + 3 context.
  *(proposed, to confirm with team)*

## Outputs (7 per timestep)

- 6 muscle activations, sent to the arm exactly as in `RandomTarget`.
  *(proposed, to confirm with team)*
- `Trigger`: one extra output that is **not** sent to the arm.
  *(proposed, to confirm with team)*

## Workspace

- Center-out: every trial starts at the same fixed arm posture.
  *(proposed, to confirm with team)*
- Hand goal = center + R * (`TargetX`, `TargetY`), where center is the fingertip
  position at the starting posture and R is the reach distance.
  *(proposed, to confirm with team)*

## Trigger

- `Trigger` should emit a 40 ms pulse (4 timesteps) once the hand has stayed
  within `reach_radius` of the goal for 100 ms (10 timesteps).
  *(proposed, to confirm with team)*
- The correct `Trigger` trace is not precomputed. The loss computes it on each
  trial from the hand path the arm actually took.
  *(proposed, to confirm with team)*

## Timing

- 10 ms timesteps (the MotorNet default, same as `RandomTarget`).
  *(proposed, to confirm with team)*
- The planner currently runs at `bin_ms=20`. It will need `bin_ms=10` for
  Loop 3 so the two networks share a clock.
  *(proposed, to confirm with team)*

## Trial types

- v1: one target per trial, plus 20% catch trials in which `TargetValid` never
  turns on and the arm must hold still at the start posture.
  *(proposed, to confirm with team)*
- v2, later: 2-3 targets per trial. *(proposed, to confirm with team)*

## Open questions

- **R and the start posture.** Which posture, and what R keeps the whole circle
  of goals inside the arm's reachable space?
- **`reach_radius`.** What value, and is it fixed or tightened during training?
- **Trial length.** `RandomTarget` uses 3 s (300 timesteps). Same here? When
  can `TargetValid` turn on, and how long does it stay on?
- **After `TargetValid` turns off.** Does the hand hold at the goal, return to
  center, or is it unconstrained?
- **Catch trials.** What do `TargetX` and `TargetY` carry when `TargetValid` is
  off: zeros, or whatever the planner happens to output?
- **Input scaling and noise.** `RandomTarget` uses a go cue of 0.15 and adds
  noise (0.04) to its context inputs. What amplitude does `TargetValid` use,
  and is noise added to match what the planner really produces?
- **Training inputs.** Is the actor trained on synthetic clean inputs, on
  recorded planner outputs, or both?
- **Trigger when the hand never arrives.** Is the correct trace all zeros for
  that trial? Should a late arrival still earn a pulse?
- **Trigger loss weighting.** A pulse is 4 of roughly 300 timesteps, so an
  unweighted loss is satisfied by never firing. How is the pulse weighted?
- **Trigger and feedback delay.** The loss sees the true hand position, but the
  network sees vision delayed by 50 ms. Is the pulse expected at the true
  arrival time, or shifted to allow for the delay?
- **Where Trigger goes.** In Loop 3, does `Trigger` feed back into the planner
  to advance it to the next target, and in what form?
- **Bumps.** `RandomTarget` applies a random force to the hand on half the
  trials. Keep, drop, or make optional?
- **Loss curriculum.** `RandomTarget` only scores the early part of each trial
  at first and grows the window over 200 epochs (`full_trial_epoch`). Keep it?
- **Shared code.** An output that is not sent to the arm needs changes in
  `task_wrapper.py` and `task_training.py`, which every task uses. Change them
  behind a default-off option, or subclass the wrapper for the actor?
- **Planner at `bin_ms=10`.** Does halving the bin size require retraining the
  planner, and does its trial still fit in 300 timesteps?
- **v2 sequencing.** Between targets, does `TargetValid` drop, and is there one
  `Trigger` pulse per target?
