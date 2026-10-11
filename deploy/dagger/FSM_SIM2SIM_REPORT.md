# Go1 deployment FSM sim2sim validation

Date: 2026-10-09

This report covers the simulation candidate only.  None of these FSM changes or
gains have been copied to the NX real-robot controller yet.

## Configuration

- Student: `dagger_raycaster_v3_ft2.pt`
- MuJoCo physics: 250 Hz (`dt=0.004`)
- Policy: 50 Hz
- Fixed actuator delay: 1 physics step
- Pose control: Kp/Kd = 50/5
- RL control: Kp/Kd = 35/0.5
- Damping: Kd = 1
- Stairs: 0.12 m rise, 0.30 m tread
- Velocity limits used for diagnostics: hip/thigh 30.1 rad/s, calf 20.06 rad/s

The run directory has no `params/env.yaml`.  Therefore the script used the
documented nominal dynamics rather than a recoverable copy of the exact training
configuration.  This is an important limitation of the validation.

## Results

| Test | Result | RL trunk contacts / falls | Peak joint speed | Velocity-limit exceedance |
|---|---|---:|---:|---:|
| Full FSM cycle on flat | PASSIVE -> STAND -> RL -> SITDOWN -> DAMPING -> PASSIVE completed | 0 / 0 | 1.40 rad/s | 0 steps |
| Flat, vx=0.4 m/s, 30 s RL | stable locomotion | 0 / 0 | 9.08 rad/s | not recorded by the older run |
| Stairs, vx=0.3 m/s, 30 s RL | ascent and partial descent; no fall | 0 / 0 | 26.13 rad/s (RL calf) | 2 of 1500 RL steps |
| Stairs, vx=0.4 m/s, 30 s RL | full up/platform/down traversal | 0 / 0 | 17.55 rad/s (RL calf) | 0 of 1500 RL steps |

The 0.3 m/s run having a larger peak than the 0.4 m/s run shows that peak joint
speed is gait/contact-phase dependent; lowering the forward command alone is not
a sufficient safety measure.  The two violations at 0.3 m/s were isolated, but
they remain a transfer risk because MuJoCo does not enforce the URDF velocity
limits as hard constraints.

The corrected 0.4 m/s stair video is:

`logs/rsl_rl/rough_direct/2026-09-09_01-41-12/videos/dagger_play/mj_sim2sim_go1_dagger_raycaster_v3_ft2_fsm.mp4`

## Issues found and fixed in the simulation candidate

1. An explicit RL -> SITDOWN request was previously pre-empted by the released
   RL-enable latch and entered DAMPING. Explicit controlled descent now has priority.
2. Student actions predicted while STAND/SITDOWN was active were incorrectly
   written into `previous_action`, although they were never executed.  Non-RL
   modes now keep that observation at zero.  This reduced the first RL-cycle
   stair transition error from about 0.96 rad to 0.42 rad.
3. FSM control now uses torque actuators and explicit per-physics-step PD, so
   PASSIVE, DAMPING and state-specific gains have the same semantics as LowCmd.
4. Joint speed-limit exceedance count is now part of `FSM_METRICS`.

## Current conclusion

The state transitions and nominal locomotion are functionally correct in
sim2sim, including smooth stand/sit interpolation and a successful 12 cm stair
traversal.  Before copying this candidate to the real robot, run a small matrix
over actuator delays, lateral/yaw commands and repeated stair trials, and decide
on a real-time joint-velocity watchdog for the NX controller.
