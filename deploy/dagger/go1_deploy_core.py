"""Shared Go1 policy/FSM core used by both MuJoCo and the physical robot.

This file deliberately supports Python 3.6 (JetPack on the Go1 NX).  It owns
all control semantics; backends may only acquire state/depth, run the network,
and apply the returned joint command.
"""
from __future__ import print_function

import hashlib
import json

import numpy as np


PASSIVE = "PASSIVE"
DAMPING = "DAMPING"
STAND = "STAND"
SITDOWN = "SITDOWN"
RL = "RL"

POLICY_JOINT_NAMES = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
)
SDK_JOINT_NAMES = (
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
)
POLICY_TO_SDK = np.array([3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8], dtype=np.intp)
DEFAULT_JOINT = np.array([0.0] * 4 + [0.9] * 4 + [-1.8] * 4, dtype=np.float32)
COMMAND_LIMIT = np.array([0.8, 0.4, 0.8], dtype=np.float32)
SDK_POSITION_LOWER = np.tile((-1.047, -0.663, -2.721), 4).astype(np.float32)
SDK_POSITION_UPPER = np.tile((1.047, 2.966, -0.837), 4).astype(np.float32)
SDK_VELOCITY_LIMIT = np.tile((30.1, 30.1, 20.06), 4).astype(np.float32)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def core_sha256():
    return file_sha256(__file__)


def verify_bundle(manifest_path, config_path):
    with open(manifest_path) as source:
        manifest = json.load(source)
    actual = {"core_sha256": core_sha256(),
              "config_sha256": file_sha256(config_path)}
    for name, value in actual.items():
        if manifest.get(name) != value:
            raise RuntimeError("Deployment bundle {} mismatch: expected {}, got {}".format(
                name, manifest.get(name), value))
    return actual


def vector(name, value, size):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError("{} must be {} finite values".format(name, size))
    return result


def policy_to_sdk(value):
    policy = vector("policy joint vector", value, 12)
    sdk = np.empty(12, dtype=np.float32)
    sdk[POLICY_TO_SDK] = policy
    return sdk


def sdk_to_policy(value):
    return vector("SDK joint vector", value, 12)[POLICY_TO_SDK].copy()


def sdk_position_limit(q):
    return np.clip(vector("q", q, 12), SDK_POSITION_LOWER,
                   SDK_POSITION_UPPER).astype(np.float32)


def metric_depth(raw_z16, scale, rows, columns):
    """Convert native D435 Z16 into clipped optical-Z sampled at 106x60."""
    raw = np.asarray(raw_z16)
    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise ValueError("Expected a two-dimensional uint16 depth image")
    sampled = raw[rows[:, None], columns[None, :]]
    depth = sampled.astype(np.float32) * float(scale)
    depth[sampled == 0] = 2.0
    np.clip(depth, 0.1, 2.0, out=depth)
    return depth


def projected_gravity(quaternion_wxyz):
    q = vector("world-from-base quaternion", quaternion_wxyz, 4)
    norm = np.linalg.norm(q)
    if norm < 1e-6:
        raise ValueError("Quaternion norm is too small")
    q = q / norm
    inverse_xyz = -q[1:]
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    t = 2.0 * np.cross(inverse_xyz, gravity_world)
    return gravity_world + q[0] * t + np.cross(inverse_xyz, t)


def common_frame(gyro_base, quaternion_wxyz, command, q_sdk, dq_sdk,
                 previous_raw_action):
    gyro = vector("base gyroscope", gyro_base, 3)
    cmd = vector("velocity command", command, 3)
    if (np.abs(cmd) > COMMAND_LIMIT + 1e-6).any():
        raise ValueError("Velocity command exceeds student training limits")
    q = sdk_to_policy(q_sdk)
    dq = sdk_to_policy(dq_sdk)
    previous = vector("previous raw action", previous_raw_action, 12)
    result = np.concatenate((gyro, projected_gravity(quaternion_wxyz), cmd,
                             q - DEFAULT_JOINT, dq, previous))
    if result.shape != (45,):
        raise AssertionError("common observation frame must have 45 values")
    return result.astype(np.float32, copy=False)


class ObservationHistory(object):
    def __init__(self):
        self.frames = np.zeros((5, 45), dtype=np.float32)

    def push(self, frame):
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = vector("observation frame", frame, 45)
        return self.frames.reshape(225).copy()

    def reset(self):
        self.frames.fill(0.0)


class DepthHistory(object):
    """Five policy frames with the checkpoint's one-control-step delay."""
    def __init__(self):
        self.frames = np.zeros((6, 60, 106), dtype=np.float32)
        self.index = 0
        self.initialized = False

    def initialize(self, frame):
        value = np.asarray(frame, dtype=np.float32)
        if value.shape != (60, 106) or not np.isfinite(value).all():
            raise ValueError("depth frame must be finite 60x106")
        self.frames[:] = value
        self.index = 0
        self.initialized = True

    def push(self, frame):
        if not self.initialized:
            self.initialize(frame)
        else:
            value = np.asarray(frame, dtype=np.float32)
            if value.shape != (60, 106) or not np.isfinite(value).all():
                raise ValueError("depth frame must be finite 60x106")
            self.index = (self.index + 1) % 6
            self.frames[self.index] = value
        delayed_newest = (self.index - 1) % 6
        indices = (np.arange(5) + delayed_newest - 4) % 6
        return np.ascontiguousarray(self.frames[indices])

    def reset(self):
        self.frames.fill(0.0)
        self.index = 0
        self.initialized = False


class ActionProcessor(object):
    def __init__(self):
        self.previous_raw = np.zeros(12, dtype=np.float32)

    def process(self, network_action):
        """clip raw -> low-pass -> scale -> default -> SDK order."""
        raw = np.clip(vector("network action", network_action, 12), -3.0, 3.0)
        filtered = 0.8 * raw + 0.2 * self.previous_raw
        target_sdk = policy_to_sdk(DEFAULT_JOINT + 0.5 * filtered)
        self.previous_raw = raw.copy()
        return target_sdk

    def reset(self):
        self.previous_raw.fill(0.0)


class JointCommand(object):
    def __init__(self, mode, q, kp, kd):
        self.mode = mode
        self.q = q
        self.kp = kp
        self.kd = kd


class StateMachine(object):
    def __init__(self, stand_pose_sdk, pose_kp, pose_kd, damping_kd,
                 rl_kp, rl_kd, stand_seconds=3.0, sit_seconds=3.0,
                 pose_tolerance=0.15, position_limit=None,
                 rl_entry_seconds=0.5):
        self.position_limit = position_limit or (lambda q: q)
        self.stand_pose = vector("stand pose", self.position_limit(
            vector("stand pose", stand_pose_sdk, 12)), 12).copy()
        self.pose_kp = vector("pose kp", pose_kp, 12).copy()
        self.pose_kd = vector("pose kd", pose_kd, 12).copy()
        self.rl_kp = vector("rl kp", rl_kp, 12).copy()
        self.rl_kd = vector("rl kd", rl_kd, 12).copy()
        self.damping_kd = vector("damping kd", damping_kd, 12).copy()
        if any((x < 0).any() for x in (self.pose_kp, self.pose_kd,
                                       self.rl_kp, self.rl_kd, self.damping_kd)):
            raise ValueError("gains must be nonnegative")
        if min(stand_seconds, sit_seconds, pose_tolerance) <= 0:
            raise ValueError("pose durations and tolerance must be positive")
        if not np.isfinite(rl_entry_seconds) or rl_entry_seconds < 0:
            raise ValueError("RL entry duration must be finite and nonnegative")
        self.stand_seconds = float(stand_seconds)
        self.sit_seconds = float(sit_seconds)
        self.pose_tolerance = float(pose_tolerance)
        self.rl_entry_seconds = float(rl_entry_seconds)
        self.mode = PASSIVE
        self.entry_time = 0.0
        self.entry_q = None
        self.sit_pose = None
        self.pose_complete = False
        self.last_stand_start_pose = None

    def _enter(self, mode, now, measured_q):
        self.mode = mode
        self.entry_time = float(now)
        self.entry_q = vector("limited entry pose", self.position_limit(
            vector("measured q", measured_q, 12)), 12).copy()
        self.pose_complete = False
        if mode == STAND:
            self.last_stand_start_pose = vector("measured q", measured_q, 12).copy()
            self.sit_pose = self.entry_q.copy()

    def step(self, now, measured_q, request=None, rl_target=None,
             state_fresh=True, depth_fresh=True, rl_enabled=False):
        q = vector("measured q", measured_q, 12)
        if request not in (None, PASSIVE, DAMPING, STAND, SITDOWN, RL):
            raise ValueError("unknown state request")
        if not state_fresh and self.mode != PASSIVE:
            self._enter(DAMPING, now, q)
        elif request == DAMPING:
            self._enter(DAMPING, now, q)
        elif request == PASSIVE:
            self._enter(PASSIVE, now, q)
        elif request == SITDOWN and self.mode in (STAND, RL):
            self._enter(SITDOWN, now, q)
        elif self.mode == RL and (not depth_fresh or not rl_enabled or rl_target is None):
            self._enter(DAMPING, now, q)
        elif request == STAND and self.mode in (PASSIVE, DAMPING, SITDOWN):
            if state_fresh:
                self._enter(STAND, now, q)
        elif request == RL and self.mode == STAND and self.pose_complete:
            if state_fresh and depth_fresh and rl_enabled and rl_target is not None:
                self._enter(RL, now, q)

        zeros = np.zeros(12, dtype=np.float32)
        if self.mode == PASSIVE:
            return JointCommand(PASSIVE, zeros, zeros, zeros)
        if self.mode == DAMPING:
            return JointCommand(DAMPING, zeros, zeros, self.damping_kd.copy())
        if self.mode in (STAND, SITDOWN):
            target = self.stand_pose if self.mode == STAND else self.sit_pose
            duration = self.stand_seconds if self.mode == STAND else self.sit_seconds
            phase = np.clip((now - self.entry_time) / duration, 0.0, 1.0)
            alpha = phase * phase * (3.0 - 2.0 * phase)
            desired = self.entry_q + alpha * (target - self.entry_q)
            self.pose_complete = (now - self.entry_time >= duration - 1e-6 and
                                  np.max(np.abs(q - target)) <= self.pose_tolerance)
            return JointCommand(self.mode, desired, self.pose_kp.copy(),
                                self.pose_kd.copy())
        target = vector("RL target", rl_target, 12).copy()
        # A nonzero duration is retained only as an explicit ablation.  Production
        # uses zero so the target actually applied matches the raw-action history
        # observed by the policy, as it does during training.
        if self.rl_entry_seconds > 0.0:
            phase = np.clip((now - self.entry_time) / self.rl_entry_seconds, 0.0, 1.0)
            alpha = phase * phase * (3.0 - 2.0 * phase)
            target = self.entry_q + alpha * (target - self.entry_q)
        target = self.position_limit(target)
        return JointCommand(RL, target, self.rl_kp.copy(), self.rl_kd.copy())


class DeploymentCore(object):
    """Shared policy input, action-state, and FSM transition coordinator."""
    def __init__(self, machine):
        self.machine = machine
        self.observations = ObservationHistory()
        self.depth = DepthHistory()
        self.actions = ActionProcessor()

    def reset_policy_memory(self):
        self.observations.reset()
        self.depth.reset()
        self.actions.reset()

    def prepare_policy_inputs(self, gyro_base, quaternion_wxyz, command,
                              q_sdk, dq_sdk, depth_frame):
        frame = common_frame(gyro_base, quaternion_wxyz, command, q_sdk, dq_sdk,
                             self.actions.previous_raw)
        return self.depth.push(depth_frame), self.observations.push(frame)

    def step(self, now, q_sdk, request, network_action, state_fresh=True,
             depth_fresh=True, rl_enabled=False):
        previous_mode = self.machine.mode
        wants_rl = self.machine.mode == RL or request == RL
        rl_target = None
        if wants_rl and network_action is not None and state_fresh and depth_fresh:
            rl_target = self.machine.position_limit(self.actions.process(network_action))
        result = self.machine.step(now, q_sdk, request=request, rl_target=rl_target,
                                   state_fresh=state_fresh, depth_fresh=depth_fresh,
                                   rl_enabled=rl_enabled)
        if result.mode == STAND and previous_mode != STAND:
            self.observations.reset()
            self.depth.reset()
        if result.mode != RL:
            # Predictions rejected by the FSM were not executed and must never
            # enter the next observation or the low-pass filter state.
            self.actions.reset()
        return result
