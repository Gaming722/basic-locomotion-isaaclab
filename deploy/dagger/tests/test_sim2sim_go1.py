"""Run with the sim2sim Python: python -m unittest discover -s deploy/dagger/tests."""
import os
import sys
import tempfile
import subprocess
import unittest
from unittest.mock import patch
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import numpy as np
import mujoco
import torch
from play_mujoco_dagger_go1 import (build_go1_scene_xml, DaggerNet,
                                  D435_DEPTH_QUATERNION_ROS_WXYZ, CAM_POS, _sync_realtime,
                                  GO1_VELOCITY_LIMIT_SDK, _make_key_callback,
                                  _parse_fsm_script)
from go1_state_machine import DAMPING, PASSIVE, RL, SITDOWN, STAND, StateMachine
from mujoco_depth import MujocoDepthCamera


class Sim2SimTests(unittest.TestCase):
    def test_keyboard_training_limits(self):
        for amplitudes in ([.8, .4, .8], [.2, .1, .3]):
            ctl = dict(vx=0., vy=0., wz=0.)
            callback = _make_key_callback(ctl, amplitudes)
            for axis, positive, negative, limit in zip(
                    ('vx', 'vy', 'wz'), (265, 263, 81), (264, 262, 69), amplitudes):
                for _ in range(30):
                    callback(positive)
                self.assertAlmostEqual(ctl[axis], limit)
                for _ in range(30):
                    callback(negative)
                self.assertAlmostEqual(ctl[axis], -limit)
            callback(67)
            self.assertAlmostEqual(ctl['vx'], min(.4, amplitudes[0]))
            callback(86)
            self.assertEqual([ctl[k] for k in ('vx', 'vy', 'wz')], [0, 0, 0])
            callback(256)
            self.assertTrue(ctl['quit'])
            callback(66)
            self.assertTrue(ctl['sensor_blind'])
            callback(66)
            self.assertFalse(ctl['sensor_blind'])
            callback(70)
            self.assertTrue(ctl['depth_frozen'])
            self.assertIsNone(ctl['frozen_depth'])
            ctl['frozen_depth'] = object()
            callback(70)
            self.assertFalse(ctl['depth_frozen'])
            self.assertIsNone(ctl['frozen_depth'])

    def test_realtime_deadline(self):
        with patch('play_mujoco_dagger_go1.time.perf_counter', return_value=10.005), \
                patch('play_mujoco_dagger_go1.time.sleep') as sleep:
            _sync_realtime(10, .02)
            self.assertAlmostEqual(sleep.call_args.args[0], .015)
        with patch('play_mujoco_dagger_go1.time.perf_counter', return_value=10.03), \
                patch('play_mujoco_dagger_go1.time.sleep') as sleep:
            _sync_realtime(10, .02)
            sleep.assert_not_called()

    def test_optical_extrinsics_and_heightfield(self):
        w, x, y, z = D435_DEPTH_QUATERNION_ROS_WXYZ
        gl_quat = np.array([-x, w, z, -y])
        ros_R, gl_R = np.zeros(9), np.zeros(9)
        ros_quat = np.array([w, x, y, z])
        ros_quat /= np.linalg.norm(ros_quat)
        gl_quat /= np.linalg.norm(gl_quat)
        mujoco.mju_quat2Mat(ros_R, ros_quat)
        mujoco.mju_quat2Mat(gl_R, gl_quat)
        # GL right/up/back equal ROS right/-down/-forward.
        np.testing.assert_allclose(gl_R.reshape(3,3), ros_R.reshape(3,3) @ np.diag([1,-1,-1]), atol=1e-12)
        m = mujoco.MjModel.from_xml_string(build_go1_scene_xml(
            scene='perlin', perlin_amp=.18, camera_quaternion=gl_quat))
        np.testing.assert_allclose(m.cam_pos[0], CAM_POS)
        np.testing.assert_allclose(m.hfield_size[0, 2:], [.18, .001])

    def test_gains_and_intrinsics(self):
        K = [[55.85, 0, 60], [0, 55.85, 25], [0, 0, 1]]
        m = mujoco.MjModel.from_xml_string(build_go1_scene_xml(camera_intrinsics=K))
        np.testing.assert_allclose(m.actuator_gainprm[:, 0], 35)
        np.testing.assert_allclose(m.dof_damping[6:], .5)
        np.testing.assert_allclose(m.dof_armature[6:], .01)
        self.assertFalse(m.actuator_ctrllimited.any())
        np.testing.assert_allclose(m.cam_intrinsic[0], [55.85, 55.85, -7, 5])

    def test_fsm_uses_torque_actuators_and_state_specific_gains(self):
        m = mujoco.MjModel.from_xml_string(build_go1_scene_xml(torque_control=True))
        np.testing.assert_allclose(m.actuator_gainprm[:, 0], 1)
        np.testing.assert_allclose(m.dof_damping[6:], 0)
        self.assertTrue(m.actuator_forcelimited.all())
        np.testing.assert_allclose(m.actuator_forcerange[::3],
                                   np.tile([-23.7, 23.7], (4, 1)))
        np.testing.assert_allclose(m.actuator_forcerange[2::3],
                                   np.tile([-35.55, 35.55], (4, 1)))
        np.testing.assert_allclose(GO1_VELOCITY_LIMIT_SDK,
                                   np.tile((30.1, 30.1, 20.06), 4))
        ones = np.ones(12)
        machine = StateMachine(np.tile((0, .9, -1.8), 4), ones*50, ones*5,
                               ones, rl_kp=ones*35, rl_kd=ones*.5,
                               stand_seconds=1, sit_seconds=1)
        q = np.tile((-.2, 1.2, -2.7), 4)
        self.assertEqual(machine.step(0, q).mode, PASSIVE)
        stand = machine.step(0, q, request=STAND)
        np.testing.assert_allclose(stand.kp, 50)
        machine.step(1, machine.stand_pose)
        rl = machine.step(1.02, machine.stand_pose, request=RL,
                          rl_target=machine.stand_pose, rl_enabled=True)
        self.assertEqual(rl.mode, RL)
        np.testing.assert_allclose(rl.kp, 35)
        direct = StateMachine(np.tile((0, .9, -1.8), 4), ones*50, ones*5,
                              ones, rl_kp=ones*35, rl_kd=ones*.5,
                              stand_seconds=1, sit_seconds=1,
                              rl_entry_seconds=0.0)
        direct.step(0, q, request=STAND)
        direct.step(1, direct.stand_pose)
        direct_target = direct.stand_pose + 0.1
        direct_rl = direct.step(1.02, direct.stand_pose, request=RL,
                                rl_target=direct_target, rl_enabled=True)
        np.testing.assert_allclose(direct_rl.q, direct_target)
        sit = machine.step(1.03, machine.stand_pose, request=SITDOWN,
                           rl_target=machine.stand_pose, rl_enabled=False)
        self.assertEqual(sit.mode, SITDOWN)
        machine.step(2.03, machine.sit_pose)
        machine.step(2.04, machine.sit_pose, request=STAND)
        machine.step(3.04, machine.stand_pose)
        machine.step(3.05, machine.stand_pose, request=RL,
                     rl_target=machine.stand_pose, rl_enabled=True)
        self.assertEqual(machine.step(3.06, q, rl_enabled=False).mode, DAMPING)

    def test_fsm_keyboard_modes(self):
        ctl = dict(vx=0., vy=0., wz=0., rl_enabled=False)
        callback = _make_key_callback(ctl, [.8, .4, .8], fsm=True)
        for key, mode in ((49, STAND), (50, SITDOWN), (51, RL), (32, DAMPING), (48, PASSIVE)):
            callback(key)
            self.assertEqual(ctl['request'], mode)
        self.assertEqual(_parse_fsm_script('0:stand,4:RL,8:sitdown'),
                         [[0, STAND], [4, RL], [8, SITDOWN]])
        with self.assertRaises(ValueError):
            _parse_fsm_script('4:STAND,3:RL')

    def test_local_angular_velocity(self):
        m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body quat=".70710678 .70710678 0 0"><freejoint/><geom size=".1"/></body></worldbody></mujoco>')
        d = mujoco.MjData(m)
        d.qvel[3:6] = [0, 0, 1]
        mujoco.mj_forward(m, d)
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, 1, vel, 1)
        np.testing.assert_allclose(vel[:3], d.qvel[3:6], atol=1e-7)

    def test_depth_projection_range_and_cache(self):
        # An asymmetric principal point detects either vertical flips or sign errors.
        m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <camera name="c" sensorsize="106 60" resolution="106 60"
                  focal="55.85 55.85" principal="-7 5"/>
          <geom type="box" pos="0 0 -1.1" size="5 5 .1"/>
          <geom type="sphere" pos=".2 .2 -.8" size=".05"/>
          <geom type="sphere" pos="-.2 -.2 -.8" size=".05" group="3"/>
          </worldbody></mujoco>''')
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        K = [[55.85, 0, 60], [0, 55.85, 25], [0, 0, 1]]
        cam = MujocoDepthCamera(m, d, 0, intrinsics=K, update_period=1/30,
                                ray_max_distance=3)
        try:
            depth = cam.render()
            self.assertAlmostEqual(float(depth[0, 0]), 1, places=5)
            v, u = np.where(depth < .95)
            self.assertAlmostEqual(float(u.mean()), 60 + 55.85*.2/.8 - .5, delta=1)
            self.assertAlmostEqual(float(v.mean()), 25 - 55.85*.2/.8 - .5, delta=1)
            d.time = .02
            self.assertIs(cam.render(), depth)
            d.time = .04
            self.assertIsNot(cam.render(), depth)
            cam.ray_max_distance = 1.1
            d.time = .08
            limited = cam.render()
            self.assertEqual(limited[0, 0], 3)
            self.assertAlmostEqual(float(limited[25, 60]), 1, places=5)
        finally:
            cam.close()

    def test_rollout_current_and_legacy_dimensions(self):
        torch.set_num_threads(1)
        for common in (225, 245):
            with self.subTest(common=common), tempfile.TemporaryDirectory() as td:
                ckpt = Path(td) / 'student.pt'
                net = DaggerNet(vec_size=common, output_size=12, depth_channels=1)
                md = dict(common_obs_size=common, action_size=12, depth_channels=1,
                          depth_history_length=5, depth_fps=30, depth_image_size=(60,106),
                          task='Locomotion-Go1-Rough-Vision-RayCaster', depth_delay_frames=1)
                torch.save(dict(model_state_dict=net.state_dict(), metadata=md), ckpt)
                command = [sys.executable, str(HERE/'play_mujoco_dagger_go1.py'),
                           '--ckpt', str(ckpt), '--policy_steps', '2', '--no_video']
                if common == 245:
                    command += ['--scene', 'stairs', '--stair_height_cm', '8']
                result = subprocess.run(command,
                                        capture_output=True, text=True, timeout=60,
                                        env=dict(os.environ, OMP_NUM_THREADS='1'))
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('[RESULT] policy_steps=2', result.stdout)
                if common == 245:
                    self.assertIn('rise=0.080 m', result.stdout)


if __name__ == '__main__':
    unittest.main()
