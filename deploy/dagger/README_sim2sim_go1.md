# GO1 DAgger 学生策略 MuJoCo sim2sim

在 MuJoCo 里运行训练好的 GO1 DAgger 深度学生策略（Tiled 或 RayCaster），验证跨仿真器（Isaac Lab → MuJoCo）迁移。机器人模型是 mujoco_menagerie 的 `unitree_go1/go1.xml` 拷贝（`deploy/mujoco_models/go1/`），深度由 MuJoCo 离屏渲染器在训练 D435 位姿下实时渲染，预处理完全复现训练 `_sanitize_depth_data`（clip [0.1,2.0] + far 饱和 + **blur/延迟从 checkpoint metadata 读**）。

## 1. 建环境（一次性）

```bash
bash deploy/dagger/setup_sim2sim_env.sh        # 默认 env 名 sim2sim_go1
conda activate sim2sim_go1
```

## 2. 跑 sim2sim

```bash
python deploy/dagger/play_mujoco_dagger_go1.py \
  --ckpt logs/rsl_rl/rough_direct/<run>/dagger_policy.pt \
  --scene stairs --viewer
```

- **Tiled / RayCaster 通用**：两个学生相机配置相同（HFOV 87°、106×60 = D435 848×480/8、vFOV ~56.5°、位姿 (0.26,0,0.12)），脚本只读 checkpoint——换 `--ckpt` 即可。MuJoCo 相机 `fovy` 已按 106×60 设为 56.5°，渲染分辨率 `DEPTH_W/H` 同步为 106×60。
- `--viewer` 需要显示；headless 用 EGL（`MUJOCO_GL=egl`，默认）。
- 键盘：`Up/Down`=vx、`Left/Right`(+A/D)=vy、`Q/E`=wz（每次 ±0.1）、`V`=停止、`C`=巡航(0.4,0,0)、`ESC`=退出。
- 默认录双窗视频到 `<ckpt目录>/videos/dagger_play/mj_sim2sim_go1.mp4`；`--no_video` 跳过。

### 场景

| `--scene` | 说明 |
|---|---|
| `flat` | 平地 |
| `stairs` | 上 12 阶→平台→下 12 阶（`--step_rise` 默认 0.15m、`--stair_width` 默认 6m，可调）|
| `perlin` | 分形噪声 hfield（`--perlin_amp`）|
| `course` | 平地起步 → 楼梯上/下 → perlin 崎岖地形（`--cmd "0.5 0 0"` 纯前向）|

## 关键实现点

- **控制**：go1.xml 自带 `<position>` 执行器，脚本写**目标位置**到 `ctrl`；XML 里把 `kp=100` 覆盖为 `kp=30`（对齐训练 stiffness=30），关节 `damping=2/armature=0.01` 已与训练一致。位置执行器内部完成 PD。
- **obs**：49 维/帧（**无 base_lin_vel**）`[ang_vel, gravity, cmd, qpos-default, qvel, prev_action, clock]` × 5 历史 = 245，与训练学生 `common` 一致。
- **深度**：MuJoCo 离屏渲染 → `>3m / <0.01m → far` → clip[0.1,2.0] → **高斯 blur（σ 从 metadata 的 `depth_sensor_noise.blur_sigma`）** → 5 帧历史（含 `depth_delay_frames` 延迟）。
- **GRU**：每步 `net(seq, common, hidden=None)` 零状态，与训练一致。
- **指标**：`base_contact`（trunk 受外力次数）、最终 trunk 位置。

## 注意事项

- **深度方向**：未做 Isaac Lab↔MuJoCo 图像方向校准工具；相机用正交帧构造（image-up=world-up 投影）。若 `--viewer` 深度窗里上下颠倒，调 `CAM_PITCH_DEG` 或 depth 的 flip。
- **kp/动力学**：`kp=30` 若步态不稳，改 `_go1_xml_inner()` 里的 `kp="30"` 微调。
- **blur/延迟**：必须从 metadata 复现（训练默认开），否则输入分布偏移。
