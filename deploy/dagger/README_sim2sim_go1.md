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

- **Tiled / RayCaster 通用**：从 checkpoint 读取分辨率、完整内参 K 和 ROS optical 外参，转换到 MuJoCo/OpenGL 相机坐标系。相机位置是深度光心，不是安装螺孔；旧 checkpoint 缺少标定时会告警并使用当前名义标定。
- `--viewer` 需要显示；headless 用 EGL（`MUJOCO_GL=egl`，默认）。
- `--viewer` 默认按真实时间限速（目标 1×），headless 不限速。计算/渲染太慢时仍会慢于实时，不修改物理步长或跳过控制步。
- 键盘：`Up/Down`=vx、`Left/Right`(+A/D)=vy、`Q/E`=wz（每次 ±0.1）、`V`=停止、`C`=巡航(0.4,0,0)、`ESC`=退出。
- 键盘范围读取学生 checkpoint 的 `command_a`，缺失时告警并使用 ±(0.8,0.4,0.8)。巡航及 viewer 初始 `--cmd` 也限制在该范围内；headless 的显式 `--cmd` 保持不裁剪。
- 默认录双窗视频到 `<ckpt目录>/videos/dagger_play/mj_sim2sim_go1_<checkpoint名>.mp4`；`--no_video` 跳过。

### 场景

| `--scene` | 说明 |
|---|---|
| `flat` | 平地 |
| `stairs` | 上 12 阶→平台→下 12 阶（`--step_rise` 默认 0.12m、`--stair_width` 默认 6m，可调）|
| `perlin` | 分形噪声 hfield（`--perlin_amp`）|
| `course` | 平地起步 → 楼梯上/下 → perlin 崎岖地形（`--cmd "0.5 0 0"` 纯前向）|

## 关键实现点

- **控制**：从 checkpoint 同目录的 `params/env.yaml` 读取物理步长、decimation、动作缩放、滤波开关和均一 PD/armature 参数。缺少文件会告警，名义值为 dt=0.004、decimation=5、kp=35、kd=0.5、armature=0.01。目标位置不额外 clamp；动作滤波使用上一帧 raw clipped action。可用 `--kp/--kd` 显式覆盖。
- **obs**：当前学生每帧 45 维 `[ang_vel_b, gravity_b, cmd, qpos-default, qvel, prev_action]`，5 帧共 225 维，无 base_lin_vel/clock。根据 checkpoint 维度也兼容旧 49×5=245 的 clock 模型。freejoint 的 `qvel[3:6]` 已在 base 坐标系，不再次旋转。
- **深度**：返回米制 optical Z-depth，图像原点在左上；保留机器人 visual mesh 的自遮挡，与训练 MultiMeshRayCaster 一致，RayCaster 模式按 3m 射线长度排除超范围命中。复用训练预处理、噪声、历史和控制步延迟。相机按 metadata 的采样周期缓存，历史每控制步更新，可能包含重复帧。
- **GRU**：每步 `net(seq, common, hidden=None)` 零状态，与训练一致。
- **指标**：`base_contact` 是 trunk 有几何接触的控制步数，不是跌倒率/成功率；另输出最终 trunk 位置。

## 注意事项

- **验证**：`python -m unittest discover -s deploy/dagger/tests -v` 检查角速度坐标系、增益、非居中主点投影、深度单位/方向、射线范围、缓存和 225/245 维端到端 rollout（需要 EGL）。
- **动力学与评估**：MuJoCo Menagerie 与 IsaacLab 的机器人模型、接触和随机化并非完全一致。sim2sim 测试迁移效果；判断训练是否收敛仍应在 IsaacLab 做固定地形/指令的纯学生评估。这里不会自动跌倒重置；建议先低速 `--cmd "0.3 0 0"`，不要用超出训练范围的键盘指令。
- **blur/延迟**：必须从 metadata 复现（训练默认开），否则输入分布偏移。
