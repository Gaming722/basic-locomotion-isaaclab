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

### 部署状态机仿真

在连接真机以前，用同一套状态机语义验证 PASSIVE、DAMPING、STAND、
SITDOWN 和 RL。快捷入口默认使用当前 `dagger_raycaster_v3_ft2.pt`：

`go1_deploy_core.py` 是 MuJoCo 与 NX 真机部署逐字节相同的唯一控制核心，统一负责
关节顺序、225 维观测、深度历史/延迟、raw-action low-pass、状态切换和 PD 命令。
两端只分别实现 MuJoCo 与 LowState/D435/LowCmd I/O。启动时会用
`deployment_manifest.json` 校验核心和 `fsm_sim_config.json` 的 SHA-256；任一端文件
漂移都会拒绝运行，并在日志打印当前核心哈希。

```bash
bash deploy/dagger/play_mujoco_fsm_go1.sh
```

第一个参数选择地形，第二个参数选择楼梯高度（厘米）：

```bash
bash deploy/dagger/play_mujoco_fsm_go1.sh stairs 8
bash deploy/dagger/play_mujoco_fsm_go1.sh stairs 10
bash deploy/dagger/play_mujoco_fsm_go1.sh stairs 12
bash deploy/dagger/play_mujoco_fsm_go1.sh stairs 15
bash deploy/dagger/play_mujoco_fsm_go1.sh stairs 18
```

支持的快捷预设为 6/8/10/12/15/18 cm；也可直接运行 Python 并用
`--step_rise 0.13` 指定任意正数高度。自定义 checkpoint 可通过
`GO1_STUDENT_CKPT=/path/student.pt` 传入快捷脚本。

状态键为 `0=PASSIVE`、`Space/Esc=DAMPING`、`1=STAND`、`2=SITDOWN`、
`3=RL`、`X=退出`。RL 只能从已到位的 STAND 进入。
速度键严格沿用原有 sim2sim：`Up/Down=vx`、`Left/Right=vy`、`Q/E=wz`
（每次 0.1），`V=速度指令全零`、`C=0.4 m/s 前进`、`B=深度失能/恢复`。
`N` 可实时开启/关闭本体观测高斯噪声。
不使用 MuJoCo 的 `W=wireframe` 和 `R=reload`；左右方向键的 viewer 历史步进
在 passive 模式下不驱动物理。仿真从 `fsm_sim_config.json` 的趴地姿态开始；进入
STAND 时保存当次实测姿态作为 SITDOWN 终点。

该模式将 MuJoCo 执行器改为力矩接口，并逐物理步计算
`tau=Kp*(q_des-q)-Kd*dq`，从而让 PASSIVE、DAMPING 和各状态独立增益的
含义与真机 LowCmd 一致。当前候选参数为姿态控制 `50/5`、RL `35/0.5`、
DAMPING `Kd=1`；它们首先只在仿真中验证，不会自动同步到真机。

无窗口回归序列：

```bash
conda run -n sim2sim python deploy/dagger/play_mujoco_dagger_go1.py \
  --ckpt logs/rsl_rl/rough_direct/<run>/student.pt --scene flat \
  --fsm --fsm_scripted --policy_steps 550 --no_video
```

可用 `--fsm_script '0:STAND,4:RL,25:SITDOWN'` 自定义无窗口状态时间线。
结束时的 `FSM_METRICS` 会按状态汇总机身触地、跌倒判据、最低机身高度、
最大姿态角、最大关节误差/速度及其发生时间和关节，并统计超过 Go1 URDF
关节速度限值的控制步/样本数。当前固定延迟测试结果见
[`FSM_SIM2SIM_REPORT.md`](FSM_SIM2SIM_REPORT.md)。录制较长回归时可加
`--video_stride 5` 只降低视频到 10 FPS，不改变物理或策略频率。

- **Tiled / RayCaster 通用**：从 checkpoint 读取分辨率、完整内参 K 和 ROS optical 外参，转换到 MuJoCo/OpenGL 相机坐标系。相机位置是深度光心，不是安装螺孔；旧 checkpoint 缺少标定时会告警并使用当前名义标定。
- `--viewer` 需要显示；headless 用 EGL（`MUJOCO_GL=egl`，默认）。
- `--viewer` 默认按真实时间限速（目标 1×），headless 不限速。计算/渲染太慢时仍会慢于实时，不修改物理步长或跳过控制步。
- 键盘：`Up/Down`=vx、`Left/Right`=vy、`Q/E`=wz（每次 ±0.1）、`V`=速度指令全零、`C`=巡航(0.4,0,0)、`B`=深度全零/恢复；普通模式 `ESC`=退出，FSM 模式 `X`=退出。`B` 开启时 actor 立即收到全零深度，后续历史帧也写零；恢复后历史逐步填满。预览按原有近白远黑映射呈白色。这是超出正常预处理分布的硬失能测试，不代表真实设备失效时的全部行为。
- **本体噪声 A/B**：加 `--proprio_noise --no_depth_noise` 后，以训练默认标准差对
  `ang_vel/gravity/q/dq` 分别加入每控制步零均值高斯噪声
  `0.02/0.002/0.001/0.02`，其余 command 和 previous-action 保持不变；`N` 可在
  viewer 中实时切换。`--no_depth_noise` 只关闭随机 additive/dropout，仍保留 checkpoint
  中的深度 clip、blur、采样率与延迟，适合隔离本体噪声。可用四个
  `--proprio_*_std` 参数覆盖幅度，并用 `--proprio_noise_seed` 固定随机序列。
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
