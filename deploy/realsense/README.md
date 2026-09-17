# RealSense D435 真机深度测试

独立于 IsaacLab 和学生策略的硬件测试。默认打开原始、未对齐 RGB 的
`848×480 @ 30 Hz` Z16 深度流，不启用软件滤波/补洞或策略预处理。
所有脚本支持 `--width`、`--height`、`--fps`、`--serial`、`--timeout_ms`。

## 安装与运行

在项目根目录、已激活的 Python 环境中安装：

```bash
python -m pip install -r deploy/realsense/requirements.txt
python deploy/realsense/test_depth_distance.py
python deploy/realsense/view_depth.py
python deploy/realsense/export_intrinsics.py
```

测距使用 Ctrl+C 退出；预览用 Q、Esc 或关闭窗口退出（需要桌面显示）。
中心像素为 0 表示没有有效深度，不表示目标就在相机上。
距离是深度 optical frame 的 Z 值，不是机器人 base 坐标系距离，也不是射线欧氏长度。
预览中的有效比例是非零像素比例，host FPS 是应用接收帧速率，不是硬件精度指标。

## 标定输出与训练对齐

内参导出默认保存到项目 `logs/realsense/`（已被 Git 忽略），包含设备序列号、
固件、实际原始流内参、畸变模型/系数和 Z16 到米的比例，以及缩放到 `106×60` 的 K。
可以指定输出位置和目标尺寸：

```bash
python deploy/realsense/export_intrinsics.py \
  --target_width 106 --target_height 60 \
  --output logs/realsense/d435_intrinsics.json
```

输出文件已存在时拒绝覆盖。默认使用带时间戳的新文件名。
缩放 K 使用完整图像直接缩放、连续坐标像素中心约定（index+0.5）；
若使用其他像素坐标约定、裁剪、矫正或 depth-to-color alignment，需相应调整标定。
畸变系数会保留，不能假定真实相机一定是零畸变。

`position_base` / `quaternion_wxyz` 故意留空：驱动不能测出深度光心相对于机器人
base 的安装外参，需要另行测量/标定。这里导出的 JSON 不会自动替换训练或
sim2sim checkpoint 的相机参数，也不是一个可直接导入的学生 checkpoint。

## 常见启动问题

优先使用 USB 3，关闭占用相机的 RealSense Viewer/其他程序。超时或权限错误时，
检查设备连接和 librealsense udev 规则；不要默认用 sudo 运行 Python。
若提示不支持流参数，可尝试 `--width 640 --height 480 --fps 30`，但必须重新导出
该分辨率的内参。预览需要带 GUI 的 OpenCV，不使用 opencv-python-headless。
安装系统驱动/udev 规则前请按当前操作系统查看 SDK 官方安装说明。

参考：[SDK Python 示例](https://github.com/realsenseai/librealsense/tree/master/wrappers/python/examples)、
[config API](https://realsenseai.github.io/librealsense/python_docs/_generated/pyrealsense2.config.html)。
