# 给 TA 的无视觉托箱交付包

直接下载 [Team15_no_vision_booster_deploy_VALIDATED_20261030.zip](Team15_no_vision_booster_deploy_VALIDATED_20261030.zip)。在 GitHub 文件页面点击 **Download raw file**，不要把网页另存为 ZIP。解压后得到 `team15_proprio_package/`，保留完整结构。

这个 ZIP 是本地已经解压、安装并实际运行仿真验证过的同一个文件。SHA-256：

```text
6d759b186aa54bdeef5509061643bc25aeabb1e88be0858de425a4edc992d981
```

模型、配置、测试输入、完整评测和代码都在包里；不需要先合并其他 PR，不需要安装训练环境。这个分支基于 main，仅增加独立交付文件。当前范围是**仿真任务和真机只读诊断**，不是可直接接管机器人电机的任务。

## 现场先做什么

1. 给 TA 看 [英文交付说明](../deployment/team15_proprio_package/TA_FIRST_READ_EN.md)。
2. 按 [安装方法](../deployment/team15_proprio_package/README_ZH.md)准备固定版本的 `booster_deploy`、`booster_assets` 和独立仿真 Python 环境。不要覆盖实验室正在使用的仓库或 ROS 环境。
3. 按 [现场测试步骤](../deployment/team15_proprio_package/TA_ONSITE_ZH.md)复现 0.5 kg 仿真，再由 TA 在机器人上运行只读 ROS 诊断。

解压后，把三个目录并排放：

```text
test_workspace/
  team15_proprio_package/
  booster_deploy/
  booster_assets/
```

如果直接 clone 本分支，展开的交付包位于 `deployment/team15_proprio_package/`；可以把这个目录完整复制到上面的 `test_workspace/`。它与 ZIP 中的内容逐字节相同。

## 模型与验证范围

- 72 维输入：IMU、关节位置/速度、上一动作。无需相机、箱体状态、重量输入或滑动检测。
- 22 维动作，50 Hz。模型不是默认 walking ONNX 的直接替代品。
- 原策略通过修正箱子质心后的部署仿真：0.25、0.5、1.0 kg 各 6 次 20 秒；0.5 kg 另 3 次 60 秒，共 **21/21**。这是评测初始条件变化，不是多个独立训练 seed。
- 本次修复将箱子局部质心设为几何中心，未重训，也未放宽判定标准。
- 演示优先用 **0.5 kg**。1 kg 最大箱体倾角 9.624°，接近 10° 阈值。
- 箱子尺寸 20 × 40 × 12 cm，质心居中；前臂接触摩擦 0.7、脚底接触摩擦 1.0，尚未按实物材料标定。
- 测试从双臂已抬起、箱子已放好的状态开始。空手运行、抬臂、搬放箱子和真机控制尚未验证。

[20 秒原速演示](../deployment/team15_proprio_package/demo/demo_0p5kg.mp4) · [完整验证说明](../deployment/team15_proprio_package/VALIDATION_STATUS_ZH.md) · [ZIP 解压复测记录](VALIDATED_20261030.verification.json)

## 文件完整性与历史记录

展开目录中的 `SHA256SUMS.json` 覆盖其余 73 个文件。仓库保留原始字节，避免换行转换破坏校验值。`PROVENANCE.json` 记录的是打包时的来源状态，其中 `unpublished_local_changes` 表示当时尚未发布，不是当前 GitHub 发布状态。

包内也保留了旧错误质心场景的失败记录，已经注明被修正场景复测取代；不要把它们与当前通过的 21 次评测混用。

真机操作前，TA 仍需完成关节/IMU 标定、控制增益与保护、启动姿态和放箱/退出流程，以及实际发送电机指令的控制路径。`shadow_check.py` 只读数据，不会接管机器人。
