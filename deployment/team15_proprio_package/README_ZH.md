# Team15：无视觉托箱 · booster_deploy 对接包

**修正箱子质心后，官方部署场景验证通过：0.25 / 0.5 / 1.0 kg各6次20秒；0.5 kg另3次60秒。** 模型、奖励、摩擦系数和评分阈值未改，本次没有重训。默认演示用 **0.5 kg**。详见[实际验证结果](VALIDATION_STATUS_ZH.md)，或先看 `demo/demo_0p5kg.mp4`。

本包可在固定版本 `booster_deploy` 的工作副本运行带箱仿真。真实机器人目前提供**只读传感器/推理检查**，尚无可接管电机的硬件任务。现场操作见[TA步骤](TA_ONSITE_ZH.md)。

## 已准备好依赖和固定版本副本时，直接运行

在并排放有 `team15_proprio_package/`、`booster_deploy/`、`booster_assets/` 的目录执行：

```bash
python team15_proprio_package/scripts/install_task.py --deploy-root booster_deploy
python team15_proprio_package/scripts/validate_package.py --deploy-root booster_deploy --assets-root booster_assets --output sim_check_050kg --episodes 6 --mass-kg 0.5 --seconds 20 --seed 21031
cd booster_deploy
python -m tasks.team15_proprio.simulate --mass-kg 0.5 --paused
```

验证应返回码0、`report.json` 中 `all_passed: true`；窗口按 **Space** 暂停/继续。安装脚本不覆盖冲突文件；若装过旧版，用新的固定版本副本。尚未准备环境时，按下方安装说明操作。

控制网络只读72维IMU、关节状态和上一动作。无需相机、箱体位置、滑动检测或输入重量。箱体位置/接触只用于模拟器初始化和测试评分。不要混用旧97维 `forearm_hold.onnx`，也不要直接替换默认walking模型。

## 给TA哪些文件

发整个ZIP，保留解压后的 `team15_proprio_package/` 结构。主要文件：

| 路径 | 用途 |
|---|---|
| `tasks/team15_proprio/` | 独立任务、22关节适配、自由箱体场景和控制器扩展 |
| `tasks/team15_proprio/models/proprio_hold.onnx` | 只含actor的模型，输入 `[1,72]`，输出 `[1,22]` |
| `models/proprio_hold.contract.json`（在上面任务目录内） | 输入顺序、单位、关节映射、默认角、缩放、哈希 |
| `models/proprio_hold.test_vectors.npz` | 模型输入和预期输出，检查推理是否一致的“test文件” |
| `scripts/validate_package.py` | 真正运行独立部署仿真，保存成功/失败及轨迹 |
| `scripts/shadow_check.py` | 机器人上只读传感器并推理，绝不发送动作 |
| `TA_ONSITE_ZH.md` | 现场命令、日志解释、停止点和待核对硬件信息 |
| `demo/demo_0p5kg.mp4` | 修正场景后的0.5 kg仿真演示 |
| `reports/deploy_0p25kg/`、`reports/deploy_0p5kg/`、`reports/deploy_1kg/` | 各6次20秒部署仿真报告和轨迹 |
| `reports/deploy_0p5kg_60s/` | 0.5 kg额外3次60秒报告和轨迹 |
| `SHA256SUMS.json` / `PROVENANCE.json` | 文件校验、来源、版本和验证范围 |

不包含训练环境、GPU缓存、课程全部工程、ROS安装或机器人资产。推理和本包仿真使用CPU即可。

## 安装到独立测试副本

下面三个目录并排放，不要覆盖实验室已有工作目录：

```text
test_workspace/
  team15_proprio_package/
  booster_deploy/
  booster_assets/
```

在 `test_workspace/` 执行：

```bash
git clone https://github.com/BoosterRobotics/booster_deploy.git
git -C booster_deploy checkout 7bb1462e2ac742eba5c01659b815e2e99fbe617f
git clone https://github.com/BoosterRobotics/booster_assets.git
git -C booster_assets checkout 12a97516f57469078707afb587a5f84c1cb110d1
```

用**独立的仿真Python环境**，此次本地验证使用Windows、Python 3.13.14。Linux/现场ROS环境仍需复测；不要为了这个仿真锁文件升级TA的ROS Python。

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r team15_proprio_package/requirements-sim.txt
python -m pip install -e booster_assets
python team15_proprio_package/scripts/install_task.py --verify-only
python team15_proprio_package/scripts/install_task.py --deploy-root booster_deploy
python team15_proprio_package/scripts/validate_package.py --deploy-root booster_deploy --assets-root booster_assets --output sim_check_050kg --episodes 6 --mass-kg 0.5 --seconds 20 --seed 21031
```

脚本拒绝版本不符、上游控制器改动、文件哈希不符和不同内容的目标文件冲突。输出目录必须新建；不会覆盖过去的测试。

仅检查模型推理一致性可给最后一条命令加 `--export-only`；这不测试站立。

看仿真：

```bash
cd booster_deploy
python scripts/deploy.py --list
python -m tasks.team15_proprio.simulate --mass-kg 0.5 --paused
```

窗口Space暂停/继续。**必须使用专用入口**：官方原入口假定只有机器人，没有额外自由箱体，因此会被本任务拒绝。此扩展显式索引22个关节和电机；没有修改官方仓库原文件。更换质量用 `--mass-kg 0.25` 并重新启动仿真。

## 模型和控制接口

50 Hz，单帧72维：机身角速度3、机身坐标下重力方向3、关节相对默认角22、关节速度22、上一动作22。没有walking的历史堆叠、额外速度缩放、tanh或随机采样。

ONNX/TorchScript内已包含训练归一化和[-1,1]裁剪。适配器计算 `q_default + scale * action`，再把目标投影到官方XML关节范围；下一次输入使用发送目标对应的有效动作。这个后处理与训练环境存在差别，必须以本包部署仿真结果评价。

只读ROS诊断为每帧零上一动作，属于冷启动格式/推理检查，**不是**执行上述闭环。它不导入Portal、不创建电机publisher、不调用模式RPC。

## 箱子与适用范围

箱子全尺寸：前后20 cm × 左右40 cm × 高12 cm；均匀刚体、质心在几何中心。靠双前臂接触承托，无焊接、绑缚或固定基座。箱子—前臂滑动摩擦系数设为0.7，脚底—地面为1.0；尚未用现场材料实测标定。

当前部署场景测试点为0.25、0.5和1.0 kg。1.0 kg的全程最大箱体倾角为9.624°，接近10°阈值，因此演示优先用0.5 kg。这些有限测试不代表任意重量、摩擦或拿箱姿势都能成功。箱子一开始已放好，不包含拿箱、加载/卸载、从垂臂进入托箱的策略；也没有把0 kg箱子当作空手测试。

旧部署失败由箱子局部质心设在箱体外造成，已修正为几何中心。不要把旧错误场景的失败继续当作当前策略结论。原训练框架中的1 kg失败来自另一份场景，和这里的部署复测应分开报告。

上机所缺的参数和步骤见 `TA_ONSITE_ZH.md`。**不要解除仿真拦截、把只读日志目标接到电机，或把仿真Kp/Kd直接发给真实电机。**
