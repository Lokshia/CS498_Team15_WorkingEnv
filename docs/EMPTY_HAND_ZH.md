# 空手稳站：独立任务

这个任务只保留机器人和地面，**没有箱子实体、箱子碰撞、1 g 质量代理或虚拟箱子观测**。机器人从原来的前臂抬起姿态开始，学习在双脚接触地面时保持较平、较直、较平顺的站姿。

它与旧托箱命令的 `--mass 0` 不同：旧任务仍有一个 0.001 kg 的接触箱体，模型也仍需要箱子状态。空手任务独立训练，不加载旧托箱权重。

## 输入与动作

| 项目 | 定义 |
|---|---|
| Actor 输入 | 72 维：机身角速度 3、重力方向 3、关节相对默认位置 22、关节速度 22、上一动作 22 |
| Critic 输入 | 上述 72 维，加仿真中的机身线速度 3 和离地高度 1，共 76 维 |
| 动作 | 原 K1 的 22 个关节位置动作，保留原动作缩放和默认姿态偏移 |
| 控制频率 | 50 Hz；物理步长 0.005 s，每个动作执行 4 个物理步 |

奖励关注机身高度和倾角、双脚接触和脚底倾角、不过度偏向单脚的承重、默认抬臂姿态，以及动作、关节速度和力矩。成功与否由独立评测决定，不能只看训练 reward。

## 运行方法

先按 [项目 README](../README.md#1-安装只做一次) 安装环境。在项目根目录运行以下 Windows PowerShell 命令。Linux 将解释器换成 `./.venv/bin/python`。

每次使用新的 `--output` 目录名，程序不会覆盖旧实验。CUDA 训练和评测需要已安装 GPU 环境；单机器人 viewer 可使用 CPU。

**从头训练一个空手模型：**

```powershell
.\.venv\Scripts\python.exe -m lifting_lab.empty_hand_cli train --device cuda --num-envs 256 --seed 11 --iterations 300 --output outputs/empty_hand_my_train
```

这是训练命令示例，300 次更新不代表已经学会稳站。完成后，模型与训练配置位于该输出目录的 `policy.pt` 和 `experiment.json`。

**独立评测自己训练的模型：**

```powershell
.\.venv\Scripts\python.exe -m lifting_lab.empty_hand_cli evaluate --device cuda --num-envs 16 --seed 5011 --seconds 20 --checkpoint outputs/empty_hand_my_train/policy.pt --output outputs/empty_hand_my_eval
```

查看 `evaluation.json` 的成功数、失败时间、双脚倾角和受力、躯干倾角、关节速度及力矩；原始轨迹保存在 `trace.npz`。失败后重新开始的 episode 不会替换原失败记录。换评测 seed 需要另建输出目录。

**打开已训练的空手 viewer：**

```powershell
.\.venv\Scripts\python.exe -m lifting_lab.empty_hand_cli demo --device cpu --num-envs 1 --seed 2027 --seconds 3600 --checkpoint shared_model/empty-hand-v1-20261030/policy.pt --port 8081 --paused --output outputs/empty_hand_my_viewer
```

共享模型与配置已整理在上述目录中。浏览器打开 `http://127.0.0.1:8081/`，点击 **Play**。保留终端运行。空手 viewer 没有箱子重量控件；演示中允许倒下后继续显示，不计作通过评测。

## 给队友的文件

`policy.pt` 必须与同一次训练的 `experiment.json` 放在同一目录。不要混用配置，也不要只复制权重文件。共享时同时提供评测 JSON，并注明使用的代码版本。

旧托箱模型使用 97 维输入，不能通过这个空手入口加载。新入口会检查任务、观测维度、动作和控制频率等 schema，不会自动把旧模型裁成空手模型。

已训练模型通过普通评测 32/32（每次 20 秒）、小幅速度扰动评测 16/16（每次 22 秒），以及 CPU 回放 1/1。只有一个训练 seed，不能推断广泛鲁棒性。训练预算、原始证据和适用范围见 [模型说明](../shared_model/empty-hand-v1-20261030/README.md)。

## 当前边界

- 从前臂已经抬起的姿态开始；没有从双臂自然下垂进入该姿态的控制流程。
- 不支持运行中放上箱子、卸下箱子或在两种策略间切换。
- 这是模拟任务，没有真机 adapter。72 维本体观测只是减少了对箱子状态的依赖；仍需验证真实传感器、关节顺序与方向、动作缩放、控制频率及执行器接口，才能讨论上机。
