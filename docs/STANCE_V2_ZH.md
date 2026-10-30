# stance-v2：改善躯干倾斜和左右腿姿态

本实验在已保留的 [stance-v1 模型](../shared_model/stance-v1-20261029/README.md) 上继续训练。原 upright300 训练因 CUDA 错误中断，降低并行数后的恢复运行已完成。**选择恢复末模型 1950 用作姿态改善的演示候选；它仍有轻箱启动失败，不能称为更可靠的全面替代。** 模型和评测整理在 [stance-v2 模型目录](../shared_model/stance-v2-20261029/README.md)，额外 seed 6011 复核已完成；两组评测合计 94/96 次托箱及严格平脚成功。v1 文件和结果保留，作为目前实测更可靠的备选。

## 为什么继续调整

v1 选用模型 `model_1450.pt` 的脚底已经较平，但匹配 CPU 回放的稳态躯干平均倾角为 **4.136°**，比旧 smooth-v1 的 **2.646°** 大。视觉上仍可见左右腿不对称。脚底平放的成功判据并没有额外要求躯干竖直或左右腿对称，所以不能把 v1 的 96/96 平脚及托箱结果解释成完整的站姿质量验证。

v2 的目标是：**在维持双脚平放、平顺托箱和启动可靠性的同时，减少躯干倾斜与不必要的左右腿差异。** 不要求关节锁死、膝盖伸直或两脚严格各承担一半重量。

## 新增奖励的精确定义

`stance-v2` 原样继承 v1 所有奖励，仅增加以下三项。原 `smooth-v1` 和 `stance-v1` 的定义不变；物理、摩擦、接触、97 维观测、22 个动作、动作缩放、控制频率及原终止条件不变。

| 新成本项 | 定义 | 配置权重 |
|---|---|---:|
| `trunk_tilt` | `(1 − clamp(upright, −1, 1)) / (1 − cos(10°))`。`upright` 是躯干竖直分量；倾斜 10° 时成本为 1，不惩罚偏航 | −0.75 |
| `leg_symmetry` | 六对腿关节的镜像残差 `r = q_left − sign × q_right`；成本为 `mean[(relu(abs(r) − 5°) / 10°)²]`，计算时角度转为弧度 | −0.25 |
| `failure` | 非时间上限的终止标志；包括跌倒、失去承托或禁止接触等既有终止原因 | −100 |

reward manager 会乘控制步长 `0.02 s`。因此 `failure` 的一次实际奖励为 **−2**，不是 −100；正常到达 episode 时长上限不产生这一失败罚分。

已有 PPO 本身会在失败终止时截断后续价值，不能说以前“完全没有失败惩罚”。新增显式成本只是待验证的训练改动，不是已经证实的启动问题根因或解决方案。v2 同时增加三项，也不能凭一个训练结果区分各项的独立贡献。

## 腿关节镜像符号的物理核对

当前使用的是串联踝关节 K1 模型。六对关节按以下顺序匹配：

| 关节 | `sign` | 对称时的关系 |
|---|---:|---|
| Hip Pitch | +1 | 左右同号 |
| Hip Roll | −1 | 左右反号 |
| Hip Yaw | −1 | 左右反号 |
| Knee Pitch | +1 | 左右同号，保留弯膝 |
| Ankle Pitch | +1 | 左右同号 |
| Ankle Roll | −1 | 左右反号 |

这组符号不仅依据名称判断，还做了两项核对：

1. 与固定依赖 `booster_mjlab/tasks/velocity/mdp/observations.py` 的串联踝 K1 左右镜像索引一致。其并联踝版本有不同的曲柄符号，不能混用。
2. 用实际 K1 XML 编译原生 MuJoCo 模型，检查默认托箱腿姿和关节轴；逐对施加正负 `0.02 rad` 的镜像扰动后，左右脚底位置和法向符合关于矢状面的镜像关系。本次数值检查的残差为零，默认腿关节镜像残差也为零。

左右腿分别通过精确名称、固定顺序查找。5° 的软容许区间允许姿态为负载作小幅调整，手臂不受这项镜像成本约束。以上核对只适用于当前仿真模型，不能替代真机 SDK 关节顺序、方向和零位验证。

## upright300 训练配置

从 v1 选用 checkpoint 继续训练，原 SHA-256 为 `b0093a17767ba382549e392dc0c2f5844dea66b37173a6b6f1ba08c9c98a18f4`。训练 seed 仍为 **11**；模型、归一化和 optimizer 延续，环境／随机状态重启，不是独立种子的训练。

```powershell
.\.venv\Scripts\python.exe -m lifting_lab train --device cuda --num-envs 512 --seed 11 --iterations 300 --resume shared_model/stance-v1-20261029/policy.pt --reward-profile stance-v2 --mass-range-fraction 0.0125 0.05 --learning-rate 0.00015 --entropy-coef 0.0005 --output outputs/stance_study_20261029/upright300_reproduce
```

在仓库根目录运行；输出必须是新目录。Linux 将解释器换为 `./.venv/bin/python`。每次 reset 均匀采样箱子质量约 **0.247461–0.989845 kg**，对应机器人质量的 1.25–5%；质量与均匀箱体惯量同步变化。名义质量 0.618653 kg 只是区间中点。摩擦、重心偏移、箱体相对位姿和推力设置未随机化。

原始运行目录为 `outputs/stance_study_20261029/upright300/`。元数据记录：512 个环境、每次 rollout 24 步，从 iteration **1451** 开始，计划 300 次更新，增加 **3,686,400 transitions**，计划累计 **21,516,288 transitions**。完成情况及候选 checkpoint 的实际预算以后续 `completed.json`、checkpoint 内 iteration 和选型记录为准，不以计划量代替。

运行源码指纹：`3db870c3d1fd52617a35ec6143793a425b6bb94c430ea078c37d80f4477c8990`。本轮硬件为 RTX 5090 Laptop GPU；跨设备不保证逐位复现。

## 中断记录与实际保存预算

原 `upright300` 在日志 iteration 1589 附近发生 `CUDA error 719: unspecified launch failure`，没有生成 `completed.json`。同一时段（2026-10-29 约 20:22:18–20:22:20，America/Chicago）Windows System 日志记录了 `nvlddmkm` 13／153 事件。这里记录的是同时观察到的 GPU／驱动错误，不据此断言特定硬件故障或奖励代码是根因。

**该运行不能记作完成 300 次更新。** 最后可读取的已保存 checkpoint 是 `upright300/model_1550.pt`，已在 CPU 上核对其内容：

| 项目 | 实际保存值 |
|---|---:|
| 最后完成的 iteration（从零编号） | 1550 |
| v2 本阶段已保存更新数 | 100 |
| 本阶段保存的 transitions | 1,228,800 |
| 当前 checkpoint 累计 transitions | 19,058,688 |

checkpoint SHA-256 为 `40bd97a5c57b8ac4fb80f87b3986b05e419f5bd7911992c783a30d79c398d7f7`。最后一次保存之后曾继续消耗计算，但这些未保存更新不属于恢复 checkpoint 的训练经历，不能加进该模型的实际预算。

完成小批量 GPU 检查后，实际从 `model_1550.pt` 以 **256 个环境、400 次更新**继续；rollout 仍为 24，增加 **2,457,600 transitions**，累计保存预算达到 **21,516,288 transitions**。恢复运行已生成 `completed.json`，最后完成 iteration **1950**。这仅匹配原计划的环境交互步数；批大小和优化器更新次数改变，不能视为逐位恢复或完全相同的训练过程。

恢复运行目录为 `outputs/stance_study_20261029/upright_recovery400/`，末模型 SHA-256 为 `b0177c11ad67e6351b10b84ace80d79bac48face440eb3dc93946d6bd1802ada`。复现其配置：

```powershell
.\.venv\Scripts\python.exe -m lifting_lab train --device cuda --num-envs 256 --seed 11 --iterations 400 --resume outputs/stance_study_20261029/upright300/model_1550.pt --reward-profile stance-v2 --mass-range-fraction 0.0125 0.05 --learning-rate 0.00015 --entropy-coef 0.0005 --output outputs/stance_study_20261029/upright_recovery400_reproduce
```

源代码指纹保持前述 `3db870...c8990`。训练完成仅证明完成了循环和数值检查，不证明控制策略通过任务评测。

## 评测与选择原则

先用与 v1 相同的三种负载、16 个环境、20 秒、seed 5011 做匹配评测；再检查另一个评测 seed，并单独检查约 0.495 kg、seed 2027 的启动与稳态姿态。原脚平放／托箱判据保持不变，不能为了让 v2 通过而放宽阈值。

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_stance.py --checkpoint outputs/stance_study_20261029/upright_recovery400/policy.pt --device cuda --num-envs 16 --seed 5011 --seconds 20 --mass 0.247461246173745 --mass 0.494922492347489 --mass 0.989844984694979 --output outputs/stance_study_20261029/upright_recovery400_eval_reproduce
.\.venv\Scripts\python.exe scripts/diagnose_stance.py --checkpoint outputs/stance_study_20261029/upright_recovery400/policy.pt --device cpu --mass 0.4949224923474894 --seed 2027 --seconds 20 --output outputs/stance_study_20261029/upright_recovery400_cpu_reproduce
```

上述命令针对恢复运行末模型，即此次选用的姿态演示候选，也可将路径替换为 `shared_model/stance-v2-20261029/policy.pt`。失败的 `upright300` 本身没有最终 `policy.pt`。对于中间 checkpoint，必须按 checkpoint 内 iteration 及阶段记录计算预算。

评测时分别报告：

- 原托箱成功数与严格平脚成功数；任何失败的时间、接触原因，启动失败不能被后续稳态视频掩盖。
- 左右脚倾角和垂直受力、躯干倾角、关节速度与力矩；启动 `t < 2 s` 和稳态 `t >= 2 s` 分开统计。
- 左右腿按上述真实符号计算的镜像残差，而不是只凭正面截图判断对称。
- 与 v1 的取舍。只有总 reward 增加或躯干更直，但托箱失败增加，不足以认定为全面替代；若用于姿态演示，须明确失败并保留原模型。

## 恢复末模型的结果与取舍

CPU、约 0.494922 kg、seed 2027、20 秒回放的稳态 `t >= 2 s`：

| 指标 | 旧 smooth-v1 | 已保留的 stance-v1 | v2 恢复末模型（1950） |
|---|---:|---:|---:|
| 左／右脚平均倾角 | 0.025°／10.215° | 0.019°／0.391° | 0.062°／1.220° |
| 左／右脚平均垂直支持力 | 121.61／77.32 N | 113.10／85.93 N | 111.69／87.38 N |
| 躯干平均倾角 | 2.646° | 4.136° | 1.549° |
| 腿镜像残差 RMS | 9.809° | 10.934° | 7.256° |
| 全部关节速度 RMS | 0.01982 rad/s | 0.01699 rad/s | 0.00998 rad/s |
| 全部关节仿真力矩 RMS | 2.096 N·m | 1.815 N·m | 2.237 N·m |

腿镜像残差使用上文六对关节的实际符号，计算所有稳态样本和六对关节的**原始角度残差 RMS**，没有扣除奖励中的 5° 容许区间。该数值不是每个关节的最大误差，也不表示完全对称。

该模型的躯干更直、腿姿更接近镜像、关节运动较慢，但力矩 RMS 相比旧 smooth-v1 **增加约 6.76%**，相比所保留的 stance-v1 也增加。脚倾角仍较小，但高于 v1 的单次诊断；不能把这些变化统称为所有指标改善。原始数据位于 `upright_recovery400_cpu/stance_summary.json` 与 `stance_trace.npz`。

GPU seed 5011 的三负载评测已完成，来源为 `upright_recovery400_eval/stance_evaluation.json`，评测期间源码未变化：

| 箱子质量 | 托箱成功 | 严格平脚成功 |
|---|---:|---:|
| 0.247461 kg | 15/16 | 15/16 |
| 0.494922 kg | 16/16 | 16/16 |
| 0.989845 kg | 16/16 | 16/16 |

轻箱 slot 7 在 **0.26 秒**因连续 **7 步**失去承托而终止。总计 **47/48**，未达到 v1 在同组条件下的 48/48；虽然单次姿态更好，也不能忽略启动失败。

**当前模型选择：1950 末模型用于姿态改善的演示；v1 保留为实测启动更可靠的备选。** 中间 checkpoint 1800 的轻箱同组结果为 13/16，比末模型的 15/16 更差，因此未取代它；其余中间模型结果以实际评测文件为准。停止继续搜索。选择后的 seed 6011 复核已完成：轻箱 15/16、中箱 16/16、重箱 16/16；轻箱 slot 4 同样在 0.26 秒、连续 7 步失去承托后终止。两组共 94/96，分质量为 30/32、32/32、32/32；失败计入全部分母。原始复核文件见模型目录的 `evaluation_seed6011.json`。模型选择不是宣称已解决所有启动失败。

不同阶段训练预算不等，且沿用同一训练 seed；这不是独立多种子或等预算消融实验。因为 seed 5011 已用于多次候选比较，它属于选型数据，不是未见过的最终测试集。v1 模型及其原始记录未被覆盖。

## 真机边界

97 维观测仍依赖仿真提供的箱子状态等信息；真机传感、估计器、控制映射和进入托箱姿态的过程尚未接好。奖励改动和仿真姿态改善都不能直接成为上机依据。参见 [SIM_TO_REAL_ZH.md](SIM_TO_REAL_ZH.md)。
