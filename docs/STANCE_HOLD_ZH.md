# 双脚平放托箱：stance-v1 实验记录

这是 2026-10-29 的姿态改进实验。`pilot300` 和质量随机化续训 `mixed400` 已完成；**mixed400 末模型因启动失败增加而未选用**。最终选用其中的 `model_1450.pt`（混合质量阶段完成 151 次更新），通过了 seed 5011 的匹配评测和选择后 seed 6011 的复核，共 96/96 次托箱及平脚成功。本页各阶段数据不能互相替代。任何候选都不是可直接部署到真机的控制器。

## 为什么修改

旧 `smooth-v1` 能维持托箱，但没有要求脚底平放。脚边缘接触地面也可能满足原接触判据；旧 `balance_approx` 仅使用两脚中心的包围矩形，不是真实接触支撑区域。

在约 **0.494922 kg、seed 2027、CPU、20 秒** 的匹配回放中，取 `t >= 2 s` 的稳态窗口：

| 指标 | smooth-v1 | pilot300 | 选用的 model_1450 |
|---|---:|---:|---:|
| 左／右脚平均倾角 | 0.025°／10.215° | 0.030°／0.173° | 0.019°／0.391° |
| 左／右脚平均垂直支持力 | 121.61／77.32 N | 115.68／83.44 N | 113.10／85.93 N |
| 双脚同时接触且各自倾角 <5° 的采样比例 | 0% | 100% | 100% |
| 躯干平均倾角 | 2.646° | 4.065° | 4.136° |
| 全部关节速度 RMS | 0.01982 rad/s | 0.01417 rad/s | 0.01699 rad/s |
| 全部关节仿真力矩 RMS | 2.096 N·m | 1.930 N·m | 1.815 N·m |

脚底平放明显改善，**但躯干倾角增加，不能说所有姿态指标都改善**。左右承重不必严格各占一半。以上为单次回放的诊断均值，不是成功率；启动前两秒另行记录，不能用稳态均值掩盖启动瞬态。

原始文件：

- `outputs/stance_study_20261029/baseline_cpu/stance_summary.json`
- `outputs/stance_study_20261029/pilot300_cpu/stance_summary.json`
- `outputs/stance_study_20261029/mixed151_cpu/stance_summary.json`
- 各目录的 `stance_trace.npz` 保存逐步数据。

对应摘要和选用模型的 CPU trace 已整理到 [shared_model/stance-v1-20261029](../shared_model/stance-v1-20261029/README.md)，无需将整个 outputs 目录发给队友。

## 奖励改动

`stance-v1` 继承 `smooth-v1` 的托箱、平顺动作、接触、滑移、速度、力矩及关节限位奖励。新增或调整如下。权重指配置中的乘数；reward manager 还会乘控制步长 `0.02 s`。

| 项目 | 计算 | 权重 |
|---|---|---:|
| 脚底平放成本 `feet_flat` | 两只脚分别取脚底本地 +Z 轴变换到世界后的竖直分量 `n_z`，成本为 `Σ(1 − clamp(n_z, −1, 1))` | −20 |
| 承重比例成本 `foot_load_share` | 正的垂直支持力按接触标志过滤，求两脚份额 `s_i`；成本为 `Σ[relu(0.30 − s_i)/0.30]²`。总支持力不超过机器人加箱子重量的 10% 时，成本为 1 | −0.5 |
| 腿姿成本 `leg_posture` | 仅 12 个腿关节：`mean[((q − q_default)/0.25 rad)²]`；默认姿态保留弯膝，手臂不受此项约束 | −0.5 |
| 躯干直立 `upright` | 保留原 `clamp(upright, 0, 1)²` 计算 | 2 → 4 |
| 旧平衡矩形 `balance_approx` | 仅在新 profile 中删除该奖励项 | 删除 |

承重比例有 **30–70% 的零成本区间**，没有要求精确 50/50。脚平放成本在腾空时也存在，原有缺脚接触成本仍保留。脚底法向接近竖直加上接触标志，仍不等于测得完整接触面积或压力中心（COP）。

物理参数、自由接触的箱子、动作控制方式、97 维观测、22 个动作、50 Hz 策略频率、终止条件和原托箱成功判据均未修改。旧 `smooth-v1` profile 保持原定义。

## 受力与倾角怎么测

- 通过脚名称匹配左右脚，不依赖正则展开顺序。
- `netforce` 是世界坐标下脚对地面的合力；**取负号**得到地面对脚的反作用力。没有取绝对值或猜符号。
- 脚底倾角是脚底 site 的本地 +Z 与世界 +Z 的夹角，对偏航不敏感。
- 每个 50 Hz 控制步读取一次末次物理子步的接触力，不是四个子步的平均值，也不能捕获所有子步峰值。
- 总静态重量参考值约为 `(19.796900 + 0.494922) × 9.81 = 199.06 N`。力矩为仿真执行器力矩，不是真机实测值。

## pilot300 训练与复现

在仓库根目录运行。下面使用新的输出目录，避免覆盖已有结果；Linux 将解释器路径换为 `./.venv/bin/python`。

```powershell
.\.venv\Scripts\python.exe -m lifting_lab train --device cuda --num-envs 512 --seed 11 --iterations 300 --resume shared_model/smooth-v1-20261028/policy.pt --reward-profile stance-v1 --learning-rate 0.0003 --entropy-coef 0.002 --output outputs/stance_study_20261029/pilot300_reproduce
```

这是原 `smooth-v1` seed 11 模型的继续训练，不是独立的新训练种子。模型、归一化和 optimizer 延续；环境与随机数状态重启，不保证逐位复现。固定训练质量约 0.494922 kg，rollout 长度 24，512 个并行环境，增加 **3,686,400 transitions**，累计 **15,974,400 transitions**，完成 iteration 1299。

原始 pilot：`outputs/stance_study_20261029/pilot300/`，需保留相邻的 `policy.pt` 和 `experiment.json`。PPO 学习率使用 adaptive schedule；`0.0003` 是此次显式设置的起始值，不表示训练过程中始终不变。

- pilot 模型 SHA-256：`adc8223dc24db0c82dd87e29881f20f7118353528cf7cfdf44f0399e4c62260b`
- 本轮运行源码指纹：`3b09ef90d8b244caa2d602e0783a8bae129a916458ba02e198eba4e55b5717b8`

单次诊断，基线和候选各运行一次；只替换 checkpoint 与新的输出目录，其他参数保持一致：

```powershell
.\.venv\Scripts\python.exe scripts/diagnose_stance.py --checkpoint outputs/stance_study_20261029/pilot300/policy.pt --device cpu --mass 0.4949224923474894 --seed 2027 --seconds 20 --output outputs/stance_study_20261029/pilot300_cpu_reproduce
```

基线 checkpoint 为 `shared_model/smooth-v1-20261028/policy.pt`。

## 独立的托箱与平脚评测

下面为实际 pilot 评测配置的复现命令，输出改为新目录：

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_stance.py --checkpoint outputs/stance_study_20261029/pilot300/policy.pt --device cuda --num-envs 16 --seed 5011 --seconds 20 --mass 0.247461246173745 --mass 0.494922492347489 --mass 0.989844984694979 --output outputs/stance_study_20261029/pilot300_eval_reproduce
```

匹配比较须使用相同设备、源码、seed、并行环境数、质量和时长。每个 slot 只计最初的一次 episode；失败后即冻结统计，即使为了继续仿真而 reset，也不会用新 episode 替换失败。

两个判据分别报告：

1. **原托箱成功**：完成 20 秒；全程箱体倾角 <10°、高度误差绝对值 <5 cm；无跌倒、非有限状态或禁止接触；连续失去前臂承托最多允许 6 个控制步。
2. **额外的严格平脚成功**：先满足原托箱成功，并有完整 `t >= 2 s` 稳态窗口；稳态中至少 95% 的样本，双脚同时接触且各自脚底倾角严格 <5°。此判据不额外限制躯干倾角或左右承重比例。

原始匹配结果在 `baseline_eval/stance_evaluation.json` 和 `pilot300_eval/stance_evaluation.json`：

| 箱子质量 | 旧模型：托箱／平脚 | pilot300：托箱／平脚 |
|---|---:|---:|
| 0.247461 kg | 16/16 ／ 0/16 | 16/16 ／ 16/16 |
| 0.494922 kg | 16/16 ／ 0/16 | 16/16 ／ 16/16 |
| 0.989845 kg（约 1 kg） | 16/16 ／ 0/16 | 14/16 ／ 14/16 |

约 1 kg 的两次失败均在 **0.8 秒** 出现连续 **7 步**的承托缺失，超过宽限；不是已经证明机器人本体倒地。汇总字段 `fallen` 也覆盖终止失败，应结合 `contact_failure` 等原因字段解释。这表示 pilot 的平脚表现改善，但该负载的启动承托可靠性退步，不能只报较轻负载的成功。

## mixed400：质量随机化续训与最终选型

为改善 pilot 在约 1 kg 时的启动承托问题，从 pilot300 继续训练同一 `stance-v1` 奖励；每次 reset 将箱子质量均匀采样为机器人质量的 1.25–5%，即约 **0.247461–0.989845 kg**。只改变质量及对应均匀箱体惯量，几何、摩擦与重心设置不变。元数据中的名义质量约 0.618653 kg 是区间中点，不能理解成整个训练只用了这个重量。

```powershell
.\.venv\Scripts\python.exe -m lifting_lab train --device cuda --num-envs 512 --seed 11 --iterations 400 --resume outputs/stance_study_20261029/pilot300/policy.pt --reward-profile stance-v1 --mass-range-fraction 0.0125 0.05 --learning-rate 0.0003 --entropy-coef 0.001 --output outputs/stance_study_20261029/mixed400_reproduce
```

训练阶段实际预算如下；已核对各阶段的 `completed.json` 与 `experiment.json`。

| 阶段 | 训练分布 | 本阶段 transitions | 累计 transitions |
|---|---|---:|---:|
| 原 smooth-v1 | 固定约 0.494922 kg | 12,288,000 | 12,288,000 |
| pilot300 | 固定约 0.494922 kg | 3,686,400 | 15,974,400 |
| mixed400 | 均匀质量约 0.247461–0.989845 kg | 4,915,200 | 20,889,600 |
| 其中选用的中间 model_1450 | 相同混合质量分布，仅取阶段前 151 次更新 | 1,855,488 | 17,829,888 |

最后一行是 mixed400 训练途中的快照，不是另外训练了一次。所选模型的预算不能写成完整 mixed400 的 20,889,600。

三者训练预算不同，且都是同一个训练 seed 的延续，因此比较用于判断本轮迭代是否改善，**不是等预算奖励消融，也不是多个独立训练种子的复现**。

`mixed400_eval/stance_evaluation.json` 的相同 GPU 评测（seed 5011，每种质量 16 次）显示：

| 箱子质量 | pilot300：托箱／平脚 | mixed400 末模型：托箱／平脚 | 选用的 model_1450：托箱／平脚 |
|---|---:|---:|---:|
| 0.247461 kg | 16/16 ／ 16/16 | 9/16 ／ 9/16 | 16/16 ／ 16/16 |
| 0.494922 kg | 16/16 ／ 16/16 | 16/16 ／ 16/16 | 16/16 ／ 16/16 |
| 0.989845 kg | 14/16 ／ 14/16 | 13/16 ／ 13/16 | 16/16 ／ 16/16 |

mixed400 共 38/48 成功，低于 pilot300 的 46/48。轻箱的 7 次失败在 0.10–0.54 秒触发禁止接触；重箱的 3 次失败在 0.56 秒出现 7 步连续承托缺失。**继续训练没有自动提升表现，因此未选择该阶段末模型。** 后续检查中间 checkpoint 属于基于该评测的选型，不能把这组数据再称为完全独立的最终测试集。

**最终选择：model_1450.pt。** 原始位置为 `outputs/stance_study_20261029/mixed400/model_1450.pt`；整理目录中重命名为 `shared_model/stance-v1-20261029/policy.pt`。SHA-256 为 `b0093a17767ba382549e392dc0c2f5844dea66b37173a6b6f1ba08c9c98a18f4`。完整选型评测在 `mixed151_eval/stance_evaluation.json`，汇总复制为整理目录的 `evaluation_seed5011.json`。

随后用相同设备、源码、并行环境数、质量和时长，以评测 seed **6011** 复核：三种质量各 **16/16** 次同时通过托箱和平脚判据，共 **48/48**。与选型 seed 5011 合计 96/96。复核结果来自 `mixed151_holdout/stance_evaluation.json`，整理目录中为 `evaluation_seed6011.json`；确认评测期间源码未改变。要复现，使用前面的评测命令，替换为所选模型、`--seed 6011` 和新的输出目录。

所选模型取舍是：保留脚底平放、恢复这三个负载的选型成功率；**没有达到完全对称站姿，躯干也比旧模型更倾斜**。不能以脚平放替代所有站姿质量要求。

整理目录中的 `selection.json` 记录实际 checkpoint iteration 和预算。`experiment.json` 原样保留完整混合质量 run 的元数据；`original_training_completed.json` 属于该 run 后续的末模型，不对应当前 `policy.pt`，不能将其哈希或 400 次训练预算套用到当前模型。

打开当前所选模型：

```powershell
.\.venv\Scripts\python.exe -m lifting_lab demo --device cpu --checkpoint shared_model/stance-v1-20261029/policy.pt --mass 0.4949224923474894 --seed 2027 --seconds 3600 --viewer --paused
```

## 开发检查

本轮完整测试套件运行 262 项，其中 245 项通过、17 项跳过。检查覆盖新奖励的受力符号、左右脚映射、倾角、默认弯膝姿态、profile／checkpoint 兼容，以及失败 slot 统计冻结。跳过项不计为通过；这些检查也不代替上面的实际模型评测或真机验证。

## 使用范围

本轮仅一个训练 seed 11、三个评测负载点、固定摩擦，没有推力、箱体相对位置扰动或实物材料标定。seed 5011 已用于选择候选；seed 6011 是选择后的初态 seed 复核，不能被说成多个独立训练 seed 的验证，也不能证明整个连续负载区间的鲁棒性。

本项目的 97 维观测仍包含仿真箱子状态。未接通真机测量、状态估计、关节映射和进入托箱姿态的过程。**不能把平脚评测通过解释为可以直接真机部署。** 真机边界见 [SIM_TO_REAL_ZH.md](SIM_TO_REAL_ZH.md)。
