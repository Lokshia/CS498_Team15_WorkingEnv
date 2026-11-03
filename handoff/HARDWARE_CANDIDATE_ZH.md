# 给 TA：新增真机控制适配候选

> **当前现场已改用原版 deploy.py task 流程。** 请优先使用 [原生 task 安装包](Team15_native_booster_deploy_task_20261103.zip) 和 [操作说明](../deployment/booster_deploy_task/README_ZH.md)。下面是保留的独立 bridge 方案，不是新 task 的启动方法。

[下载完整新包](Team15_hardware_control_candidate_20261103.zip)。保留解压后的 `deployment/` 目录结构；模型/合同在 `team15_proprio_package/`，新的控制入口在 `hardware_bridge/`。

```text
Team15_hardware_control_candidate/
  README_FIRST_ZH.md
  HANDOFF_STATUS.json
  SHA256SUMS.json
  deployment/
    hardware_bridge/
      run.py
      README_ZH.md
      profiles/site.template.json
      docs/TA_QUESTIONS_ZH.md
      docs/TA_QUESTIONS_EN.md
      validation/
    team15_proprio_package/
      tasks/team15_proprio/models/
      ...
```

**单独复制旧 `tasks/team15_proprio/` 不能控制真机。** 新入口使用固定版本 `booster_deploy` 的 ROS 2 协议，包含实际 SERIAL 电机发布、Custom/退出模式、审核路点、限位限速、超时和停止路径。

现场先看 [中文 TA 提问清单](../deployment/hardware_bridge/docs/TA_QUESTIONS_ZH.md) 或 [英文版](../deployment/hardware_bridge/docs/TA_QUESTIONS_EN.md)。让 TA 提供该机器人实际配置/接口文件，填写本地现场配置，先预检和只读，之后由 TA 主持有支撑的控制测试。[具体步骤](../deployment/hardware_bridge/README_ZH.md)。

当前验证：**76 项软件/模拟接口测试通过；新动作处理链在原生部署仿真中 0.5 kg、3 次 20 秒通过。** 模型 534 条推理测试向量通过。未连接真实机器人，真实电机、准备/放箱过程和站稳效果均未验证。

配置模板的未知项是 `null`；缺项会拒绝运行。`simulation_only` 配置不能用于 `--live`。不要使用仿真 Kp/Kd 填写真机参数，也不要把测试通过当成现场标定完成。

上传分支：`codex/proprio-hardware-adapter`。此分支包含完整依赖的冻结包，可直接下载，无需先合并其他 PR。原验证 ZIP 没有修改。
