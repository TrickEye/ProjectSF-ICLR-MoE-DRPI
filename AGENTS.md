# AGENTS.md

本文规定在本仓库中进行研究、编码、实验和写作时必须遵守的规则。适用于仓库根目录及其所有子目录。

## 0. 项目概况

1. 开发机器：macOS (18GB unified memory); 另外可获取服务器资源：windows (16GB Ram, Nvidia RTX 5060ti 16GB VRAM)；可用conda创建环境或检索已有环境
2. 比较偏好的模型：OLMoE-7B，可在int4/int8量化下运行
3. 代码风格偏好：执行最小化的改动和创建，不用对未来可能的功能点做预留，主要以实现当前目标为目的

## 1. 项目定位

本项目研究 MoE 中局部 router-null 干预的下游动态泄漏，以及 Dynamic Route-Preserving Intervention（DRPI）。当前仓库首先是研究设计仓库；代码、实验结果和论文主张必须随着证据逐步建立。

开始工作前完整阅读：

1. `Idea2_Dynamic_Route_Preserving_Intervention.md`
2. `Idea2_Introduction_and_Implementation_Guide.md`
3. `readme.md`

若文档之间存在冲突，采用更保守、可证伪且更接近实际模型行为的解释，并在变更说明中记录冲突。不要静默扩大论文主张。

## 2. 不可破坏的科学边界

- `R_l @ delta ~= 0` 只保证注入层当前 token 的 router logits 在数值容差内不变，不保证后续层路径、输出行为、隐私或无副作用。
- “即时保持”“纵向保持”和“自由生成行为”是不同评测对象，禁止混用。
- 纵向路径比较必须使用相同 token 序列和 teacher forcing。自由生成文本不同后，不得把未对齐 token 的路径距离作为路径保持主证据。
- 主实验中的 shared-state intervention 必须作用于 router 与 expert 共同接收的实际张量。router-output/logit 修改只能作为基线，不能称为 shared-state intervention。
- 第一版使用保守的 `ker(W)`。只有严格验证公共 logit shift、top-k 和 gate 权重均保持后，才可启用 centered `ker(C @ W)`。
- DRPI 对当前层的盲性必须实测；对下游层仅可表述为分布条件下、局部线性化的近似约束。
- 不把 router-blind 方向命名为“内容通道”，不把 router-visible 方向命名为“控制通道”，除非有独立因果证据。
- 不预设路径保持有益。负结果、不可兼得区域和失败案例必须保留。

## 3. 实现边界

目标代码结构遵循 `src/drpi/` 的模块划分：

- `model_adapter.py`：唯一知道模型内部 module 名称、hook path 和架构差异的模块。
- `router_capture.py`：捕获 router 输入、logits、top-k；构建梯度时不得提前 `detach()`。
- `static_space.py`：盲空间基、投影和数值断言。
- `interventions.py`：shared-state pre-hook 注入；不得原地修改可能被残差分支复用的张量。
- `margins.py`：临界 top-k 边界的定义和提取。
- `gradients.py`：下游 margin 相对注入状态的梯度。
- `subspace.py`：危险子空间和 DRPI 方向。
- `datasets.py`：固定序列、反事实配对和划分。
- `metrics.py`：路由、目标和质量指标。
- `runner.py`：teacher forcing 与 generation 的统一执行入口。

模型适配器之外禁止硬编码具体 Transformers 版本的 module path。先运行模型检查脚本，记录 `named_modules()`、router 输入形状、权重形状、bias、norm、共享专家和 top-k 规则，再实现 adapter。

所有 hook 必须有明确生命周期，并在 `finally` 或上下文管理器退出时移除。一次实验不能遗留 hook 影响后续 forward。

## 4. 实现顺序

严格按以下阶段推进，前置验证失败时不得跳到下一阶段：

### Stage 0：模型检查

- 确认 router 和 expert 的真实输入点。
- 确认干预位于 RMSNorm/LayerNorm 前还是后。
- 记录 router 权重方向、bias、额外变换、top-k 和 gate 规则。

产物：`router_hook_report.json`。

### Stage 1：静态盲性与捕获

- 实现 blind basis、投影、capture 和注入。
- 先限定 `batch_size=1` 和一个 token 位置，排除 padding、cache 和广播歧义。
- 通过 T1–T3 后才扩大批量。

### Stage 2：动态泄漏

- 使用固定提示、teacher forcing、固定随机方向和按 hidden-state RMS 定义的强度。
- 绘制不同注入层、强度和层距下的路径生存曲线。
- 在查看任务效果前先确认泄漏是否真实存在。

### Stage 3：可预测性

- 对最弱已选 expert 与最佳未选 expert 的 margin 求梯度。
- 先通过有限差分 T4，再评估 AUROC、Brier score 和 reliability curve。
- 若预测很差，依次检查 hook、有限差分、alpha 和 horizon，不直接换大模型。

### Stage 4：DRPI 与效果匹配评测

- 梯度必须在静态 blind coordinates 中分解。
- calibration、validation 和 test 严格分离。
- 比较 full、static blind、DRPI、随机等维子空间、无路径约束目标优化及直接路由基线。
- 扫描强度后按相同目标效果插值比较，不得只在相同 `alpha` 下宣布胜出。

## 5. 强制测试

新增核心代码时至少覆盖：

1. **T1 router-null**：baseline/edited 注入层 logits 在约定容差内一致，top-k 完全相同。
2. **T2 capture consistency**：hook 捕获与公开 router 输出或手工前向一致。
3. **T3 intervention locality**：`alpha=0` 复现 baseline；当前 router 的非目标 token 不变。
4. **T4 finite difference**：小 blind-space 方向上的 margin 有限差分与梯度内积一致。

还应测试：

- projector 的维度、正交性、dtype 和 device 保持；
- padding 下每个样本最后有效 token 的解析；
- hook 移除后模型恢复 baseline；
- top-k 集合指标与顺序敏感指标的区别；
- 空危险子空间、rank 截断和零方向等边界情况。

修改核心数学或 hook 逻辑时运行完整测试集。若由于模型、算力或依赖无法运行，必须明确列出未验证项，不能声称通过。

## 6. 数据与统计纪律

- 方向提取、危险子空间构建、超参数选择和最终评测分别使用 train/calibration、validation 和 test 数据。
- prompt ID 和随机种子必须落盘；不得让测试提示参与方向、rank、layer、alpha 或正则系数选择。
- 核心结论使用逐提示配对比较，报告效应量、95% bootstrap CI 和失败分布。
- 主要终点预先固定为效果匹配下的下游路径分歧，并附输出质量非劣约束。
- 同时保留范数匹配、目标效果匹配和路由效果匹配结果。
- 不只报告均值、最佳层、最佳强度或单一种子；至少提供分位数和典型失败案例。
- pilot 后才能冻结最小实际改善阈值。目标值不得写成已观察结果。

## 7. 结果与产物规范

结果文件优先使用结构化 JSON/JSONL。每次运行至少记录：

- schema/version 和生成时间；
- 模型 ID、revision 和 router weight hash；
- 代码 commit（若仓库已初始化 Git）；
- dtype、设备和实际量化状态；
- hook path、输入/权重形状、SVD tolerance/rank；
- 数据划分、prompt ID、随机种子；
- 注入层、token 位置、方向来源、alpha、horizon、dangerous rank；
- 即时盲性误差、路由指标、目标指标和输出质量指标。

原始逐样本结果不可被汇总文件覆盖。图表和表格由脚本从原始记录生成，不手工修改数值。生成的 DRPI artifact 必须绑定模型 revision、router weight hash、hook 点和 calibration IDs；不跨模型或 revision 复用。

不要提交模型权重、缓存、访问令牌、私有数据或超大中间张量。引入这些目录时同步更新 `.gitignore`，但不要删除用户已有产物。

## 8. 工程实践

- Python 代码使用类型标注和简短 docstring；复杂张量注明 shape。
- 保持 dtype/device 显式，不做隐式 CPU/GPU 搬运。
- 避免在保留计算图时积累多个 batch；每次只收集少量边界并及时释放图。
- 大规模执行前先跑小批量 smoke test。长任务提供进度与可恢复输出。
- 配置、代码和结果分离；实验参数进入 YAML/CLI，不散落为脚本常量。
- 优先做局部、可审阅的修改。未经任务要求不重构无关文件，不覆盖用户现有改动。
- 新依赖必须说明用途并固定兼容范围；不要为尚未实施的扩展提前增加依赖。

## 9. 写作规则

明确区分四类陈述：

- **已验证事实**：由当前仓库可追溯实验支持；
- **外部证据**：有准确文献来源；
- **研究假设**：尚待检验；
- **目标或预期**：用于设计，不是结果。

禁止在缺乏证据时使用：

- “首次”“state of the art”；
- “不改变计算过程”；
- “数学保证无副作用”；
- “blind space 就是内容空间”；
- “跨语言零成本迁移”；
- 将当前层不变性写成完整路由轨迹或下游行为保证。

论文结果措辞必须由实际数据决定。文献引用应指向原始来源；提交前对 activation steering、causal tracing、route editing、Jacobian subspace intervention、MoE interpretability 和 route-preserving optimization 做系统检索。

## 10. 停止与转向条件

- T1 失败：停止下游实验，修复 hook 或投影。
- 静态 blind 没有有意义的动态泄漏：研究稳定条件，不强行开发 DRPI。
- 小扰动、短 horizon 且有限差分正确时 Jacobian 仍接近随机：转向非线性或结构性泄漏分析。
- 动态约束删除几乎全部目标方向：报告目标与路径保持不可兼得。
- 效果匹配后 DRPI 不优于静态投影或简单基线：不包装成新控制方法，转为机制结论。

任何转向都必须保留负结果和失败原因，不通过扩大搜索空间或挑选正面样本改变故事。

## 11. 完成定义

一个实现任务只有在以下条件全部满足时才算完成：

- 代码符合上述科学与模块边界；
- 相应单元测试或 smoke test 已运行并记录结果；
- 配置与输出可追溯；
- README 或相关文档已同步更新；
- 已明确报告未运行的验证、已知限制和结论边界。

首个研究里程碑应产生四项可审阅产物：`router_hook_report.json`、静态 blind 泄漏曲线、通过有限差分的 margin-gradient 报告，以及 static blind 与 DRPI 在效果匹配下的路径分歧图。
