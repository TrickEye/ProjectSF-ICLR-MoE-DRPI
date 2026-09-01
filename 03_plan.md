# 最终方案：以路径因果解释为主线的 DRPI 研究

## 总体定位

中心问题确定为：

> MoE 的 expert route 是否真正介导或解释行为变化，还是仅仅是一条粗粒度内部日志？

DRPI 首先是构造 route-controlled 因果对照的工具。只有证明减少 expert switching 会降低真实非目标损失，并优于直接输出 KL/保留集约束后，才将其升级为副作用控制方法。

当前仓库只有研究文档，没有代码、测试、配置或实验结果。默认主模型为固定 revision 的 `allenai/OLMoE-1B-7B-0924`。

## 计算环境

- 本地开发环境使用现有 `moe-steering` conda 环境和 PyTorch MPS。
- Stage 0-2 优先在 macOS MPS 上运行，采用 `batch_size=1`、`use_cache=False`、短序列和少量层。
- 先用 BF16 做无梯度模型检查；若具体算子不支持 BF16，则仅将该算子或整次 forward 降为 FP16/FP32 并记录，不能静默改变精度。
- 不预设 bitsandbytes 等 CUDA 量化方案可在 MPS 工作。量化必须通过独立 backend smoke test 后才能启用。
- Stage 3-5 若 MPS 在保留计算图时内存不足，迁移至 RTX 5060 Ti 16GB；服务器默认使用 int8 权重、router 保持 BF16/FP32，并验证输入梯度和有限差分未被量化破坏。
- 本地与服务器结果不得混合汇总，除非模型 revision、router hash、量化状态、hook、精度和数值测试均分别落盘。
- 当前 OLMoE 缓存不完整；正式执行前完成缺失权重校验，不把下载不完整误判成模型或 MPS 故障。

## 实施阶段

### 0. 工程与实验契约

- 初始化 Git 但不推送；建立 `pyproject.toml`、`.gitignore`、`configs/`、`src/drpi/`、`scripts/`、`tests/`、`results/raw/`、`results/summary/` 和 `artifacts/`。
- JSONL 保存不可覆盖的逐样本记录，JSON 保存检查与汇总报告。
- 每条记录包含模型 revision、router hash、commit、设备、dtype、实际量化状态、hook、prompt ID、split、seed、layer、token position、alpha 和 horizon。
- `alpha` 定义为扰动 RMS 与注入 hidden-state RMS 的比例，pilot 扫描 `[0.01, 0.03, 0.10, 0.30]`。
- 所有模型权重冻结；只有注入状态参与 autograd。

### 1. Stage 0：模型与后端检查

开发 `model_adapter.py` 和 `inspect_model.py`。

- 先运行 MPS backend smoke test：模型加载、一次无梯度 forward、一次需要输入梯度的短 forward，并记录峰值内存、fallback、dtype 和耗时。
- 检查全部 `named_modules()`、router/MoE 路径、输入输出 shape、weight shape、bias、norm、top-k、gate 归一化和共享专家。
- 运行时确认 `post_attention_layernorm` 输出是否同时进入 gate 和 experts；只有确认后，MoE block pre-hook 才能称为 shared-state 注入点。
- 手工 router 前向与模型公开 router logits 逐元素比较。
- 产出 `router_hook_report.json`，包含 router weight SHA-256 和 MPS 后端报告。
- 若没有 router/expert 共用输入，停止 shared-state DRPI，改为分别研究 router-only 与 expert-state intervention。
- 若 MPS 只能完成无梯度 forward，Stage 1-2 仍可在 MPS 完成，Stage 3 起迁移 CUDA；不得因此修改科学定义。

### 2. Stage 1：静态盲空间、捕获与注入

实现 `router_capture.py`、`static_space.py`、`interventions.py` 和 `runner.py`。

- 第一版只使用保守 `ker(W)`；SVD 使用 float32，basis 为 `[hidden, blind_dim]` 正交列基。
- capture 将 router 输出统一整理成 `[batch, seq, experts]`，保存 logits、gate probabilities、有序 top-k 和无序 expert 集合。
- 注入使用上下文管理器和 pre-hook，clone 后只修改最后一个有效 token，不原地修改共享输入。
- 先完成 T1-T3：router-null、capture consistency、`alpha=0`、非目标 token 局部性和 hook 清理。
- T1 失败立即停止，不通过放宽容差继续实验。
- centered `ker(CW)` 暂不实现。

### 3. Stage 2：动态泄漏与外部效度筛查

实现 `metrics.py`、`datasets.py` 和 `leakage_curve.py`。

- 注入层预注册为 `[2,4,6,8,10,12,14]`，每条 prompt 使用最后有效 token、8 个固定 Gaussian blind 方向和 4 个 RMS 强度。
- 使用固定 token 序列和 teacher forcing，记录 centered-logit 漂移、routing JS、top-k Jaccard、有序 Hamming、exact match、baseline margin 和首次切换层。
- 原始记录支持断点续跑；图表完全从 JSONL 生成。
- 泄漏必须超过数值噪声，在至少 3 个种子复现，并随强度、层距或 margin 呈稳定结构。
- 若没有结构化泄漏，停止 DRPI，转向研究静态 blind 长期稳定的条件。
- 泄漏成立后，用 full、static blind、route-visible 和随机方向做低成本外部效度扫描，同时记录输出 KL、next-token NLL 和目标 token log-prob。

### 4. Stage 3：margin/Jacobian 机制

实现 `margins.py`、`gradients.py` 和预测评测脚本。

- 每个下游层只取“最弱已选 expert-最强未选 expert”边界。
- 梯度相对于注入层、目标 token 的 shared state 求取；capture 构图阶段禁止提前 `detach()`。
- T4 使用多个 epsilon 的中心有限差分寻找稳定区间；至少 90% 边界相对误差不超过 10%，符号一致率至少 95%。
- calibration/test 严格分离；预测量固定为 `m + g^Tdelta`，报告 AUROC、Brier 和 reliability curve。
- 继续条件为 AUROC 的 95% CI 下界高于 0.5，且 Brier 优于常数基率预测。
- 若失败，依次缩小 alpha、缩短 horizon、复查 T4；仍失败则转向非线性、attention mixing 或二阶泄漏分析。
- 检验 blind-coordinate 梯度矩阵的谱、跨 calibration 子集主角度和 held-out 重构误差；低秩结构不稳定时不构建分布级 DRPI。

### 5. Stage 4：最小 DRPI

实现 `subspace.py`、`build_drpi.py` 和版本化 artifact。

- 将梯度投影为 `G_B = G B`，只在 blind coordinates 中做 SVD。
- dangerous rank 候选为 `[4,8,16,32]`，horizon 为 `[1,4,8]`；validation 选择 rank、horizon、layer 和 alpha。
- 第一版只做硬投影：先得到 `d_static`，再删除 dangerous subspace 分量。
- artifact 绑定模型 revision、router hash、hook、backend、dtype、量化状态、basis/dangerous rank、calibration IDs、配置和 commit。
- 若危险投影删除超过 80% 的目标方向范数，或目标效果无法恢复，报告目标改变与路径保持不可兼得，不继续增加算法复杂度。

### 6. Stage 5：路径因果意义与真实效用

统一比较 full steering、static blind、DRPI、随机等维子空间、target-only、直接 router bias、输出 KL 保持和保留集损失约束。

- 机制任务先使用实体/属性反事实方向。
- 外部指标固定为目标 token log-prob、held-out 输出 KL、next-token NLL 和非目标 token 漂移。
- 扫描强度后构造两类配对：目标效果相同但 route divergence 不同；route divergence 相同但目标效果不同。
- 机制主终点：受控改变 route divergence 是否导致预注册外部损失变化。
- 方法主终点：相同目标效果下，DRPI 是否优于 static blind、普通 steering、直接 KL 和保留集约束。
- 使用逐提示配对效应、10,000 次 bootstrap 95% CI 和配对置换检验；多层探索使用 Holm 校正。
- 路径与外部损失无因果联系时，论文转为 route-trace 解释边界；DRPI 只降低路径指标时，只称 route-control 工具。

### 7. 扩展与交付

- 第一里程碑必须产生：`router_hook_report.json`、静态 blind 泄漏曲线、通过 T4 的 gradient 报告、效果匹配的 static-vs-DRPI 图和 route/行为四象限报告。
- 里程碑通过后，才增加格式控制任务、第二种 MoE、自由生成质量和计算成本。
- 自由生成只比较任务和文本质量，不比较分叉后未对齐的 route。
- centered `ker(CW)` 仅作为后续消融，并实测公共 shift、top-k 和 gate weights。
- 跨语言、安全任务、第三模型和系统延迟不进入首轮。
- 每阶段同步更新 README，明确已运行验证、未验证项、backend 差异和触发的 Go/No-Go 分支。
