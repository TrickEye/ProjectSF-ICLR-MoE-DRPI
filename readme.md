# Dynamic Route-Preserving Intervention (DRPI)

面向深层 Mixture-of-Experts（MoE）模型的动态路由保持干预研究。本项目研究一个严格但容易被过度解读的局部事实：位于某层 router 零空间中的状态扰动不会改变该层的相对路由决策，但它仍会经过专家、残差连接、注意力和归一化，并可能在后续层重新影响专家选择。

项目首先测量这种局部路由不变性的“寿命”，再尝试利用下游路由 margin 的梯度，在改变目标状态或行为的同时，尽量保持固定输入上的后续专家路径。该方法称为 **Dynamic Route-Preserving Intervention（DRPI）**。

> 当前状态：Stage 0-5 的首轮工程骨架、实验入口和合成测试已经落地；真实 OLMoE 研究结果尚未产生。本文中的假设、预期贡献和目标阈值均不代表已经得到实验验证。

## 当前实现状态

- 已实现 OLMoE adapter、shared-state route capture、保守 `ker(W)`、生命周期安全注入和 teacher-forced runner。
- 已实现泄漏指标、临界 margin、T4 有限差分、blind-coordinate 危险子空间和硬投影 DRPI。
- 已实现固定四划分的反事实数据、追加式 JSONL、模型/revision/router-hash 绑定 artifact，以及直接 router bias、目标优化、输出 KL 与保留集损失基线。
- 已实现效果匹配插值、配对 bootstrap、配对置换检验和 route/behavior 四象限汇总。
- toy MoE 测试覆盖 T1-T4、padding、hook 清理、集合/顺序路由指标、空子空间及 rank 截断。

这些工程测试只验证实现契约，不构成“OLMoE 存在动态泄漏”“Jacobian 能预测切换”或“DRPI 优于基线”的实验结论。真实实验必须严格通过下面的阶段门槛。

## 核心问题

给定第 \(l\) 层 router：

$$
z_l = R_l h_l + b_l,
$$

若扰动 \(\delta_l\) 满足 \(R_l\delta_l=0\)，则它不会改变当前层的 router logits。然而对后续层 \(k>l\)，通常不能推出：

$$
z_k(h_l+\delta_l)=z_k(h_l).
$$

本项目围绕四个问题展开：

1. 静态 router-null 扰动会在多深的位置引发下游路径分歧？
2. 原始路由 margin、跨层 Jacobian、干预层和 token 类型能否预测这种泄漏？
3. 静态盲空间中是否仍存在可承载目标干预、同时对下游路径低敏感的非平凡子空间？
4. 在效果匹配条件下，路径保持是否真的减少非目标输出漂移？

项目不预设“router-blind 等于内容通道”或“router-visible 等于控制通道”，也不预设路径保持始终有益。不可兼得的结果同样是需要报告的因果边界。

## 方法概览

### 1. 静态盲空间

第一版使用 router 权重 \(R_l\) 的保守零空间 \(\ker(R_l)\)。目标方向 \(d\) 先投影为：

$$
\delta_l^{\mathrm{static}}=P_{\ker(R_l)}d.
$$

只有在逐元素 logits、top-k 和 gate 行为均通过测试后，才考虑放宽为 centered null space \(\ker(CR_l)\)，其中 \(C\) 去除所有专家 logits 的公共平移方向。

### 2. 动态泄漏测量

在固定 token 序列和 teacher forcing 下，比较干预前后同一 token 在后续层的路由。主要观察：

- top-k exact match、Jaccard 和顺序敏感 Hamming；
- centered-logit 漂移与 routing JS divergence；
- 首次专家切换层距和路径生存曲线；
- 原始 margin 与实际专家切换之间的关系。

自由生成用于评估行为和质量。生成文本一旦不同，token 不再对齐，因此未对齐轨迹的 Hamming 距离不能作为纵向路径保持证据。

### 3. DRPI

DRPI 始终在当前层静态盲空间内优化。对临近 top-k 边界的下游专家对收集 margin 梯度，将梯度投影到盲空间后做低秩分解，识别最容易导致未来路径切换的“危险方向”，再从静态目标方向中移除或软惩罚这些分量。

因此其保证边界是：

- 当前注入层的相对路由不变性应通过数值测试严格验证；
- 下游路径保持是数据条件下、局部线性化得到的近似性质；
- 最终行为、副作用和跨样本泛化必须独立实测。

## 仓库结构

```text
.
├── AGENTS.md
├── README.md / readme.md
├── 00_Idea2_Dynamic_Route_Preserving_Intervention.md
├── 00_Idea2_Introduction_and_Implementation_Guide.md
├── 03_plan.md
├── configs/
│   └── olmoe_pilot.yaml
├── src/drpi/
│   ├── model_adapter.py
│   ├── router_capture.py
│   ├── static_space.py
│   ├── interventions.py
│   ├── margins.py
│   ├── gradients.py
│   ├── subspace.py
│   ├── datasets.py
│   ├── metrics.py
│   ├── runner.py
│   ├── records.py
│   ├── statistics.py
│   └── artifacts.py
├── scripts/
│   ├── inspect_model.py
│   ├── leakage_curve.py
│   ├── verify_margin_gradient.py
│   ├── build_drpi.py
│   ├── evaluate_interventions.py
│   └── summarize_effect_matching.py
├── tests/
└── results/
```

其中 `model_adapter.py` 是唯一允许感知具体模型模块路径的工程边界。其他模块只依赖统一的 adapter 接口，不能硬编码某个 Transformers revision 的 OLMoE module path。

## 运行入口

```bash
export PYTHONPATH=src
conda run -n moe-steering pytest -q
conda run -n moe-steering python scripts/generate_counterfactuals.py
conda run -n moe-steering python scripts/inspect_model.py \
  --config configs/olmoe_pilot.yaml \
  --out results/summary/router_hook_report.json
```

MPS 必须在能访问 Metal 的非沙箱进程中运行。`inspect_model.py` 会先检查 safetensors 索引列出的每个权重分片；缓存不完整时写出 `blocked_incomplete_cache` 报告并停止，而不是把加载错误归因于 MPS。

Stage 0 通过后，按顺序进行小规模 smoke test：

```bash
conda run -n moe-steering python scripts/leakage_curve.py --limit 8
conda run -n moe-steering python scripts/verify_margin_gradient.py \
  --layer 6 --downstream-layer 7 --limit 8
conda run -n moe-steering python scripts/build_target_direction.py \
  --layer 6 --out artifacts/target_layer6.pt --limit 16
conda run -n moe-steering python scripts/build_drpi.py \
  --target-direction artifacts/target_layer6.pt \
  --layer 6 --horizon 4 --rank 8 --limit 16 \
  --out artifacts/drpi_layer6.pt
```

上述 `--limit` 仅用于 smoke test。正式 calibration/validation/test 运行不得用 test 数据选择 layer、rank、horizon、alpha 或正则系数。

## 最小实验路线

实现必须按以下顺序推进：

1. 检查目标模型结构，确定 router 与 expert 实际接收的张量、归一化位置和权重形状。
2. 实现 route capture、共享状态注入和静态盲空间投影。
3. 通过即时 router-null、capture 一致性、注入局部性和 margin 梯度有限差分测试。
4. 在 OLMoE-1B-7B 上绘制静态 blind 扰动的下游路径生存曲线。
5. 在独立测试提示上评估 `margin + g^T delta` 对专家切换的 AUROC、Brier score 和校准。
6. 仅在动态泄漏可重复且一阶预测有效时构建第一版 DRPI。
7. 在一个对齐反事实任务上，按相同目标效果比较 full steering、static blind、DRPI 和强基线。

首轮应保持范围克制：单模型、单 token 位置、短下游 horizon、少量临界 margin、full precision 或 bfloat16。量化、长生成、多模型、跨语言和安全任务均属于后续扩展。

## 必须通过的验证

在开展大规模实验前，至少完成以下测试：

- **Router-null 正确性**：注入位置的 baseline 与 edited router logits 在容差内一致，top-k 完全相同。
- **Route capture 可信性**：hook 捕获结果与模型公开 router logits 或手工 router 前向一致。
- **注入局部性**：`alpha=0` 复现 baseline；当前 router 中未注入 token 的 logits 不变。
- **梯度有限差分**：下游 margin 的有限差分与 `grad_margin dot v` 一致。

若第一项失败，必须停止所有下游实验并修正 hook 点或投影实现，不能把结构无效的结果纳入研究结论。

## 评测与基线

核心比较同时采用：

- 相同干预范数；
- 相同目标效果；
- 相同路由效果。

至少包含普通 activation steering、静态 blind/visible 投影、随机等维子空间、直接 router-logit bias、最小 margin crossing、强制 expert 替换或 masking、专家输出缩放，以及不含路径约束的目标优化。主要终点是：在相同目标成功率下，DRPI 是否比静态 blind 投影降低下游路径分歧，同时满足预设输出质量非劣界限。

所有结果采用逐提示配对比较，报告效应量和 95% bootstrap 置信区间；超参数只在验证集选择，最终结论来自独立测试集。不能只展示最佳层、最佳强度或单一种子。

## 可追溯产物

每次实验结果至少记录：

- 模型名称与 revision、代码 commit、dtype 和设备；
- router hook path、权重形状、SVD 阈值和数值秩；
- prompt ID、数据划分、随机种子和 token 位置；
- injection layer、alpha、horizon 和方向构建配置；
- 目标指标、路径指标、输出质量指标和即时盲性误差。

DRPI 方向不能跨模型、跨 revision 或跨 hook 点复用。原始逐样本记录应保留，汇总图表由脚本生成，不能手工改写结果。

## Go/No-Go 标准

扩大实验规模前，需要同时观察到：

1. 可重复的下游动态泄漏；
2. 可接受的一阶 margin/Jacobian 预测能力；
3. DRPI 相对静态 blind 的非平凡帕累托改善。

如果只有前两项成立，项目应转为机制研究；如果静态 blind 本身长期稳定，应研究其成立条件；如果目标方向被动态约束几乎全部删除，应报告“目标改变与路径保持不可兼得”的任务边界。不能通过扩大模型或挑选层来掩盖 No-Go 结果。

## 项目文档

- [`00_Idea2_Dynamic_Route_Preserving_Intervention.md`](00_Idea2_Dynamic_Route_Preserving_Intervention.md)：完整研究问题、方法、实验设计、统计方案和投稿边界。
- [`00_Idea2_Introduction_and_Implementation_Guide.md`](00_Idea2_Introduction_and_Implementation_Guide.md)：论文叙事、代码架构、关键实现蓝图、测试和首轮实验清单。
- [`01_术语清单与技术解释.md`](01_术语清单与技术解释.md)：面向计算机科学本科生的术语、公式和实验概念说明。
- [`01_极简解释.md`](01_极简解释.md)：面向计算机科学爱好者的研究脉络、方法和意义概览。
- [`02_研究意义质疑与叙事重构.md`](02_研究意义质疑与叙事重构.md)：route trace 的因果意义、外部效度和论文叙事边界。
- [`03_plan.md`](03_plan.md)：批准后的分阶段实施与 Go/No-Go 方案。
- [`04_项目运行手册.md`](04_项目运行手册.md)：从环境、模型检查到冻结 test 的逐步运行命令、产物和停止条件。
- [`AGENTS.md`](AGENTS.md)：面向后续编码代理和贡献者的执行约束。

## 参考起点

- Charles Ye, Bo Yuan, Lee Sharkey. *Polysemantic Experts, Monosemantic Paths: Routing as Control in MoEs*. arXiv:2604.17837, 2026. <https://arxiv.org/abs/2604.17837>
- [ICLR 2027 Dates and Deadlines](https://iclr.cc/Conferences/2027/Dates)

完整投稿前仍需系统检索 activation steering、causal tracing、route editing、Jacobian subspace intervention、MoE interpretability 和 route-preserving optimization。未经检索与实验验证，不使用“首次”“已证明无副作用”或“无需代价迁移”等表述。
