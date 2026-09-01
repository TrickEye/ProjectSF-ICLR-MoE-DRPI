# DRPI 补充：Introduction 叙事与可执行实现指南

> 本文档服务于《从局部零空间到动态路由保持干预》。目标是把论文说清楚、把第一版实验跑出来；不把数学记号当作贡献本身。

---

## 1. 这篇论文到底讲什么

一句话版本：**MoE 中一个看不见当前路由器的改动，并不等于它看不见后续路由器；我们测量这种影响何时回来，并尝试在不改变当前选择的前提下，让它在后续几层也尽量不回来。**

不要把论文讲成“我们发现了内容通道并在其中编辑内容”。这个说法既强又脆弱，因为它默认了一个没有被证明的语义分工。更稳健的主张是：

- 单层 router-null space 是精确、易计算的局部不变性；
- 深层 MoE 不是单层系统，专家输出会把该扰动带到下一层；
- 因此，真正值得研究的是 **local invariance 的寿命**；
- 若可预测这种泄漏，就可以在当前层允许的自由度内主动避开会改变后续路由的方向；
- 这给出一种更干净的“路径保持状态干预”，也告诉我们它在哪些任务中根本不可行。

论文的主角是“局部保证与深层动力学之间的缺口”，DRPI 是把这个缺口变成可操作干预的工具。

---

## 2. Introduction 应如何讲

### 2.1 审稿人需要在前两段理解的冲突

MoE 的每层都必须根据一个连续状态作出离散专家选择。解释性研究已经能从某一层的 router 权重得到一个非常干净的事实：有些状态改动不会改变这一层的专家选择。这个事实很诱人，因为它似乎为“改内容、不改计算路径”提供了坐标系。

但是 MoE 的计算并不在这一层结束。这个当前 router 看不到的改动仍被专家网络处理，并写回残差流。它到下一层时可以重新影响路由。于是，一个直观但未被回答的问题出现：**局部路由不变性究竟能活多久？如果我们真想保持路径，单层投影够不够？**

这就是 introduction 的张力。不要先谈 SVD、行空间、路径单义或跨语言；先让读者看到“已有的漂亮保证为什么不够”。

### 2.2 六段结构

#### 第 1 段：问题的重要性

说明 expert routing 是 MoE 的稀疏计算机制，也是模型内部计算过程最可观测的离散轨迹。若希望做可解释干预、模型编辑或控制，能否在改变状态时保持或有意改变这一轨迹，是一个基础因果问题。

避免泛泛说“MoE 很重要”。一句具体的动机足够：专家路径改变时，输出变化究竟来自内容状态还是计算路径，很难分辨；能够控制路径是把两者拆开的前提。

#### 第 2 段：已有发现和它的诱惑

介绍现有 router-null / visible-blind 分解：给定一层 router，只需线性代数就能构造对**当前层**路由 logits 不起作用的扰动。这比从数据训练出来的 probe 更强，因为该局部陈述是精确的。

此处必须加上限定词“当前层”“当前 token”“router 的实际输入点”。这不是弱化卖点，而是建立可信度。

#### 第 3 段：核心缺口

指出目前缺少的是动态因果检验。一个局部 blind 扰动经过 selected experts、attention、residual 和下一层 norm 后，会再进入新的 router。局部盲性没有理由自动变成跨层盲性。

可以用一个不带公式的类比：一扇门看不到的信号，不代表走过房间后下一扇门也看不到它。这里的“房间”是被选中的专家及后续网络。

#### 第 4 段：本文问题与方法直觉

本文先绘制“局部盲性泄漏”的深度曲线：从第 $l$ 层注入后，后续第几层开始选到不同专家？然后使用下游临界专家对的梯度，识别哪些当前可行的扰动方向最容易导致未来路径切换。DRPI 仅在当前层 blind space 中选方向，同时惩罚或约束这些未来敏感方向。

这段只说“梯度告诉我们哪些方向危险”，不必放 Jacobian 符号。

#### 第 5 段：为什么这不是简单投影

预先回应最自然的审稿人质疑：单层 blind projection 的确只是线性投影，且未必比简单 router-logit bias 更好。DRPI 的贡献不在于“让当前层 routing 不变”，那本来就容易；它要求在不同干预强度下同时比较目标改变、后续路径分歧和输出副作用，并必须击败静态投影、随机等维子空间、最小 margin crossing 和直接路由修改。

#### 第 6 段：贡献与边界

用三项贡献收束，不要列五六项小贡献：

1. 一个区分即时、纵向和自由生成路径保持的评测协议；
2. 对静态 router-null 干预跨层泄漏的系统测量与可预测性分析；
3. DRPI，一个仅在当前 router-blind 自由度内优化、但显式控制下游路由风险的方法。

最后明确：本文不假设路径保持总是有益；我们识别它何时有益、何时目标行为必须伴随路径改变。这个边界使负结果具有论文价值。

### 2.3 可直接改写为论文的英文骨架

以下是论证骨架，不是最终润色稿；提交前需要按实际实验结果把 `often`、`can`、`we find` 等措辞收紧。

```text
Mixture-of-Experts models expose a discrete trace of computation: for each
token, a router selects a small set of experts at every layer. This trace is
central to both efficient inference and mechanistic interpretability. Yet it
remains unclear whether one can alter a model's internal state while preserving
the computation selected by its routers.

For a single MoE layer, this question has an appealing exact answer. A
perturbation in the null space of the router leaves that layer's routing logits
unchanged. This local invariance has motivated an interpretation of router-blind
directions as a content channel. However, a perturbation that is invisible to
one router is still processed by its selected experts and written back into the
residual stream. It can therefore become visible again to routers at subsequent
layers. Whether, when, and why this happens has not been systematically tested.

We study the lifetime of local routing invariance. We first measure how
router-null perturbations propagate through depth under teacher forcing, and
show how downstream routing margins and their local sensitivities predict route
divergence [replace with observed result]. We then introduce Dynamic
Route-Preserving Intervention (DRPI). DRPI restricts interventions to the exact
null space of the injection-layer router while selecting, within that space,
directions that minimally affect vulnerable downstream routing boundaries.

Our contribution is not another way to flip an expert choice: direct logit
biases and minimum-margin interventions already do that cheaply. Instead, DRPI
tests and controls a stricter property: changing a target behavior while keeping
the subsequent computation as stable as possible. Across ... [only after
experiments], we evaluate this trade-off against static projections, direct
routing interventions, and effect-matched activation steering. The results
identify both regimes where path preservation reduces collateral changes and
regimes where the desired behavior necessarily changes the route.
```

### 2.4 不要写进 introduction 的话

- “blind channel 就是内容，visible channel 就是控制。”应改为“这提供了一个局部路由不变方向”。
- “我们在不改变计算过程的情况下修改内容。”应改为“我们尝试在固定输入轨迹上保持后续路由，并实证量化失败程度”。
- “数学保证无副作用。”没有这种保证。
- “第一个。”除非完成系统文献检索并能精确排除相近工作。
- “跨语言零成本迁移。”这是可选实验，不应承担主故事。

---

## 3. 方法应如何讲，不靠 math wash

方法正文只需要回答四件事。

### 3.1 在哪里加干预

对一个 transformer block，先确认 router 真正看到的张量 `router_input`。在不同 MoE 实现中它可能是：

```text
residual -> pre_moe_norm -> router_input -> router logits -> top-k experts
                                  |                 \
                                  |                  -> gate weights
                                  -> expert input
```

主实验只能在 `router_input` 同时也是 expert input 时讨论 shared-state intervention。若 router 使用归一化后的状态、expert 使用归一化前状态，必须分开报告；不能把 hook 在 block 输出的结果误称为 router-null intervention。

### 3.2 什么叫“保持路径”

把路由保持拆为三个可测对象：

| 名称 | 比较对象 | 使用场景 |
|---|---|---|
| 即时保持 | 注入层当前 token 的 experts/logits | 静态投影单元测试 |
| 纵向保持 | 同一固定 token 序列在更深层的 experts | 主要机制和 DRPI 评测 |
| 自由生成行为 | 解码文本与任务指标 | 应用效果，不能直接比较未对齐路径 |

纵向保持必须使用 `labels` 或输入 tokens 做完整 teacher forcing。干预前后每一个位置都是同一 token，才能说“这个 token 在后续层是否换路”。

### 3.3 DRPI 实际做什么

第一版不用解大型约束优化。按下列顺序实现：

1. 对注入层的 router 权重 $R_{l}$ 做一次 SVD，得到静态的路由不可见投影权重 blind projector `P_blind`；
2. 用该 projector 把原始目标方向 `d` 变成合法的 `d_static`，这个d_static 保证不会改变注入层的路由；
3. 对少量（比较近的）下游层和最接近 top-k 边界（几乎选不上和几乎要选上）的专家对，计算“该边界相对注入层 hidden state 的梯度”；
4. 把这些梯度投影到 blind space；
5. 对投影后的梯度做低秩 SVD，抽取最危险的 `rank` 个方向；
6. 从 `d_static` 中删掉这些危险方向，得到 `d_drpi`；
7. 用同一个 hook 在评测集注入 `alpha * d_drpi`。

直观上，静态 projection 说“别影响此刻 router”；DRPI 进一步说“在仍然合法的选择里，尽量少碰会在未来掀翻专家边界的方向”。

### 3.4 第一版应刻意简化的地方

- 只干预一个 token 位置，例如最后一个 prompt token；
- 下游约束只覆盖接下来 4、8 或 12 层，而非直到最后一层；
- 每个下游 router 只取 baseline top-k 中最弱的已选 expert，与分数最高的未选 expert 构成一个 margin；
- 每层只随机采样少量 prompt/token，收集 128–512 个梯度；
- 首先只做一个方向：对齐实体替换的 mean-difference direction；

这些简化不会改变研究问题，反而使机制结论更容易诊断。

---

## 4. 代码目录与职责

以下目录适用于新的独立实验仓库；不要把 DRPI 混入已有无关项目。

```text
drpi/
  configs/
    olmoe_pilot.yaml
  src/drpi/
    model_adapter.py       # 唯一知道模型内部 module 名称的地方
    router_capture.py      # 捕获输入、logits、top-k 路由
    static_space.py        # router blind projector 和数值断言
    interventions.py       # forward-pre-hook 注入器
    margins.py             # 从 router logits 提取临界边界
    gradients.py           # autograd.grad 收集下游 margin 梯度
    subspace.py            # 低秩危险子空间与 DRPI direction
    datasets.py            # 固定 token 的 prompts / counterfactual pairs
    metrics.py             # route survival、JS、任务与质量指标
    runner.py              # teacher forcing 和 generation 的统一入口
  scripts/
    inspect_model.py
    capture_baseline.py
    leakage_curve.py
    build_drpi.py
    evaluate_interventions.py
  tests/
    test_static_space.py
    test_intervention.py
    test_metrics.py
```

`model_adapter.py` 是最重要的工程边界。每个模型版本只改这里，其他逻辑只依赖统一接口：

```python
class MoEAdapter(Protocol):
    def num_layers(self) -> int: ...
    def router(self, layer: int) -> torch.nn.Module: ...
    def router_weight(self, layer: int) -> torch.Tensor: ...
    def router_input_module(self, layer: int) -> torch.nn.Module: ...
    def router_logits_from_input(self, layer: int, x: torch.Tensor) -> torch.Tensor: ...
```

先用 `scripts/inspect_model.py` 打印 `named_modules()`、router 权重形状和一次 forward 中 router 输入的形状。不能凭某个 Transformers 版本的类名假设 OLMoE 的 module path。

---

## 5. 关键实现：静态 blind projector

对 router 的线性层权重 `W`，一般形状为 `[num_experts, hidden_size]`。为保持当前 top-k 路由，只要让 `W @ delta` 为常数向量即可；工程上最简单且更保守的选择是让它为零，即使用 `ker(W)`。这会少保留一个公共 logit-shift 自由度，但避免因为 router bias、后处理或实现差异而引入不必要的错误。

先实现这个保守版本；在单元测试通过后，才把 `ker(W)` 放宽为 centered `ker(C @ W)`，并单独验证 top-k 与 gate softmax 是否保持。

```python
# src/drpi/static_space.py
from __future__ import annotations
import torch


@torch.no_grad()
def blind_basis(weight: torch.Tensor, rtol: float = 1e-6) -> torch.Tensor:
    """Return B [hidden, blind_dim] with orthonormal columns and W @ B ~= 0."""
    w = weight.detach().float()
    _, s, vh = torch.linalg.svd(w, full_matrices=True)
    tol = rtol * s[0].item() if s.numel() else 0.0
    rank = int((s > tol).sum().item())
    # vh is [hidden, hidden] when full_matrices=True; right singular vectors are rows.
    return vh[rank:].T.contiguous().to(device=weight.device, dtype=weight.dtype)


@torch.no_grad()
def project_to_blind(direction: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """direction [hidden] or [..., hidden], preserving its original dtype/device."""
    return (direction @ basis) @ basis.T


@torch.no_grad()
def assert_router_blind(weight: torch.Tensor, delta: torch.Tensor, atol: float = 2e-5) -> None:
    residual = torch.linalg.vector_norm(delta.float() @ weight.float().T, dim=-1).max()
    if residual.item() > atol:
        raise AssertionError(f"router-null check failed: max ||W delta||={residual.item():.3e}")
```

最先写的测试不是任务指标，而是：随机 `d` 经投影后，`router(x + alpha*d_static)` 的 logits 与 `router(x)` 在注入 token 上逐元素相同或只存在可解释的浮点误差。

---

## 6. 关键实现：捕获路由与注入状态

### 6.1 捕获器

不依赖 `output_router_logits=True` 是否被某个模型实现支持。直接对 router module 注册 hook，统一存储：

- `router_input`: `[batch, seq, hidden]`；
- `router_logits`: `[batch, seq, experts]`；
- `topk_indices`、`topk_values`。

```python
# src/drpi/router_capture.py
from dataclasses import dataclass
import torch


@dataclass
class RouteRecord:
    router_input: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    topk_indices: torch.Tensor | None = None
    topk_values: torch.Tensor | None = None


class RouterCapture:
    def __init__(self, routers: dict[int, torch.nn.Module], top_k: int):
        self.records = {layer: RouteRecord() for layer in routers}
        self.top_k = top_k
        self._handles = []
        for layer, router in routers.items():
            self._handles.append(router.register_forward_pre_hook(self._pre_hook(layer)))
            self._handles.append(router.register_forward_hook(self._post_hook(layer)))

    def _pre_hook(self, layer: int):
        def hook(_module, args):
            self.records[layer].router_input = args[0]
        return hook

    def _post_hook(self, layer: int):
        def hook(_module, _args, output):
            logits = output[0] if isinstance(output, tuple) else output
            values, indices = logits.topk(self.top_k, dim=-1)
            rec = self.records[layer]
            rec.logits, rec.topk_values, rec.topk_indices = logits, values, indices
        return hook

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
```

这里故意不 `detach()`：构建 DRPI 时，后续 margin 要对 injection state 求梯度。仅在 baseline capture 或落盘时再 `detach().cpu()`。

### 6.2 注入器

干预要挂在 router 的 `forward_pre_hook` 上，且该 hook 必须在 capture hook 之前或通过统一 wrapper 控制顺序。不要在 router output 上改 logits 后又宣称修改了 shared state。

```python
# src/drpi/interventions.py
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Injection:
    layer: int
    token_index: int             # -1 means final non-padding token; resolve before hook
    direction: torch.Tensor      # [hidden]
    alpha: float


class RouterInputInjector:
    def __init__(self, module: torch.nn.Module, injection: Injection):
        self.injection = injection
        self.handle = module.register_forward_pre_hook(self._hook)

    def _hook(self, _module, args):
        x = args[0]
        if x.ndim != 3:
            raise ValueError(f"expected [batch, seq, hidden], got {tuple(x.shape)}")
        y = x.clone()  # never mutate a tensor possibly reused by residual branches
        pos = self.injection.token_index
        d = self.injection.direction.to(device=x.device, dtype=x.dtype)
        y[:, pos, :] = y[:, pos, :] + self.injection.alpha * d
        return (y, *args[1:])

    def remove(self):
        self.handle.remove()
```

对于 padding batch，先由 attention mask 计算每个样本的最后有效位置；初版可以限定 `batch_size=1`，消除位置广播和 cache 的复杂度。

---

## 7. 关键实现：margin 梯度与 DRPI 方向

### 7.1 选一条真正会导致切换的边界

对每个下游 layer、token：

```python
selected = baseline_topk_indices[layer][0, token]    # top-k experts
outside  = highest_logit_expert_not_in(selected)
inside   = selected[-1]                               # weakest selected expert
margin   = logits[0, token, inside] - logits[0, token, outside]
```

这个 margin 越小，越接近专家集合切换。第一版每个 token/layer 只保留这一条边界，避免对所有 expert pair 求导。

### 7.2 把 injection state 留给 autograd

实现上最可靠的方式是：在 injection-layer router 的 pre-hook 中保存 `x`，并调用 `x.retain_grad()`；先运行一次无干预 forward，再从某个下游 router 的 logits 选定 margin，对保存的 `x` 求梯度。

```python
# src/drpi/gradients.py
import torch


def margin_gradient(
    injection_state: torch.Tensor,
    downstream_logits: torch.Tensor,
    token: int,
    inside_expert: int,
    outside_expert: int,
) -> torch.Tensor:
    """Gradient of one downstream expert boundary w.r.t. injection router input.

    Returns [hidden] for batch=1 and the selected injection token.
    """
    margin = downstream_logits[0, token, inside_expert] - downstream_logits[0, token, outside_expert]
    grad, = torch.autograd.grad(margin, injection_state, retain_graph=True)
    return grad[0, token].detach()


@torch.no_grad()
def dangerous_basis(projected_grads: torch.Tensor, rank: int) -> torch.Tensor:
    """projected_grads: [num_boundaries, blind_dim], output Q: [blind_dim, rank]."""
    _, singular_values, vh = torch.linalg.svd(projected_grads.float(), full_matrices=False)
    usable = min(rank, int((singular_values > 1e-8).sum().item()))
    return vh[:usable].T.contiguous().to(projected_grads.device)


@torch.no_grad()
def drpi_direction(target_direction: torch.Tensor, blind_basis: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Project d to current blind space, then remove downstream-dangerous components."""
    u = target_direction @ blind_basis          # [blind_dim]
    if q.numel():
        u = u - (u @ q) @ q.T
    return u @ blind_basis.T                     # [hidden]
```

注意两点：

1. `downstream_logits` 必须来自与 `injection_state` 同一次、未 `torch.no_grad()` 的 forward；
2. 为保持当前层盲性，`dangerous_basis` 在 blind coordinates 中做 SVD，而不是直接在 hidden space 中删方向。

### 7.3 构建方向的流程

```python
def build_drpi_for_layer(adapter, calibration_batches, injection_layer, horizon, rank, target_d):
    B = blind_basis(adapter.router_weight(injection_layer))
    all_projected_grads = []

    for batch in calibration_batches:
        capture = adapter.capture_routes(require_grad=True)
        adapter.forward_teacher_forced(batch)
        injection_x = capture.records[injection_layer].router_input

        for downstream_layer in range(injection_layer + 1,
                                      min(adapter.num_layers(), injection_layer + 1 + horizon)):
            logits = capture.records[downstream_layer].logits
            for token in adapter.sample_valid_tokens(batch, limit=4):
                inside, outside = weakest_inside_and_best_outside(logits[0, token], adapter.top_k)
                g = margin_gradient(injection_x, logits, token, inside, outside)
                all_projected_grads.append(g @ B)
        capture.remove()

    G = torch.stack(all_projected_grads)  # [M, blind_dim]
    Q = dangerous_basis(G, rank)
    return drpi_direction(target_d, B, Q), {"blind_basis": B, "dangerous_basis": Q}
```

这段是实现蓝图。真实代码要处理：每 batch 重新 forward、显存释放、随机 token 固定种子、以及所采样下游边界的元数据落盘。不要在同一保留计算图上连续处理很多 batch。

---

## 8. 四个必须先过的单元测试

### T1：router-null 正确性

随机 `d` 投影后，在指定层/位置比较 baseline 与注入后的 logits：

```python
assert torch.allclose(base_logits, edited_logits, rtol=1e-4, atol=2e-5)
assert torch.equal(base_topk, edited_topk)
```

不通过时，优先检查 hook 点是否在 RMSNorm 前、router 是否包含额外变换、以及预 hook 是否影响了同一张量的其他分支。

### T2：route capture 可信性

捕获的 top-k 必须和模型公开输出的 router logits（若提供）或手动 `router(router_input).topk()` 一致。先在一个 batch、一个 layer 上做逐元素验证。

### T3：注入的局部性

`alpha=0` 必须逐 token 完全复现 baseline logits；注入某个 token 时，其余 token 在**当前 router** 的 logits 不应改变。若改变，说明模型在 router 前进行了跨 token 混合，hook 点错了，或 capture 时机不对。

### T4：梯度的有限差分检查

对一个下游 margin，随机取 blind-space 小方向 `v`，验证：

```text
(margin(x + eps*v) - margin(x)) / eps  ~=  grad_margin(x) dot v
```

只需少量样本。没有这项检查，不应解释 Jacobian 预测失败还是代码错误。

---

## 9. 首轮实验的精确清单

### A. 静态 blind 泄漏曲线

目的：回答“局部不变性是否会在下一层及以后失效”。

```text
model:            OLMoE 小模型
prompts:          256 个固定 English prompts，长度 32–128 tokens
layers:           6–8 个均匀采样的 injection layers
positions:        每 prompt 最后一个非 padding token
directions:       8 个固定随机 Gaussian directions，经 blind 投影并单位化
strengths:        4 个 alpha；按 hidden-state RMS 比例定义，而非绝对数值
mode:             teacher forcing, batch_size=1 for first pass
outputs:          每个未来层的 top-k exact match、Jaccard、JS、min margin
```

先画生存率对层距的曲线，不看生成文本。该图若没有明确动态变化，先不要开发 DRPI。

### B. 泄漏可预测性

目的：检查一阶方向是否有用。

```text
calibration:      128 prompts, fixed sampled downstream margins
test:             128 held-out prompts
predictor:        baseline margin + g dot delta
label:            chosen expert set changed / unchanged
report:           AUROC, Brier score, reliability curve
```

若接近随机，先缩小 `alpha`、缩短 horizon，检查有限差分；不是立刻扩展到更大模型。

### C. 一个反事实目标任务

目的：检查 DRPI 是否有“目标改变同时减缓路径漂移”的空间。

用 100–300 个模板化对：

```text
The capital of France is Paris.  -> The capital of France is Lyon.
Alice lives in Rome.             -> Alice lives in Madrid.
The box is red.                  -> The box is blue.
```

从 A/B 在注入位置的状态差均值提取 `target_d`。在 A 的固定输入上注入，并用下一 token 或指定续写中的目标 token log-prob 作为目标指标。比较：

- full `target_d`；
- static blind `P_blind(target_d)`；
- DRPI；
- 随机等维子空间；
- 只做目标梯度优化；
- 直接 router-logit bias（作为“改路由很容易”的对照）。

每个方法都扫描 `alpha`，随后按相同目标 log-prob 提升进行插值比较路径分歧，而不是在相同 `alpha` 下比较。

---

## 10. 运行顺序与落盘格式

建议先有以下四个可重复命令；参数名只是建议，核心是输入/输出边界明确。

```bash
python scripts/inspect_model.py --model allenai/OLMoE-1B-7B-0924
python scripts/leakage_curve.py --config configs/olmoe_pilot.yaml --out results/leakage.json
python scripts/build_drpi.py --config configs/olmoe_pilot.yaml --out artifacts/drpi_layer12.pt
python scripts/evaluate_interventions.py --config configs/olmoe_pilot.yaml \
  --direction artifacts/drpi_layer12.pt --out results/counterfactual.json
```

每个 `json` 结果至少保存：模型 revision、代码 commit、dtype、设备、router hook path、SVD rank/tolerance、prompt IDs、随机种子、injection layer/position、alpha、horizon、目标指标和所有路径指标。没有这些字段，跨日或跨模型的图无法追溯。

`artifacts/drpi_layer12.pt` 保存：

```python
{
  "direction": direction.cpu(),
  "injection_layer": 12,
  "router_weight_sha256": "...",
  "basis_rank": int(B.shape[1]),
  "dangerous_rank": int(Q.shape[1]),
  "horizon": 8,
  "calibration_prompt_ids": [...],
  "config": {...},
}
```

方向绝不能跨模型、跨 revision 或跨 hook 点复用。

---

## 11. 真实风险与处理顺序

| 现象 | 最可能原因 | 优先处理 |
|---|---|---|
| 注入层 router logits 变了 | hook 在 norm 前，或投影用错权重方向 | 停止所有下游实验，修 T1 |
| 当前层不变、下一层几乎全变 | 这是可能的科学结果；也可能 alpha 过大 | 先扫更小 alpha，再报告泄漏曲线 |
| DRPI 完全没有目标效果 | 目标方向大多位于下游敏感空间 | 测量被删掉的目标范数；这是 F3 结论 |
| DRPI 只在 calibration 上有效 | 梯度子空间过拟合 | 减少 rank，严格 train/validation/test 划分 |
| 一阶预测很差 | alpha 太大、非线性强、梯度 hook 错 | 先做有限差分和短 horizon |
| 显存爆炸 | 同时保留太多下游 router 图 | 每次 forward 只收集少量边界后反向，或逐层重跑 |
| 生成文本变化但路径无法对齐 | 正常现象 | 仅报告生成质量与任务效果，不把它当路径保持证据 |

---

## 12. 写作时的结果决策树

```text
静态 blind 是否出现下游路由泄漏？
├─ 否：论文主结论是局部不变性在特定条件下稳定；DRPI 不应作为主方法。
└─ 是：margin/Jacobian 能预测泄漏吗？
   ├─ 否：论文转为非线性或结构性泄漏分析；不要声称 DRPI 有理论根据。
   └─ 是：DRPI 在效果匹配时降低路径漂移吗？
      ├─ 否：报告预测成立但控制不足，避免包装为新方法。
      └─ 是：路径保持能降低非目标输出漂移吗？
         ├─ 否：DRPI 是 route-control 工具，不应声称副作用优势。
         └─ 是：形成完整 ICLR 方法故事。
```

这棵树不是保守写法，而是让每一个实验结果都能决定下一步，不依赖事后挑选故事。

---

## 13. 最小交付目标

在写任何大规模评测前，代码应能产生以下四项可审阅产物：

1. 一个 `router_hook_report.json`：明确每层 router 的路径、输入形状、权重形状和即时盲性误差；
2. 一张静态 blind 泄漏曲线；
3. 一个有限差分通过的 margin-gradient 测试报告；
4. 一张 static blind 与 DRPI 在效果匹配下的路径分歧图。

拿到这四项后，才知道这是否是可写的机制论文，还是值得扩大为方法论文。
