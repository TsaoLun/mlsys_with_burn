# Burn 0.21 之后 PR 方向分析（面向 0.22 与后续）

- 分析日期：2026-08-10
- 上游仓库：https://github.com/tracel-ai/burn
- 本项目 pin：`pins.toml` → Burn `0.22.0-pre.1` @ `976aa9c5…`
- 方法：以官方 0.21 发布说明中的 “What’s Next” 为显式目标；以
  `v0.21.0`（2026-05-07）之后的 merged/open PR、0.22.0-pre.1 release notes、
  burn-book 同步 PR 为证据。**不是**官方已冻结的里程碑文档（仓库无
  GitHub Milestone）。

## 时间线锚点

| 标记 | 日期 | 含义 |
|------|------|------|
| `v0.21.0` | 2026-05-07 | 稳定版；本分析的“之后”起点 |
| `v0.22.0-pre.1` | 2026-07-29 | 预发布；本书当前写作线 |
| main（分析日） | ~2026-08-10 | pre.1 之后又合入约 50+ PR |

自 0.21 起约有 **250+** 个 merged PR；分析日仍有约 **22** 个 open PR。

## 官方显式目标（0.21 → “next”）

来源：[Burn 0.21.0 release post — What’s Next](https://burn.dev/blog/release-0.21.0)

1. **用户 API 去掉 `Backend` 泛型**  
   `burn-dispatch` 已是基础；下一版本要把 Device/dispatch 推到用户面，使
   model / optimizer / training 代码不再处处写 `B: Backend`。
2. **CubeCL、CubeK、Burn 更紧的一体化**  
   自定义 Kernel、复用 CubeK 构件应成为一等公民，而不是多 crate 拼装。

这两条是后续 PR 集群的“北极星”；下面各主题是对它们的实现路径与并行产品能力。

## 0.22 线已落地的主目标（按主题）

证据等级：**S** = 发布说明/维护者 PR 正文；**R** = 多 PR 集群可复现模式；
**I** = 结合 open PR / 设计解读（非官方承诺）。

### 1. Device 前台化与 Backend 泛型退场（S/R）

- 0.21 引入 `burn-dispatch` 与 Device 选择；0.22 线在 book 同步
  （[#5276](https://github.com/tracel-ai/burn/pull/5276)）明确写到
  “Removed backend generics”、重写 Backend & Device / Extension 文档。
- Extension 能力扩展：async/futures、primitive 互操作
  （[#5049](https://github.com/tracel-ai/burn/pull/5049)）；struct/enum
  输入（[#5221](https://github.com/tracel-ai/burn/pull/5221)）。
- 编译体验：into-scalar 等改动继续压增量编译成本
  （[#5037](https://github.com/tracel-ai/burn/pull/5037)、[#5061](https://github.com/tracel-ai/burn/pull/5061)），
  与 0.21 文中“实验性 <1s 增量编译”方向一致。

**意图**：训练/推理代码以 Device 为入口；后端可替换、可扩展，且编译反馈接近脚本语言。

### 2. Fusion / 图捕获 / 可扩展编译栈（S/R）

- Graph capture 与安全 API（[#5146](https://github.com/tracel-ai/burn/pull/5146)、
  [#5148](https://github.com/tracel-ai/burn/pull/5148)）。
- Fusion DAG / Cat 融合 / 别名与计划修复（[#5133](https://github.com/tracel-ai/burn/pull/5133)–[#5138](https://github.com/tracel-ai/burn/pull/5138)）。
- **Custom fusion optimization**：用户可注册与内置同级的 `FusedOperation`
  （[#5240](https://github.com/tracel-ai/burn/pull/5240)）——直接响应“自定义 Kernel
  一等公民”。
- CubeCL memory / persistent memory 与 fusion 协同
  （[#5158](https://github.com/tracel-ai/burn/pull/5158)）。
- Autotune 吞吐与 bounds、pooling/interpolate、ROCm/CPU 吞吐等持续合入。

**意图**：动态图 + JIT 融合仍是性能主干；把优化面从“框架内置”打开为可插拔。

### 3. Remote / 多设备执行面（R）

- 单 server 多设备、`Device::remote` / enumerate、同机免往返与跨设备
  all-reduce（[#5036](https://github.com/tracel-ai/burn/pull/5036)）。
- Client 侧 op-graph 缓存，稳定结构工作负载少传完整 op 序列
  （[#5088](https://github.com/tracel-ai/burn/pull/5088)）。
- Remote backend extension，目标含分布式数据加载
  （[#5101](https://github.com/tracel-ai/burn/pull/5101)）。
- Iroh 传输路径（[#5111](https://github.com/tracel-ai/burn/pull/5111)）。

**意图**：把 0.21 的分布式/collective 叙事延伸到“远程后端即执行面”，降低多机/多卡编排成本；与训练栈、fusion 缓存正交叠加。

### 4. 序列化与训练状态体系重做（S/R）

- burnpack 抽到 `burn-pack` + 最小 Record（[#5064](https://github.com/tracel-ai/burn/pull/5064)）。
- Record → store / `RecordState`；optimizer 独立为 `burn-optim`
  （[#5083](https://github.com/tracel-ai/burn/pull/5083)）——维护者明确：用新二进制格式与命名张量/标量状态替换旧 record。
- Param group、多 optimizer/多 LR、按组 freeze/quantize
  （[#5086](https://github.com/tracel-ai/burn/pull/5086)、[#5121](https://github.com/tracel-ai/burn/pull/5121)、[#5154](https://github.com/tracel-ai/burn/pull/5154)）。
- 自定义 checkpointer、checkpoint/metric 边界修复持续进行。

**意图**：checkpoint/PEFT/多组优化器可组合；格式与 crate 边界为长期部署与外部权重交换铺路。

### 5. PEFT 与参数重参数化抽象（R → 泛化）

- LoRA / QLoRA（[#5139](https://github.com/tracel-ai/burn/pull/5139)）。
- 再泛化为 `Reparameterization` / `Reparameterizer`（[#5311](https://github.com/tracel-ai/burn/pull/5311)，关闭 #2738）。

**意图**：微调不只是 LoRA 特例，而是可扩展的权重重参数化机制（WeightNorm 等）。

### 6. 量化栈加深（R，0.22 核心产品线之一）

已合入方向包括：

- 量化 op autodiff（[#5317](https://github.com/tracel-ai/burn/pull/5317)）。
- UE4M3 scale + 精度 harness（[#5253](https://github.com/tracel-ai/burn/pull/5253)）。
- BitNet 风格 Q2S 三元 matmul（[#5075](https://github.com/tracel-ai/burn/pull/5075)）。
- 子字节 reshape、量化 transpose、fusion 对不支持 scale 的拒绝等边界。

Open / 近未来：

- 双层量化（block + per-tensor，NVFP4 启发）
  （[#5262](https://github.com/tracel-ai/burn/pull/5262)，依赖 CubeCL/CubeK）。
- FP8 WIP（[#5096](https://github.com/tracel-ai/burn/pull/5096)）。

**意图**：训练与推理共享同一量化表示；向更窄格式、双层 scale、低比特训练/微调推进。

### 7. Tensor / NN 表面与 PyTorch 对齐（R）

大量“小而广”的合入：负轴维、mask_select、索引赋值、`extract`/`empty_like`、
大批 activation/loss、LSTM 状态 API、GQA/MQA flex attention、vision metric 等。

**意图**：降低迁移成本；在 Device/dispatch 新 API 下保持算子覆盖面。

### 8. Flex / 嵌入与 CPU 路径（R）

0.21 已用 Flex 取代 ndarray 作为轻量 eager CPU；0.22 继续：默认测试后端、
SIMD/rayon 测试开关、add_bias/interpolate 向量化、Xtensa 等 no_std 编译修复
（open [#5331](https://github.com/tracel-ai/burn/pull/5331)）。

**意图**：WASM/嵌入/无驱动环境仍是一等目标，与 GPU 加速栈并行。

## 未来版本信号（open PR 与上游耦合，I）

| 方向 | 信号 | 说明 |
|------|------|------|
| 编译器 IR 统一 | Burn [#5324](https://github.com/tracel-ai/burn/pull/5324) + CubeCL “Make everything Pliron” | 用 Pliron 替换自研 IR，利于长期优化与工具链 |
| 低比特/FP8 | [#5096](https://github.com/tracel-ai/burn/pull/5096)、[#5262](https://github.com/tracel-ai/burn/pull/5262) | 量化路线向现代 GPU 格式靠拢 |
| 设备侧稀疏选择 | [#5309](https://github.com/tracel-ai/burn/pull/5309) | argwhere/mask_select 流压缩，避免整 mask 回读 |
| 序列化瘦身 | [#5100](https://github.com/tracel-ai/burn/pull/5100) CBOR | 减依赖（WIP/stale，方向性） |
| Complex backend | [#3608](https://github.com/tracel-ai/burn/pull/3608) | 长期复数路径 |
| 可复现归约 | [#5156](https://github.com/tracel-ai/burn/pull/5156) | 确定性 float reduce |

这些**尚未**构成官方 0.23 路线图声明；但对本书跟踪 pins 时，应视为高概率演进面。

## 一张图：目标层级

```
显式（0.21 What’s Next）
├─ 用户面：Device / dispatch，去掉 Backend 泛型
└─ 栈一体：CubeCL ↔ CubeK ↔ Burn（自定义 Kernel/融合一等公民）
         │
         ▼
0.22 PR 集群（已合入或 deep WIP）
├─ Fusion 可扩展 + graph capture + autotune/内存
├─ Remote 多设备 + op-graph 缓存 + Iroh
├─ Record→store/burnpack + param groups + LoRA→Reparam
├─ 量化加深（autodiff / UE4M3 / BitNet；双层·FP8 在途）
└─ Tensor/NN 覆盖 + Flex CPU/嵌入
```

## 对本项目（mlsys_with_burn）的含义

1. **当前 pin（0.22.0-pre.1）已覆盖**：dispatch/Device 叙事、fusion 主干、
   burnpack/store 初版、remote 雏形、LoRA 初版；但 **落后于 main** 约一个月
   （reparam 泛化、custom fusion、量化 autodiff、大量负轴维与 NN 补齐等）。
2. **更新 pins 前必须先决策记录**（项目规则）：API 破环面主要在
   Record→store、Backend 泛型退场、Module/optimizer crate 边界、量化 dtype。
3. **章节敏感面**：
   - 第 2/4 章：Device/dispatch、fusion 可扩展、graph capture。
   - 第 6/9 章：remote 多设备与 collective 边界可能变宽。
   - 第 7 章：burnpack/`burn-pack`、store、量化部署。
   - 第 6/训练：param group、多 optimizer、LoRA/reparam。
4. **不要**把 open PR（Pliron、双层量化、FP8）写成“Burn 已支持”；应标为
   上游演进中。

## 关键 PR 索引（抽样）

| PR | 主题 |
|----|------|
| [#5036](https://github.com/tracel-ai/burn/pull/5036) | 多设备 remote |
| [#5049](https://github.com/tracel-ai/burn/pull/5049) | Extension async + primitive 互操作 |
| [#5064](https://github.com/tracel-ai/burn/pull/5064) / [#5083](https://github.com/tracel-ai/burn/pull/5083) | burnpack / Record 重构 |
| [#5088](https://github.com/tracel-ai/burn/pull/5088) | Remote fusion op-graph 缓存 |
| [#5101](https://github.com/tracel-ai/burn/pull/5101) | Remote backend extension |
| [#5111](https://github.com/tracel-ai/burn/pull/5111) | Iroh remote |
| [#5121](https://github.com/tracel-ai/burn/pull/5121) / [#5154](https://github.com/tracel-ai/burn/pull/5154) | 多 optimizer / 按组操作 |
| [#5139](https://github.com/tracel-ai/burn/pull/5139) / [#5311](https://github.com/tracel-ai/burn/pull/5311) | LoRA → 通用 reparam |
| [#5146](https://github.com/tracel-ai/burn/pull/5146) | Graph capture |
| [#5240](https://github.com/tracel-ai/burn/pull/5240) | Custom fusion optimization |
| [#5253](https://github.com/tracel-ai/burn/pull/5253) / [#5317](https://github.com/tracel-ai/burn/pull/5317) | 量化 scale / autodiff |
| [#5276](https://github.com/tracel-ai/burn/pull/5276) | Book：去掉 backend 泛型文档 |
| [#5324](https://github.com/tracel-ai/burn/pull/5324) | Migrate to pliron（open） |

## 验证记录

```text
gh api repos/tracel-ai/burn/releases/tags/v0.21.0
gh api repos/tracel-ai/burn/releases/tags/v0.22.0-pre.1
gh search prs --repo tracel-ai/burn --merged --merged-at='>2026-05-07'
curl/解析 https://burn.dev/blog/release-0.21.0 的 What’s Next / burn-dispatch 节
抽样 gh pr view（上表）
```

结论陈述区分：官方 “What’s Next” 原文 vs PR 集群归纳 vs open PR 前瞻。
