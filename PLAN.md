# Mechanist — Master Plan

> **An open-source platform for understanding how AI systems actually work — from the tensor
> to the transistor.** Every concept is taught through real numbers you can play with, the
> math that explains them, the code that implements them, the silicon that runs them, the
> papers that invented them, and the ways they break.

*Working name. Alternatives: **Anatomy**, **Glassbox**, **Teardown**, **Lucida**. Needs a
GitHub-org / npm / trademark check before it is fixed.*

---

## The one-paragraph pitch

ML engineers can name concepts and call libraries, but cannot predict what happens inside.
Blogs give intuition without numbers, papers give deltas without mechanism, and framework docs
give code without reasons — and almost nothing explains what the GPU is actually doing.
Mechanist closes that gap by making every concept a **live, executable object**: a real tensor
flowing through a real operation, with its cost in FLOPs and bytes, its derivation, its kernel,
its lineage, and its failure modes — all in one page, all verified by CI, all free and
community-owned.

## The documents

| # | Document | What it settles |
|---|---|---|
| 01 | [Vision & Pedagogy](docs/01-vision-and-pedagogy.md) | The problem, the six-lens model, the ten pedagogical commitments, personas, non-goals |
| 02 | [Curriculum Atlas](docs/02-curriculum-atlas.md) | All 14 tracks, ~420 concepts, every signature lab, learning paths |
| 03 | [Technical Architecture](docs/03-architecture.md) | Three-tier execution model, stack decisions with rationale, cost engine, correctness system |
| 04 | [Content Schema & Lab Kit](docs/04-content-schema-and-lab-kit.md) | The concept folder, `concept.yaml`, lens templates, lab SDK, assessments, paper entries |
| 05 | [Roadmap, Risks & Success](docs/05-roadmap.md) | Five phases with exit criteria, risk table, metrics that measure understanding |
| 06 | [Community & Governance](docs/06-community-and-governance.md) | Contribution ladder, review rubric, governance, sustainability, adoption |

---

## The ten decisions that define this project

1. **Real numbers, never hand-drawn animations.** Every visualisation runs actual tensors —
   live in the browser (WebGPU/WASM) or replayed from verified PyTorch traces.
2. **Six lenses per concept**: Mechanism · Math · Code · **Systems** · Lineage · **Failure**.
   The last two are what everyone else omits, and they are mandatory here.
3. **Silicon is a main track, not an appendix.** GPU/TPU architecture, CUDA, Triton, CUTLASS,
   Metal/MLX, torch.compile, TensorRT, vLLM — with simulators and a real-GPU playground.
4. **One shared cost engine.** FLOPs, bytes, arithmetic intensity, memory, latency and roofline
   come from a single model, so a number cannot be right on one page and wrong on another.
5. **Three-tier execution.** Precomputed traces (always works) → in-browser compute (interactive)
   → optional GPU sandbox (proof). No concept ever *requires* the expensive tier.
6. **Verified claims or labelled estimates — never blurred.** Golden numerical tests against
   PyTorch in CI; every numeric claim carries `derived | measured | cited` provenance and a badge.
7. **Break it to understand it.** Every major concept has a sabotage mode with its characteristic
   failure signature — the most transferable debugging skill in the field.
8. **Papers as a graph, not a list.** Walk forward and backward through lineage; each entry states
   the one idea, the delta, what held up and what did not.
9. **Assessment by prediction, not recall.** Predict the tensor, find the bug, diagnose the curve,
   derive the cost, choose and defend.
10. **A concept folder is the unit of contribution.** Drop in a directory; the build discovers,
    validates, graphs, routes and indexes it. Code Apache-2.0, content CC BY-SA 4.0, free forever.

## The flagship artefacts (what makes people share it)

| Artefact | Track | Why it lands |
|---|---|---|
| 🔥 **Attention Microscope** | Transformer | One token, one head, real matrices: Q·K, mask, softmax, weighted V |
| 🔥 **Forward Pass Theatre** | Transformer | Scrub a timeline through all 12 layers of GPT-2 and watch a prediction form |
| 🔥 **Matmul Ladder** | Accelerators | 8 kernels, naive → tensor cores, with measured timings and roofline at every rung |
| 🔥 **Roofline Lab** | Accelerators | Plot any op on any accelerator; the most reused tool on the site |
| 🔥 **Flash Attention Tiler** | Accelerators | Tiles streaming through SRAM, running max and sum updating live |
| 🔥 **Triton Playground** | Accelerators | Write a kernel in the browser, compile on a real GPU, diff vs PyTorch, see your profile |
| 🔥 **Parallelism Simulator** | Training | Drag a 5D split; get per-GPU memory, pipeline bubble %, comm volume, projected MFU |
| 🔥 **Serving Simulator** | Inference | Discrete-event sim: arrival rates → TTFT/TPOT percentiles per scheduling policy |
| 🔥 **MoE Router Scope** | Architectures | Watch tokens route live; kill the aux loss and watch the router collapse |
| 🔥 **Diffusion Steppers** | Generative | Noise your own image, then reverse it watching the predicted x₀ sharpen |
| 🔥 **Circuit Tracer** | Interpretability | Patch activations and watch the causal path light up in a real model |

## Phases at a glance

| Phase | Weeks | Deliverable | Exit criterion |
|---|---|---|---|
| **0 — Vertical slice** | 1–3 | Engine + 3 perfect concepts (attention, tiled matmul, KV cache) | 4 of 5 test engineers report a genuine insight |
| **1 — Transformer & launch** | 4–12 | ~65 concepts, paper atlas, public launch | 5+ external contributors merge unaided |
| **2 — Silicon** | 13–24 | GPU track, Triton playground, quantisation, flash attention | Matmul Ladder becomes a linked reference |
| **3 — Breadth** | 25–44 | Modern architectures, training at scale, serving, diffusion, vision/VLM, post-training | 250 concepts at `stable` |
| **4 — Depth & durability** | 45+ | Retrieval, production, interpretability, assessments, i18n, offline | >90% of claims CI-verified |

**Phase 0 is three concepts, not three hundred.** The entire strategy rests on proving the
format is extraordinary before scaling it — and then letting the community scale it.

## Immediate next steps

1. Lock the name and claim the GitHub org / npm scope / domain.
2. Scaffold the monorepo (`apps/web`, `packages/lab-kit`, `packages/engine-cost`,
   `python/mechanist_trace`) with CI and the golden-test harness from commit one.
3. Build `engine-cost` v0 and the `.mtr` trace format — everything else depends on them.
4. Ship **Scaled dot-product attention** end to end, all six lenses, with the Attention
   Microscope. That one page is the proof of the whole thesis.
5. Show it to five working ML engineers. Fix the format based on what they *didn't* get.
6. Only then: Phase 1.
