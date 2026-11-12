# 01 — Vision & Pedagogy

## The problem we are attacking

Most ML engineers can *name* a concept and *use* a library that implements it. Far fewer can answer:

- **What** exactly happens to the numbers, step by step, with shapes?
- **Why** does this exist — what failure of the previous design forced it?
- **How** is it actually computed on silicon, and what does it cost in FLOPs, bytes and milliseconds?
- **When** does it win, and when does it break?

The gap is not a shortage of material. It is that existing material is split across three
incompatible worlds that never meet:

| World | Gives you | Never gives you |
|---|---|---|
| Blog posts / courses | Intuition, pretty animations | Real numbers, systems cost, honesty about tradeoffs |
| Papers | The delta and the claim | The mechanism at tensor level, the engineering reality |
| Framework docs / kernels | Working code | Why the design is that shape, what it replaced |

**Mechanist's thesis: a concept is only understood when you can predict its behaviour.**
Not recall it — *predict* it. Predict the output. Predict the shape. Predict which way the
curve bends when you double the sequence length. Predict whether you are compute-bound or
memory-bound. Predict what breaks at 70B.

Everything in this platform exists to build and then *test* that predictive ability.

## Core pedagogical commitments

### 1. Real numbers, never hand-drawn animations
Every visualisation is driven by actual numerical execution — a real tensor, from a real
model, through a real operation. Either computed live in the browser (WebGPU / WASM) or
replayed from a verified trace captured from PyTorch. **No fake arrows.** If a heatmap shows
attention, it is GPT-2's actual attention for the token you typed.

### 2. Six lenses on every concept
A concept page is not an article. It is six linked views of one mechanism:

| Lens | Question it answers | Form |
|---|---|---|
| **Mechanism** | What happens to the numbers? | Steppable visual execution on real tensors |
| **Math** | Why is it that formula? | Derivation where every symbol is bound to a shape |
| **Code** | How do I write it? | From-scratch (<300 lines) → production → kernel |
| **Systems** | What does it cost? | FLOPs, bytes, arithmetic intensity, roofline, memory, latency |
| **Lineage** | Why does it exist? | Papers as a genealogy: what each one fixed, and what it got wrong |
| **Failure** | When does it break? | Reproducible breakage you can trigger and watch |

A concept without the **Systems** and **Failure** lens is incomplete and will not ship.
That rule alone separates this from every other learning site.

### 3. Build it, then break it
Understanding is strongest at the boundary. Every major concept has a **sabotage mode**:
remove the residual connection, remove the `max` subtraction in softmax, set the LR 10×
too high, misalign the memory access, turn off the KV cache, drop the aux loss on an MoE
router. Watch the specific, characteristic way it fails. Failure signatures are the most
transferable debugging skill in the field and essentially nobody teaches them.

### 4. Numbers you must know, derived not memorised
A senior engineer carries a cost model in their head. We make that cost model an explicit,
interactive, parameterised object: pick `d_model`, `n_heads`, `n_layers`, `seq_len`, `batch`,
`dtype`, and accelerator — and read off FLOPs, HBM traffic, arithmetic intensity, KV cache
size, activation memory, optimizer state, step time, and whether you are compute- or
memory-bound. The same engine drives training, inference and kernel pages.

### 5. Silicon is not an appendix — it is a main track
The majority of "I know the concept but not how it works" pain sits below the framework
line: memory hierarchy, coalescing, occupancy, tensor cores, tiling, fusion, collectives.
We treat GPU/TPU architecture and low-level programming (CUDA, Triton, CUTLASS, Metal/MLX,
Pallas, torch.compile, TensorRT, vLLM) as a first-class track with its own simulators and a
real-GPU playground, not a footnote.

### 6. Papers as a connected graph, not a reading list
Every paper is attached to the concepts it introduced, modified or killed. You can walk
*forward* (what did Flash Attention enable?) and *backward* (why did anyone need it?).
Each paper entry carries: the one idea, the delta vs. its predecessor, what held up, what
did not, and the reproduction gotchas.

### 7. Industry reality is content, not flavour text
Design decisions from real systems — why Llama 3 chose GQA, why DeepSeek-V3 chose MLA +
fine-grained MoE, why SD3 moved to rectified flow, why vLLM paged the KV cache — are studied
as *engineering decision documents*, with the constraint that forced each choice.

### 8. Assessment that cannot be passed by memory
No "which of these is a transformer?" quizzes. Instead:
- **Predict the tensor** — given inputs and an op, predict shape and rough values.
- **Find the bug** — a kernel or implementation with one real defect.
- **Diagnose the curve** — "loss did this at step 14k; here are 4 metrics; what happened?"
- **Derive it** — produce the gradient / the cost / the memory bound.
- **Choose and defend** — a deployment scenario, pick the architecture, justify with numbers.

Spaced repetition is applied to *mechanisms and numbers*, not to vocabulary.

### 9. Progressive depth, honest about prerequisites
Three depths per concept, explicit and switchable: **Intuition** (10 min) → **Mechanism**
(45 min, the default) → **Research depth** (derivations, edge cases, open questions). The
knowledge graph means "you are missing X" is a link, not a wall of text.

### 10. Accessible, offline-capable, free, forever
Static-first so it is fast on a bad connection, works on a phone, self-hostable in one
command, and the content is permanently licensed open. A student in any country with a
laptop gets the same thing an engineer at a frontier lab gets.

## Who it is for

| Persona | Entry point | What they want |
|---|---|---|
| **The capable user** (uses PyTorch/HF daily, hazy below) | Transformer atom-by-atom | Mechanism, finally |
| **The systems-curious** (wants to go below the framework) | CUDA ladder, roofline lab | Silicon mental model |
| **The researcher** (knows one area deeply) | Paper atlas, lineage views | Breadth, fast, without noise |
| **The interviewee** | Numbers-you-must-know, diagnosis puzzles | Confidence under questioning |
| **The teacher** | Embeddable labs | Material they can point students at |
| **The contributor** | Lab kit + concept template | A fast path from knowing a thing to publishing it |

## Non-goals (so scope stays finite)

- Not a replacement for first courses in linear algebra or Python.
- Not a hosted training platform or a model zoo.
- Not a paper summariser — summaries without mechanism are what we are correcting.
- Not a credential/certification business.
- Not a video platform. Interactive > video, and video is unmaintainable by a community.
