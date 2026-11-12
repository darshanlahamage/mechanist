# 02 — Curriculum Atlas

The complete concept surface. 14 tracks, ~420 concept nodes. Every node gets the six lenses
(Mechanism / Math / Code / Systems / Lineage / Failure) from `01-vision-and-pedagogy.md`.

Legend for the **Signature lab** column: the one interactive artefact that makes the concept
click. Labs marked 🔥 are flagship — they are the reason someone shares the site.

---

## Track 0 — Mathematical Mechanics
*Math taught as data movement and as cost, never as symbol pushing.*

| Concept cluster | Nodes | Signature lab |
|---|---|---|
| Linear algebra as computation | matmul as composition of maps; row/col views; rank & low-rank structure; SVD; projections; orthogonality; eigen/spectra; norms & conditioning; Jacobians | 🔥 **Matmul Explorer** — drag a matrix, watch rank collapse, see the same product computed 4 ways (dot/outer/row/block) with FLOP counts |
| Probability & information | distributions, expectation, variance; entropy; cross-entropy; KL & reverse-KL asymmetry; mutual information; MLE; Bayes; sampling (temperature, top-k, top-p, min-p, typical, beam) | **Sampler Playground** — one logit vector, every decoding strategy side by side, with the resulting distribution |
| Calculus & autodiff | chain rule; computational graphs; forward vs reverse mode; VJP/JVP; backprop derived from scratch; gradient checkpointing as a time/memory trade; higher-order grads | 🔥 **Autodiff Tracer** — type an expression, see the graph built, tape recorded, and reverse sweep executed with real values |
| Numerics | IEEE754; fp32/tf32/bf16/fp16/fp8(e4m3,e5m2)/fp6/fp4/int8/int4; range vs precision; catastrophic cancellation; why softmax subtracts max; Kahan & fp32 accumulation; stochastic rounding; determinism & reduction order | 🔥 **Float Inspector** — flip bits, see the value; run the same sum in 6 dtypes and watch divergence |
| Optimization theory | convexity; curvature & Hessian; condition number; gradient descent dynamics; momentum as a filter; stochasticity & noise scale; saddle points; loss landscapes; NTK sketch | **Landscape Walker** — pick a surface and an optimizer, watch the trajectory; same optimizer, 5 conditioning numbers |
| Statistics for ML | estimators, bias/variance, bootstrap, confidence intervals on evals, significance for benchmark deltas | **Benchmark Significance Tool** — is that +0.4% real? |

## Track 1 — Neural Network Core

| Concept cluster | Nodes | Signature lab |
|---|---|---|
| The MLP | neuron, layer, universal approximation (and its uselessness in practice), depth vs width, feature composition | **Neuron Microscope** — what each unit responds to, live |
| Initialisation | why variance matters; Xavier/Glorot; He; orthogonal; μP (maximal update parameterisation); residual-aware scaling; init's effect on signal propagation | 🔥 **Signal Propagation** — activation/grad variance per layer vs depth, swap init schemes, watch explosion/vanishing appear |
| Normalisation | BatchNorm (and its train/eval trap); LayerNorm; RMSNorm; GroupNorm; QK-norm; pre- vs post- vs sandwich norm; what norm actually fixes | **Norm A/B** — train a 12-layer net, toggle norm placement, compare grad norms per layer |
| Activations & gating | sigmoid/tanh → ReLU → LeakyReLU/ELU → GELU → SiLU → SwiGLU/GeGLU; why gating wins; dead units | **Activation Atlas** — function, derivative, and effect on a real FFN block |
| Residual stream | skip connections; identity path; residual as a shared bus; why depth became possible; stream norm growth | 🔥 **Residual Stream Scope** — per-layer contribution to the stream in a real GPT-2 |
| Regularisation | L2/weight decay (and decoupling); dropout; label smoothing; stochastic depth; data augmentation; early stopping; z-loss | **Overfit Lab** — tiny dataset, toggle each, see the gap |
| Training loop anatomy | forward/backward/step; loss curves; LR; batch; epochs vs tokens; metrics that lie | **Loop Dissector** — single step with every intermediate exposed |

## Track 2 — Sequence Modelling Before Transformers
*Short track. Exists to make the transformer feel inevitable rather than arbitrary.*

n-grams & perplexity · RNN unrolling · vanishing/exploding gradients shown numerically ·
LSTM gates · GRU · seq2seq bottleneck · Bahdanau/Luong attention as the fix · the
parallelism wall that killed RNNs.
**Signature lab:** 🔥 **The Bottleneck** — same task with RNN vs attention; watch the fixed-size
context vector lose information as sequence length grows, measured not asserted.

## Track 3 — The Transformer, Atom by Atom
*The flagship track. Every single tensor in a real GPT-2 forward pass is inspectable.*

| Concept cluster | Nodes | Signature lab |
|---|---|---|
| Tokenisation | characters → words → BPE → byte-level BPE → WordPiece → Unigram/SentencePiece; vocab size tradeoffs; merge rules; tokenizer pathologies (numbers, code, whitespace, non-Latin scripts, glitch tokens); tokenizer-free & byte-latent approaches | 🔥 **Tokenizer Lab** — type anything, see merges happen step by step across 8 real tokenizers; token-cost heatmap by language |
| Embeddings | lookup as one-hot matmul; embedding geometry; tying input/output; vocab-parallel embeddings | **Embedding Space** — 3D projection of real embeddings, arithmetic, neighbourhoods |
| Attention core | Q/K/V projections; scaled dot-product and *why* √d; the softmax; attention as soft dictionary lookup; multi-head as independent subspaces; head concatenation & output projection; causal masking; padding masks | 🔥🔥 **Attention Microscope** — one token, one head: see Q·K scores, the mask, the softmax, the weighted V sum, as live matrices. Toggle heads, layers, prompts |
| Positional information | why permutation invariance is a problem; sinusoidal; learned absolute; relative (T5 bias); **RoPE** (full derivation, rotation visualised in 2D planes); RoPE scaling — PI, NTK-aware, YaRN, LongRoPE; ALiBi; NoPE; 2D/3D RoPE for vision and video | 🔥 **RoPE Visualiser** — watch query/key vectors rotate by position; extend context and see the extrapolation failure, then fix it with each scaling method |
| Attention variants | MHA → MQA → GQA → **MLA** (DeepSeek latent compression); sliding window; global+local hybrids; sparse/block-sparse; sink tokens & StreamingLLM; cross-attention | 🔥 **KV Budget Explorer** — same model, 5 attention schemes, KV cache bytes and quality tradeoff |
| Feed-forward block | up/down projection; expansion ratio; GeLU-MLP vs SwiGLU (and the 2/3 width correction); where the parameters actually live | **Parameter Census** — where every byte of a 7B model lives |
| Block assembly | pre-norm vs post-norm; residual placement; depth scaling; weight tying; final norm & LM head; logit soft-capping | **Block Builder** — assemble a block from parts, run it, break it |
| The full forward pass | end-to-end GPT-2 small with every intermediate tensor; logit lens per layer; next-token distribution formation | 🔥🔥 **Forward Pass Theatre** — scrub a timeline through all 12 layers and watch the prediction form |
| Autoregressive decoding | KV cache mechanics; prefill vs decode; cache growth; why decode is memory-bound; sampling loop | 🔥 **KV Cache Animator** — token by token, bytes allocated, FLOPs spent, time per token |
| Training the transformer | causal LM objective; teacher forcing; packing & document masking; loss masking; label shifting | **Objective Lab** — CLM vs MLM vs prefix-LM vs FIM on the same batch |

## Track 4 — Modern LLM Architectures

| Concept cluster | Nodes | Signature lab |
|---|---|---|
| Architecture families | GPT-2 → GPT-3 → Llama 1/2/3/4 → Mistral → Qwen 2/3 → Gemma → Phi → DeepSeek V2/V3/R1 → OLMo → open frontier models; what each changed and why | 🔥 **Architecture Differ** — pick two models, get a structural diff (norms, attention, PE, FFN, vocab, init) with the stated rationale from each tech report |
| Mixture of Experts | sparse FFN; top-k routing; router design; auxiliary load-balance loss; expert capacity & dropping; shared experts; fine-grained experts; expert parallelism; router collapse; MoE vs dense at equal FLOPs/params | 🔥🔥 **MoE Router Scope** — watch tokens routed live, per-expert load histogram, kill the aux loss and watch collapse happen |
| State-space & linear models | S4 intuition; HiPPO; **Mamba** selective scan; the recurrence ↔ convolution duality; Mamba-2 & SSD; parallel scan; linear attention; RWKV; RetNet; Gated DeltaNet; hybrid stacks (Jamba, Samba) | 🔥 **Scan Visualiser** — the same sequence processed as recurrence and as convolution, proven equal, with the cost curves that differ |
| Long context | the quadratic wall; ring attention; context parallelism; chunked attention; retrieval-augmented context; memory layers; context compression; the "lost in the middle" effect | 🔥 **Context Scaling Lab** — memory/time/accuracy as a function of context length, per method |
| Reasoning architectures | chain of thought; self-consistency; test-time compute scaling; RL-trained reasoning (R1-style); latent/looped reasoning; verifier-guided search; budget forcing | **Test-Time Compute Curve** — accuracy vs tokens spent, per strategy |
| Diffusion LLMs & alternatives | discrete diffusion for text; any-order AR; energy-based & hybrid decoders | **Parallel Decode Viewer** |
| Agent & tool architectures | tool-call formats; function schemas; structured output & constrained decoding (grammars, FSM, logit masking); planning loops; memory; multi-agent topologies; context engineering | 🔥 **Constrained Decoding Lab** — watch an FSM mask logits token by token to guarantee valid JSON |

## Track 5 — Training at Scale

| Concept cluster | Nodes | Signature lab |
|---|---|---|
| Data | sourcing; dedup (exact, MinHash, semantic); quality filters & classifiers; mixtures and their effect; curriculum & annealing; synthetic data; contamination; tokenizer training; packing & shuffling; streaming loaders | 🔥 **Data Mixture Simulator** — move the mixture sliders, see projected capability deltas from published ablations |
| Objectives | CLM; MLM; prefix-LM; FIM; UL2; span corruption; multi-token prediction; distillation losses | **Objective Comparator** |
| Optimizers | SGD → momentum → Nesterov → Adagrad → RMSProp → Adam → AdamW → Adafactor → Lion → Shampoo/SOAP → **Muon**; update rules visualised; optimizer memory cost; β/ε sensitivity; weight decay interaction | 🔥 **Optimizer Dissector** — one loss surface, all optimizers, per-step update vectors drawn, plus their state memory in GB for a 7B model |
| Schedules & hyperparameters | warmup; cosine; WSD (warmup-stable-decay); linear; inverse-sqrt; LR vs batch scaling; critical batch size; gradient accumulation; gradient clipping; μP hyperparameter transfer | 🔥 **Schedule Lab** — train a tiny model live under any schedule, watch the loss respond |
| Scaling laws | Kaplan vs Chinchilla; compute-optimal vs inference-optimal; data-constrained scaling; emergent ability debate; isoFLOP curves; extrapolation risk | 🔥🔥 **Scaling Law Calculator** — budget in GPU-hours → optimal (N, D), predicted loss, cost in dollars, and what a different objective buys |
| Stability | loss spikes & recovery; z-loss; QK-norm; embedding norm growth; attention entropy collapse; bf16 vs fp16 failure modes; skipped batches; divergence forensics | 🔥 **Spike Forensics** — real spike traces; you diagnose, then see the postmortem |
| Parallelism | data parallel; ZeRO stages 1/2/3; FSDP; tensor parallel (row/column split derivations); sequence parallel; **pipeline parallel** (GPipe, 1F1B, interleaved, zero-bubble); expert parallel; context parallel; 5D composition; device mesh | 🔥🔥 **Parallelism Simulator** — set a model + cluster, drag a 5D split, see per-GPU memory, bubble %, comm volume, and projected MFU |
| Memory | parameters + grads + optimizer states + activations arithmetic; activation checkpointing & selective recompute; CPU/NVMe offload; memory fragmentation; the OOM decision tree | 🔥 **Memory Budget Calculator** — exact bytes, every term labelled |
| Precision | mixed precision; loss scaling; bf16 vs fp16; fp8 training (scaling recipes, delayed scaling); master weights; fp32 accumulation points | **Precision Lab** — train the same model in 4 precisions, watch where it dies |
| Communication | NCCL collectives; ring vs tree allreduce; bandwidth & latency models; NVLink/NVSwitch/PCIe/InfiniBand topologies; gradient compression; overlap of comm & compute | 🔥 **Collective Visualiser** — animate allreduce/allgather/reduce-scatter across a topology with real byte counts and times |
| Infrastructure | checkpointing & resumption; elastic/fault-tolerant training; straggler & silent data corruption detection; job orchestration; MFU/HFU measurement; cost per training run | **MFU Estimator** |

## Track 6 — Post-Training & Alignment

| Concept cluster | Nodes | Signature lab |
|---|---|---|
| Supervised fine-tuning | instruction data design; chat templates (and template bugs); loss masking on prompts; multi-turn handling; catastrophic forgetting | 🔥 **Chat Template Inspector** — see exactly what the model is trained on, special tokens included |
| Parameter-efficient tuning | LoRA (rank, α, placement, what low rank can and cannot express); QLoRA & NF4; DoRA; rsLoRA; adapters; prefix/prompt tuning; IA³; merging adapters | 🔥 **LoRA Rank Explorer** — approximate a real weight delta at every rank, see reconstruction error vs memory |
| Reward modelling | preference data; Bradley–Terry; reward model training; reward hacking & over-optimisation; process vs outcome rewards; rubric & verifier rewards | 🔥🔥 **Reward Hacking Sandbox** — optimise against a proxy reward and watch true performance diverge |
| RL for LLMs | policy gradient → PPO (clipping, value head, KL penalty, GAE); REINFORCE/RLOO; **GRPO**; RLVR/verifiable rewards; rollout & advantage mechanics; KL control; entropy collapse; training/inference engine split | 🔥 **PPO vs GRPO Scope** — same prompts, see rollouts, advantages, and the update each algorithm applies |
| Direct preference methods | DPO derivation from the RLHF objective; IPO; KTO; ORPO; SimPO; implicit reward view; length bias | 🔥 **DPO Geometry** — the implicit reward surface, why length bias appears |
| Distillation & merging | logit/sequence distillation; on-policy distillation; task arithmetic; TIES; DARE; SLERP; model soup | **Merge Lab** — merge two real adapters, test both capabilities |
| Safety & robustness | refusal mechanisms as circuits; jailbreak classes; prompt injection; safety tuning cost; adversarial training; red-teaming method; guardrail models | **Injection Sandbox** — safe, sandboxed demonstrations of attack classes and defences |
| Evaluation | benchmark construction; contamination detection; LLM-as-judge (bias, position effects, calibration); pairwise/ELO; rubric grading; agentic evals; statistical power; eval-driven development | 🔥 **Judge Bias Lab** — same answers, permuted order, watch the judge flip |

## Track 7 — Vision

| Concept cluster | Nodes | Signature lab |
|---|---|---|
| Convolution | convolution as matmul (im2col); kernels, stride, padding, dilation; receptive field growth; pooling; depthwise/separable; 1×1 as channel mixing; translation equivariance | 🔥 **Conv Microscope** — draw on a canvas, pick a kernel, watch every sliding window and the equivalent matmul |
| CNN architectures | LeNet → AlexNet → VGG → Inception → **ResNet** → DenseNet → MobileNet/EfficientNet → ConvNeXt; what each fixed | **CNN Differ** |
| Vision Transformer | patchification; patch embedding; CLS token vs pooling; position embeddings in 2D; inductive bias vs data scale; DeiT distillation; Swin windows & shifts; hierarchical vs plain; registers | 🔥🔥 **ViT Dissector** — upload an image, watch patches become tokens, attention roll out per layer, attention-map overlays |
| Self-supervised vision | SimCLR; MoCo; BYOL; SwAV; **DINO/DINOv2/DINOv3**; MAE; iJEPA; what each learns, probed | 🔥 **Feature Probe** — same image, features from 6 pretrained backbones, compared on a live linear probe |
| Dense prediction | detection (R-CNN family → YOLO → **DETR** set prediction & Hungarian matching); segmentation (FCN, U-Net, Mask R-CNN, **SAM/SAM2**); depth (MiDaS, Depth Anything); pose; tracking | 🔥 **DETR Matching Visualiser** — watch queries bind to objects through the layers |
| Video & 3D | temporal modelling; video ViT; tubelets; NeRF; **3D Gaussian Splatting**; point clouds; world models | **Splat Explorer** |
| Image generation evaluation | FID/CLIPScore and what they miss; human eval design | **Metric Pitfalls** |

## Track 8 — Multimodal (VLM / Omni)

| Concept cluster | Nodes | Signature lab |
|---|---|---|
| Contrastive alignment | **CLIP** training mechanics; InfoNCE; batch size dependence; temperature; **SigLIP** sigmoid loss; modality gap; zero-shot transfer | 🔥 **Contrastive Space** — see the modality gap in a real joint embedding space; adjust temperature, watch it change |
| VLM connectors | linear projector; MLP projector; **Q-Former**; cross-attention (Flamingo-style); perceiver resampler; token count vs quality tradeoff; pooling & compression | 🔥🔥 **Connector Lab** — same image & LLM, swap the connector, see the visual tokens the LLM actually receives |
| Resolution handling | fixed-resolution ViT; AnyRes tiling; native dynamic resolution (NaViT/Qwen-VL); token budgets for high-res documents; aspect-ratio handling | 🔥 **Token Budget Explorer** — a 4K screenshot → how many visual tokens, under each scheme, at what cost |
| VLM training recipes | stage 1 alignment / stage 2 instruction / stage 3 high-res; freezing schedules; data mixtures; OCR & chart data; LLaVA → Qwen-VL → InternVL → Molmo → Pixtral lineage | **Recipe Differ** |
| Audio & speech | mel spectrograms; **Whisper** encoder-decoder; audio tokenisation (codec models); TTS architectures; streaming ASR; duplex/real-time voice | 🔥 **Spectrogram Lab** — speak, see the mel, see the tokens, see the decode |
| Omni & any-to-any | unified token spaces; interleaved training; image generation inside an LLM; modality routing | **Omni Token Stream** |
| Document AI | OCR-free understanding; layout; table extraction; chart reasoning failure modes | **Doc Failure Gallery** |

## Track 9 — Generative Modelling & Diffusion

| Concept cluster | Nodes | Signature lab |
|---|---|---|
| Latent variable models | autoencoders; **VAE** (ELBO derivation, reparameterisation trick, posterior collapse); VQ-VAE & codebooks; FSQ; normalizing flows | 🔥 **VAE Latent Walker** — walk the latent space, watch the reconstruction; break the KL term |
| GANs | minimax objective; discriminator/generator dynamics; mode collapse; WGAN & gradient penalty; StyleGAN's latent spaces; why GANs lost to diffusion (and where they still win) | 🔥 **GAN Training Theatre** — watch the two losses fight; induce mode collapse deliberately |
| Diffusion foundations | forward noising process; reverse process; **DDPM derivation**; score matching; the score ↔ noise ↔ ELBO equivalence; SDE/ODE formulation; noise schedules (linear, cosine, shifted); SNR & timestep weighting | 🔥🔥 **Diffusion Steppers** — scrub the forward process on your own image; then reverse it step by step watching the predicted x₀ sharpen |
| Sampling | DDIM; ancestral; Euler/Heun; DPM-Solver(++); UniPC; step count vs quality; stochasticity; **classifier-free guidance** (and guidance scale pathologies); guidance variants | 🔥 **Sampler Comparator** — identical seed & prompt, every sampler, every step count, in a grid |
| Architectures | **U-Net** (down/up blocks, skips, attention placement); **DiT** (transformer diffusion, adaLN-zero, patchification); MM-DiT (SD3); conditioning paths (text, class, time embedding) | 🔥 **DiT vs U-Net Dissector** |
| Latent & modern | latent diffusion (the VAE's role); SDXL; SD3; **Flux**; rectified flow & **flow matching** as the unifying view; v-prediction; EDM parameterisation | 🔥 **Flow vs Diffusion** — the same model family seen through both lenses, trajectories drawn |
| Control & adaptation | ControlNet; T2I-Adapter; IP-Adapter; LoRA for diffusion; textual inversion; DreamBooth; inpainting/outpainting masks; img2img strength | 🔥 **ControlNet Lab** — draw a condition, watch it steer each denoising step |
| Fast sampling | progressive distillation; consistency models; LCM; adversarial distillation (SDXL-Turbo); 1–4 step generation; the quality cliff | **Step-Count Cliff** |
| Video & beyond | temporal layers; 3D attention; video DiT; latent video; Sora-class design; motion consistency; audio generation | **Temporal Attention Scope** |
| Diffusion serving | VAE decode cost; batch structure; step caching; quantised diffusion; memory on consumer GPUs | **Diffusion Cost Model** |

## Track 10 — Embeddings, Retrieval & RAG

embedding geometry & anisotropy · pooling strategies · contrastive & hard-negative training ·
Matryoshka embeddings · similarity metrics · **ANN indexes** (flat, IVF, PQ/OPQ, **HNSW** graph
construction & search, DiskANN) · vector DB architecture · hybrid search & BM25 · chunking
strategies · query transformation · reranking (cross-encoders, late interaction/ColBERT) ·
RAG architectures & failure modes · long-context vs RAG economics · graph RAG · agentic retrieval · eval (recall@k, nDCG, faithfulness).
**Signature labs:** 🔥 **HNSW Walker** — watch the graph get built, then watch a query hop through layers.
🔥 **RAG Failure Gallery** — eight real failure modes, each reproduced live.

## Track 11 — Accelerator Architecture & Low-Level Programming
*The second flagship track. This is where the field's understanding gap is widest.*

| Concept cluster | Nodes | Signature lab |
|---|---|---|
| Hardware mental model | CPU vs GPU design philosophy; SM / warp / thread hierarchy; SIMT & divergence; register file; shared memory; L1/L2; HBM; latency & bandwidth numbers by generation; **tensor cores** and MMA shapes; async copy & TMA; warp specialisation; Hopper/Blackwell features; NVLink/NVSwitch; multi-GPU topology | 🔥🔥 **Memory Hierarchy Explorer** — a tensor's journey HBM → L2 → SMEM → registers with real latencies and sizes, per GPU model |
| Performance models | **roofline**; arithmetic intensity; compute vs memory bound; occupancy and why more is not always better; latency hiding; Little's law; achieved vs peak bandwidth; MFU | 🔥🔥 **Roofline Lab** — plot any op (your own parameters) on any accelerator; the single most reused tool in the platform |
| CUDA from zero | host/device model; kernel launch & indexing; memory coalescing; bank conflicts; shared-memory tiling; warp shuffles; reductions; atomics; streams & events; CUDA graphs; cooperative groups; unified memory; profiling with Nsight | 🔥🔥 **Coalescing Visualiser** — write an access pattern, watch the memory transactions issued, see the bandwidth achieved |
| The matmul ladder | naive → global-memory tiled → shared-memory tiled → register tiling → vectorised loads → double buffering → tensor cores → CUTLASS-level; measured speedup and roofline position at each rung | 🔥🔥🔥 **Matmul Ladder** — 8 kernels, real GPU timings, the occupancy/intensity story at every step. The platform's signature piece |
| Triton | programming model & block pointers; masking; autotuning; fused softmax; fused layernorm; **flash attention in Triton**; fused MoE; reading generated IR/PTX; when Triton loses to CUTLASS | 🔥🔥 **Triton Playground** — write a kernel in the browser, compile & profile on a real GPU, diff against a PyTorch reference, see your roofline point |
| Attention kernels | the IO-complexity argument; **online softmax derivation**; Flash Attention v1 → v2 (work partitioning) → v3 (async, fp8); backward pass; varlen & paged variants; block-sparse masks; FlexAttention | 🔥🔥 **Flash Attention Tiler** — watch tiles stream through SRAM, running max & sum update live, memory traffic counted against naive attention |
| Other kernel stacks | CUTLASS/CuTe layouts & tiles; ThunderKittens; composable kernel (ROCm/HIP); **Metal & Apple MLX** (unified memory model); Pallas for TPU; TVM/Hidet; Mojo | **Backend Differ** — the same kernel in Triton, CUDA, Metal, Pallas |
| Compilers & graph optimisation | torch.compile / Dynamo graph capture & guards; Inductor codegen; operator fusion rules; graph breaks (and how to find them); memory planning; CUDA graphs & launch overhead; XLA/HLO; ONNX; TensorRT builder & engine; autotuning | 🔥 **Fusion Inspector** — see the graph before/after fusion, with the kernel launches and the time saved |
| TPU & alternative silicon | **systolic array** mechanics; MXU; vector/scalar units; HBM & on-chip memory; pods & ICI topology; SPMD/GSPMD sharding annotations; XLA layout assignment; Trainium/Gaudi/Groq/Cerebras design points; NPUs on device | 🔥 **Systolic Array Animator** — watch weights and activations flow through the array, cycle by cycle |
| Quantisation | integer & float low-precision formats; symmetric/asymmetric; per-tensor/channel/group scaling; calibration; **PTQ** (GPTQ, AWQ, SmoothQuant, HQQ) vs **QAT**; activation outliers & why they appear; fp8/fp4 inference; KV-cache quantisation; dequant cost & fused kernels; quality evaluation done properly | 🔥🔥 **Quantisation Scope** — see a real weight distribution, place the grid, watch the error; find the outlier channels that break naive int8 |
| Sparsity & compression | magnitude/structured pruning; 2:4 semi-structured sparsity on tensor cores; lottery tickets; activation sparsity; low-rank compression; distillation for efficiency | **Sparsity Lab** |

## Track 12 — Inference & Serving Systems

| Concept cluster | Nodes | Signature lab |
|---|---|---|
| Decode economics | prefill vs decode asymmetry; why decode is bandwidth-bound; arithmetic intensity of a decode step; batch size as the only lever; TTFT/TPOT/throughput definitions; latency-throughput frontier | 🔥🔥 **Decode Arithmetic** — derive tokens/s from memory bandwidth, for any model/GPU, and check it against measurements |
| Batching & scheduling | static → dynamic → **continuous batching**; iteration-level scheduling; chunked prefill; priority & fairness; admission control; SLO-aware scheduling; preemption | 🔥🔥 **Serving Simulator** — a discrete-event simulator: set arrival rate, prompt/output length distributions, GPU count, policy; get TTFT/TPOT percentile curves and utilisation. Compare policies head to head |
| KV cache systems | **PagedAttention** block tables; fragmentation (internal/external); prefix/radix caching; cache eviction policies; cache offload to CPU/NVMe; cross-request sharing; KV compression | 🔥🔥 **Paged KV Visualiser** — blocks allocated/freed live under a request stream; turn paging off and watch fragmentation waste memory |
| Speculative decoding | draft-and-verify mathematics; acceptance rate; draft model choice; **Medusa** heads; **EAGLE**; lookahead; n-gram/prompt lookup; tree attention & verification; when speculation loses | 🔥 **Speculation Lab** — watch accept/reject per token; tune draft length; see the speedup curve turn over |
| Serving stacks | **vLLM** architecture; **SGLang** (RadixAttention, structured output); **TensorRT-LLM**; llama.cpp/GGUF; Ollama; LMDeploy; Triton Inference Server; routing & gateways; multi-LoRA serving | 🔥 **Stack Differ** — the same deployment expressed in 4 stacks, with where each wins |
| Distributed inference | tensor parallel for inference; pipeline for memory; **disaggregated prefill/decode**; expert parallel serving for MoE; KV transfer between nodes; replica autoscaling; cold starts | 🔥 **Disaggregation Lab** — same cluster, aggregated vs disaggregated, SLO attainment compared |
| On-device | quantised local inference; GGUF formats; Apple Silicon unified memory & **MLX**; mobile NPUs; browser inference (WebGPU/ONNX Runtime Web); memory-mapped weights; speculative decoding on-device | 🔥 **Local Model Fit** — your RAM/VRAM → which models fit, at what quant, at what speed |
| Economics & operations | cost per 1M tokens derived from first principles; GPU rental vs owned; utilisation & autoscaling; caching layers; observability (what to log, what to alert on); load testing; capacity planning | 🔥🔥 **Cost Model** — full $/token derivation you can argue with your finance team |

## Track 13 — Production ML Engineering & Case Studies

reproducibility & determinism (and the price of it) · experiment tracking · data & model versioning ·
config management · CI for models · regression evals & canaries · shadow deployment & A/B ·
drift detection · incident forensics · on-call for model systems · build-vs-buy · team topologies ·
compliance & model cards · carbon & energy accounting.

**Case studies** read as engineering decision documents, each mapped onto the concept graph:
Llama 3 · DeepSeek-V3 & R1 · Qwen 3 · OLMo 2 · Gemma · Mixtral · SD3 & Flux · Whisper ·
CLIP/SigLIP · vLLM · Flash Attention · Chinchilla · μTransfer · the GPT-4 system card.
**Signature lab:** 🔥 **Decision Replay** — you are given the constraints the team had (budget, hardware,
data, deadline). You choose. Then you see what they chose and why.

## Track 14 — Interpretability & the Science of Deep Learning

logit lens & tuned lens · probing classifiers · activation patching & causal tracing ·
circuits & induction heads · attention head taxonomy · superposition & polysemanticity ·
**sparse autoencoders** / dictionary learning · feature steering · grokking · double descent ·
lottery tickets · scaling-law mechanism · emergence vs metric artefacts · memorisation vs
generalisation · neural tangent kernel vs feature learning · representation convergence.

**Signature labs:** 🔥🔥 **Circuit Tracer** — pick a behaviour, patch activations, watch the causal
path light up in a real model. 🔥 **SAE Feature Browser** — real features, with steering you can apply
to a live generation. 🔥 **Grokking Live** — train to memorisation, keep going, watch generalisation snap in.

---

## Cross-cutting: "Numbers You Must Know"

A permanent, always-available panel (keyboard `⌘K → numbers`) backed by the same cost engine:

- FLOPs per token for training (6ND) and inference (2N), derived not asserted
- Memory: params + grads + optimizer + activations + KV, per dtype
- Arithmetic intensity of every core op
- Bandwidth, capacity and FLOPs of every common accelerator, with sources
- Collective communication volumes for each parallelism strategy
- Rules of thumb with their validity range and the condition that breaks them

## Cross-cutting: Learning Paths

Curated routes through the graph, each with a stated outcome and estimated hours:

| Path | Hours | Outcome |
|---|---|---|
| **Transformer from zero** | 25 | Implement a transformer from scratch; predict every shape |
| **I want to understand GPUs** | 40 | Write and optimise a Triton kernel; read a profile |
| **LLM inference engineer** | 35 | Size, deploy and cost a serving system; defend the choices |
| **Pretraining engineer** | 50 | Plan a training run: data, parallelism, stability, budget |
| **Diffusion deep dive** | 30 | Derive DDPM; build a sampler; design a conditioning path |
| **VLM builder** | 25 | Choose a connector and resolution strategy with evidence |
| **Interview preparation** | 20 | Mechanism recall + numbers + diagnosis under questioning |
| **Paper-to-production** | 15 | Read a new paper and locate it in the graph in 30 minutes |
