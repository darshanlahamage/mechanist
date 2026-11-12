import type { Concept, Track } from './types'
import { attention } from './concepts/attention'
import { softmaxNumerics } from './concepts/softmax'
import { rope } from './concepts/rope'
import { diffusionForward } from './concepts/diffusion'
import { decodeCost } from './concepts/decode'

/** A concept that is on the map but not written yet. Being honest about this
 *  is better than fake pages, and it doubles as the contributor to-do list. */
const soon = (track: string, id: string, title: string, tagline: string, difficulty: 1|2|3|4|5 = 3): Concept =>
  ({ id, title, tagline, track, difficulty, status: 'planned' })

export const TRACKS: Track[] = [
  {
    id: 'foundations',
    title: 'Foundations',
    blurb: 'The math and the number behaviour that everything else sits on. Taught as data movement and cost, not as symbols.',
    concepts: [
      softmaxNumerics,
      soon('foundations', 'matmul', 'Matrix multiply', 'Four different ways to see the same operation, and why the GPU cares which one you pick.', 1),
      soon('foundations', 'float-formats', 'Float formats', 'fp32, bf16, fp16, fp8, int8, int4 — what each one trades away.', 2),
      soon('foundations', 'autodiff', 'Automatic differentiation', 'How a graph of operations turns into gradients, one chain rule at a time.', 3),
      soon('foundations', 'entropy-kl', 'Entropy, cross entropy, KL', 'What a loss function is actually measuring.', 2),
      soon('foundations', 'optimization-basics', 'Gradient descent and curvature', 'Why some losses are easy to walk down and others are a ravine.', 2),
      soon('foundations', 'sampling', 'Sampling from a distribution', 'Temperature, top-k, top-p, min-p — same logits, very different text.', 2),
    ],
  },
  {
    id: 'nn-core',
    title: 'Network building blocks',
    blurb: 'The parts every architecture is assembled from, and the specific failure each one was invented to fix.',
    concepts: [
      soon('nn-core', 'initialization', 'Weight initialisation', 'Why the starting scale decides whether signal survives 50 layers.', 2),
      soon('nn-core', 'normalization', 'LayerNorm, RMSNorm, BatchNorm', 'What normalisation actually fixes, and why pre-norm won.', 2),
      soon('nn-core', 'activations', 'Activations and gating', 'ReLU to GELU to SwiGLU, and what gating buys.', 2),
      soon('nn-core', 'residual-stream', 'The residual stream', 'Depth became possible the day someone added a skip connection.', 2),
      soon('nn-core', 'backprop', 'Backpropagation by hand', 'Every gradient in a small network, computed and checked.', 3),
      soon('nn-core', 'regularization', 'Weight decay, dropout, label smoothing', 'Three different ways to stop a model memorising.', 2),
    ],
  },
  {
    id: 'transformer',
    title: 'The transformer',
    blurb: 'Every tensor in a forward pass, one piece at a time. This is the track everything else assumes.',
    concepts: [
      attention,
      rope,
      soon('transformer', 'tokenization', 'Tokenisation', 'BPE merges happening character by character, and why models cannot spell or count.', 2),
      soon('transformer', 'embeddings', 'Embeddings and the output head', 'A lookup table that is secretly a matrix multiply.', 1),
      soon('transformer', 'multi-head', 'Multi-head attention', 'Why split one big attention into many small ones.', 2),
      soon('transformer', 'attention-variants', 'MHA, MQA, GQA, MLA', 'Four ways to shrink the KV cache, with what each one costs you.', 3),
      soon('transformer', 'ffn', 'The feed-forward block', 'Two thirds of the parameters live here and nobody talks about it.', 2),
      soon('transformer', 'block-layout', 'Block layout', 'Pre-norm, post-norm, sandwich — small choice, large consequences.', 3),
      soon('transformer', 'kv-cache', 'The KV cache', 'What gets stored, how it grows, and what it costs.', 2),
      soon('transformer', 'forward-pass', 'A full forward pass', 'GPT-2 end to end, every intermediate tensor visible.', 3),
    ],
  },
  {
    id: 'llm',
    title: 'Modern LLM architectures',
    blurb: 'What changed after GPT-2, and the reason behind each change. Mixture of experts, state space models, long context, reasoning.',
    concepts: [
      soon('llm', 'moe', 'Mixture of experts', 'Route each token to a few experts. Watch the router collapse when you remove the balance loss.', 4),
      soon('llm', 'mamba', 'State space models and Mamba', 'The same computation as a recurrence and as a convolution, proven equal.', 4),
      soon('llm', 'linear-attention', 'Linear attention, RWKV, RetNet', 'Giving up the n² matrix, and what you lose with it.', 4),
      soon('llm', 'long-context', 'Long context', 'Ring attention, context parallelism, and why the middle gets ignored.', 4),
      soon('llm', 'small-models', 'Small models (SLMs)', 'Phi, Gemma, Qwen small — what you can and cannot buy with distillation and data quality.', 3),
      soon('llm', 'reasoning', 'Reasoning models', 'Chain of thought, RL-trained reasoning, and spending compute at test time.', 4),
      soon('llm', 'architecture-diff', 'Architecture differ', 'Llama vs Qwen vs DeepSeek vs Gemma, side by side, with the stated reasons.', 3),
      soon('llm', 'structured-output', 'Constrained decoding', 'Masking logits with a grammar so the output is always valid JSON.', 3),
    ],
  },
  {
    id: 'training',
    title: 'Training at scale',
    blurb: 'Data, optimizers, schedules, parallelism, stability. What actually happens between "we have GPUs" and "we have a model".',
    concepts: [
      soon('training', 'optimizers', 'Optimizers', 'SGD to Adam to Muon, with the update each one applies drawn on the same surface.', 3),
      soon('training', 'lr-schedules', 'Learning rate schedules', 'Warmup, cosine, WSD — and what the loss curve does under each.', 2),
      soon('training', 'scaling-laws', 'Scaling laws', 'Given a compute budget, how big a model and how much data. With the dollar number.', 4),
      soon('training', 'parallelism', 'Parallelism', 'Data, tensor, pipeline, expert, context. Memory, bubbles and communication for each split.', 5),
      soon('training', 'memory-budget', 'Training memory', 'Weights, gradients, optimizer states, activations — the four terms people forget.', 3),
      soon('training', 'mixed-precision', 'Mixed precision', 'bf16 vs fp16, loss scaling, and where fp32 is still required.', 3),
      soon('training', 'stability', 'Loss spikes and instability', 'Real spike traces. You diagnose, then see what actually happened.', 4),
      soon('training', 'data-pipeline', 'Data', 'Dedup, filtering, mixtures, curriculum. The part that decides model quality.', 3),
      soon('training', 'collectives', 'Communication', 'All-reduce, all-gather, reduce-scatter, drawn on a real topology with byte counts.', 4),
    ],
  },
  {
    id: 'posttraining',
    title: 'Post-training',
    blurb: 'Turning a text predictor into something useful. Fine-tuning, preferences, RL, and evaluation that is not lying to you.',
    concepts: [
      soon('posttraining', 'sft', 'Supervised fine-tuning', 'Chat templates, loss masking, and the bugs that silently ruin a run.', 2),
      soon('posttraining', 'lora', 'LoRA and QLoRA', 'What a low rank update can and cannot express, shown on a real weight matrix.', 3),
      soon('posttraining', 'rlhf', 'RLHF and PPO', 'Reward model, rollouts, advantages, KL penalty — each piece and why it is there.', 4),
      soon('posttraining', 'dpo', 'DPO and friends', 'Skipping the reward model, and the length bias you get for free.', 4),
      soon('posttraining', 'grpo', 'GRPO and verifiable rewards', 'How reasoning models are actually trained.', 4),
      soon('posttraining', 'reward-hacking', 'Reward hacking', 'Optimise a proxy and watch true performance walk away from it.', 4),
      soon('posttraining', 'evaluation', 'Evaluation', 'Contamination, judge bias, and whether that +0.4% means anything.', 3),
      soon('posttraining', 'merging', 'Model merging', 'Task arithmetic, TIES, SLERP — averaging weights and getting away with it.', 3),
    ],
  },
  {
    id: 'vision',
    title: 'Vision',
    blurb: 'Convolutions, vision transformers, self-supervised learning, and dense prediction.',
    concepts: [
      soon('vision', 'convolution', 'Convolution', 'Sliding windows, and the matrix multiply hiding underneath.', 2),
      soon('vision', 'cnn-architectures', 'CNN architectures', 'LeNet to ResNet to ConvNeXt, and what each one fixed.', 2),
      soon('vision', 'vit', 'Vision transformer', 'Cut the image into patches and pretend they are words.', 3),
      soon('vision', 'ssl-vision', 'Self-supervised vision', 'SimCLR, DINO, MAE — learning without labels.', 4),
      soon('vision', 'jepa', 'JEPA', 'Predict in representation space instead of pixel space, and skip the detail you do not need.', 4),
      soon('vision', 'detection-segmentation', 'Detection and segmentation', 'DETR set prediction, SAM, and dense prediction heads.', 4),
      soon('vision', 'gaussian-splatting', '3D Gaussian splatting', 'Scenes as millions of fuzzy blobs, rendered in real time.', 4),
    ],
  },
  {
    id: 'multimodal',
    title: 'Multimodal',
    blurb: 'Getting images, audio and video into a language model, and what the model actually receives.',
    concepts: [
      soon('multimodal', 'clip', 'CLIP and SigLIP', 'Contrastive training, and the gap between the two modality clusters.', 3),
      soon('multimodal', 'vlm-connectors', 'VLM connectors', 'Linear, MLP, Q-Former, cross attention — the bridge between vision and language.', 4),
      soon('multimodal', 'resolution', 'Resolution and token budgets', 'A 4K screenshot becomes how many tokens, and at what cost.', 3),
      soon('multimodal', 'vlm-training', 'VLM training recipes', 'Alignment, instruction, high resolution — what freezes when.', 4),
      soon('multimodal', 'whisper-audio', 'Audio and speech', 'Mel spectrograms, audio tokens, and streaming recognition.', 3),
      soon('multimodal', 'omni', 'Any-to-any models', 'One token space for text, image and audio.', 4),
    ],
  },
  {
    id: 'diffusion',
    title: 'Generative and diffusion',
    blurb: 'VAEs, GANs, diffusion and flow matching. Full derivations, with the sampling tricks that make them usable.',
    concepts: [
      diffusionForward,
      soon('diffusion', 'vae', 'VAE', 'The reparameterisation trick, the ELBO, and posterior collapse.', 3),
      soon('diffusion', 'gan', 'GANs', 'Two networks fighting, and the mode collapse you can trigger on purpose.', 3),
      soon('diffusion', 'samplers', 'Samplers', 'DDIM, Euler, DPM-Solver — same model, same seed, different step counts.', 3),
      soon('diffusion', 'guidance', 'Classifier-free guidance', 'Run the model twice and push away from the unconditioned answer.', 3),
      soon('diffusion', 'unet-dit', 'U-Net and DiT', 'Two backbones for the same job, and why transformers took over.', 3),
      soon('diffusion', 'flow-matching', 'Flow matching and rectified flow', 'The view that makes diffusion look simple in hindsight.', 4),
      soon('diffusion', 'latent-diffusion', 'Latent diffusion', 'Compress first, diffuse in the small space, decode once.', 3),
      soon('diffusion', 'controlnet', 'ControlNet and adapters', 'Steering generation without retraining the model.', 3),
      soon('diffusion', 'distillation-steps', 'Few-step generation', 'From 1000 steps to 4, and exactly what you give up.', 4),
      soon('diffusion', 'video-diffusion', 'Video diffusion', 'Adding a time axis, and the cost that comes with it.', 4),
    ],
  },
  {
    id: 'retrieval',
    title: 'Embeddings and retrieval',
    blurb: 'Vector search, the index structures underneath it, and why RAG systems fail.',
    concepts: [
      soon('retrieval', 'embedding-geometry', 'Embedding geometry', 'What the space looks like, and why cosine similarity is the default.', 2),
      soon('retrieval', 'ann-indexes', 'Approximate nearest neighbours', 'Watch an HNSW graph get built, then watch a query hop through it.', 3),
      soon('retrieval', 'rag', 'RAG architecture', 'Chunking, retrieval, reranking, and the eight ways it goes wrong.', 3),
      soon('retrieval', 'rerankers', 'Rerankers', 'Cross encoders and late interaction, and when they are worth the latency.', 3),
    ],
  },
  {
    id: 'gpu',
    title: 'GPUs and kernels',
    blurb: 'What the hardware actually does. Memory hierarchy, CUDA, Triton, Flash Attention, quantisation, compilers, TPUs.',
    concepts: [
      soon('gpu', 'memory-hierarchy', 'Memory hierarchy', 'A tensor travelling from HBM to registers, with real latencies and sizes.', 3),
      soon('gpu', 'roofline', 'The roofline model', 'One chart that tells you whether to optimise math or memory.', 3),
      soon('gpu', 'coalescing', 'Memory coalescing', 'Write an access pattern and watch the memory transactions it causes.', 3),
      soon('gpu', 'occupancy', 'Occupancy and warps', 'Why more threads is not always faster.', 3),
      soon('gpu', 'matmul-ladder', 'The matmul ladder', 'Eight kernels, naive to tensor cores, with measured timings at every rung.', 5),
      soon('gpu', 'tensor-cores', 'Tensor cores', 'The MMA shapes the hardware actually wants.', 4),
      soon('gpu', 'triton', 'Triton', 'Write a fused kernel, compile it, read the generated code.', 4),
      soon('gpu', 'flash-attention', 'Flash Attention', 'Tiles streaming through on-chip memory, with a running max and sum.', 5),
      soon('gpu', 'quantization', 'Quantisation', 'Place the grid on a real weight distribution and watch the error. Find the outlier channels.', 4),
      soon('gpu', 'torch-compile', 'torch.compile and fusion', 'What gets fused, what causes a graph break, and what it saves.', 4),
      soon('gpu', 'cuda-graphs', 'CUDA graphs', 'Removing launch overhead when every kernel is tiny.', 4),
      soon('gpu', 'tpu', 'TPUs and systolic arrays', 'Weights and activations flowing through the array, cycle by cycle.', 4),
      soon('gpu', 'metal-mlx', 'Apple Silicon and MLX', 'Unified memory changes the arithmetic. What runs well on a laptop.', 3),
      soon('gpu', 'sparsity', 'Sparsity and pruning', '2:4 structured sparsity, and what the hardware will actually accelerate.', 4),
    ],
  },
  {
    id: 'inference',
    title: 'Inference and serving',
    blurb: 'Making models fast and cheap in production. Batching, paging, speculation, and real cost models.',
    concepts: [
      decodeCost,
      soon('inference', 'continuous-batching', 'Continuous batching', 'Scheduling at the iteration level instead of the request level.', 3),
      soon('inference', 'paged-attention', 'PagedAttention', 'Blocks allocated and freed live. Turn paging off and watch memory get wasted.', 4),
      soon('inference', 'speculative-decoding', 'Speculative decoding', 'Draft and verify, with the acceptance rate that decides if it was worth it.', 4),
      soon('inference', 'prefix-caching', 'Prefix and radix caching', 'Sharing a cached system prompt across thousands of requests.', 3),
      soon('inference', 'disaggregation', 'Disaggregated serving', 'Separate pools for prefill and decode, each sized for its own bottleneck.', 4),
      soon('inference', 'serving-stacks', 'vLLM, SGLang, TensorRT-LLM', 'The same deployment in each, and where each one wins.', 3),
      soon('inference', 'on-device', 'Running models locally', 'GGUF, quantisation, and what actually fits on your machine.', 2),
      soon('inference', 'cost-model', 'Cost per million tokens', 'The full derivation, from GPU rental to the number on the invoice.', 3),
    ],
  },
  {
    id: 'production',
    title: 'Production engineering',
    blurb: 'Everything between a working model and a system you can operate. Plus real case studies read as engineering decisions.',
    concepts: [
      soon('production', 'reproducibility', 'Reproducibility', 'What determinism costs, and when it is worth paying.', 2),
      soon('production', 'prod-evals', 'Evaluation in production', 'Canaries, regression suites and shadow traffic.', 3),
      soon('production', 'observability', 'Observability', 'What to log for a model system, and what to alert on.', 3),
      soon('production', 'case-studies', 'Case studies', 'Llama 3, DeepSeek-V3, Qwen 3, SD3 — read as decision documents.', 3),
    ],
  },
  {
    id: 'interp',
    title: 'Interpretability',
    blurb: 'Looking inside a trained model and finding out what it is doing. The most direct answer to "but how does it actually work".',
    concepts: [
      soon('interp', 'logit-lens', 'Logit lens', 'Decode the residual stream at every layer and watch the answer form.', 3),
      soon('interp', 'activation-patching', 'Activation patching', 'Swap one activation and see which ones actually caused the output.', 4),
      soon('interp', 'induction-heads', 'Induction heads', 'The circuit that makes in-context learning work.', 4),
      soon('interp', 'sae', 'Sparse autoencoders', 'Pulling apart superposed features, and steering a live generation with them.', 4),
      soon('interp', 'grokking', 'Grokking and double descent', 'Train past memorisation and watch generalisation appear.', 3),
    ],
  },
]

export const CONCEPTS: Concept[] = TRACKS.flatMap(t => t.concepts)
export const trackOf = (conceptId: string) => TRACKS.find(t => t.concepts.some(c => c.id === conceptId))
export const BUILT = CONCEPTS.filter(c => c.status !== 'planned')
