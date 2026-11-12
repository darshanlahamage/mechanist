import TrainMemoryLab from '../../labs/TrainMemoryLab'
import { Code, Eq, Note } from '../../components/ui'
import type { Concept } from '../types'

export const trainingMemory: Concept = {
  id: 'memory-budget',
  title: 'Training memory',
  tagline: 'The weights are the small part. Four other things are bigger, and people forget all four.',
  track: 'training',
  difficulty: 3,
  status: 'stable',
  hasLab: true,
  prereqs: ['optimizers'],
  papers: [
    { title: 'ZeRO: Memory Optimizations Toward Training Trillion Parameter Models', year: 2019,
      url: 'https://arxiv.org/abs/1910.02054',
      idea: 'Data parallel keeps a full copy of everything on every GPU. Split the optimizer state, then the gradients, then the weights, and the duplication disappears.' },
    { title: 'Training Deep Nets with Sublinear Memory Cost', year: 2016, url: 'https://arxiv.org/abs/1604.06174',
      idea: 'Do not store every activation. Keep a few and recompute the rest during the backward pass.' },
  ],
  lenses: {
    intuition: () => (
      <>
        <p>"My model is 8B parameters and my GPU has 80 GB, so it fits." It does not.</p>
        <p>Five things live in memory during training:</p>
        <ul>
          <li><b>Weights</b> — 2 bytes per parameter in bf16.</li>
          <li><b>Gradients</b> — one per weight, so another 2 bytes.</li>
          <li><b>An fp32 master copy</b> — 4 bytes, because bf16 is too coarse to accumulate tiny
            updates into.</li>
          <li><b>Adam state</b> — momentum and variance, both fp32, so 8 bytes.</li>
          <li><b>Activations</b> — everything saved during the forward pass for the backward pass to
            use. This one scales with batch size and sequence length.</li>
        </ul>
        <p>That is 16 bytes per parameter before a single activation. An 8B model needs 128 GB just
        to hold the training state.</p>
      </>
    ),
    mechanism: () => (
      <>
        <p>Turn sharding and checkpointing on and watch an impossible run become possible:</p>
        <TrainMemoryLab />
        <h3>What ZeRO actually splits</h3>
        <ul>
          <li><b>Stage 1</b> — optimizer state is split across GPUs. Biggest win for least
            communication, since each GPU only updates its own slice.</li>
          <li><b>Stage 2</b> — gradients are split too. Gradients are reduce-scattered instead of
            all-reduced, which moves the same bytes.</li>
          <li><b>Stage 3 (FSDP)</b> — weights are split as well. Each layer's weights are gathered
            just before use and released after. Lowest memory, most communication.</li>
        </ul>
        <h3>Activation checkpointing</h3>
        <p>Instead of keeping every intermediate tensor, keep only the input to each block and
        recompute the rest during the backward pass. Roughly 30% more compute for a large drop in
        activation memory. Selective versions only recompute the cheap parts and keep the expensive
        ones, which is usually the better trade.</p>
      </>
    ),
    math: () => (
      <>
        <h3>The fixed part</h3>
        <Eq block tex="M_{static} = P \cdot (b_w + b_g + 4 + 8)" />
        <p>With bf16 weights and gradients that is <Eq tex="16P" /> bytes. For a 7B model: 112 GB.
        It does not fit on one 80 GB GPU, full stop, regardless of batch size.</p>
        <h3>The part that scales with your batch</h3>
        <Eq block tex="M_{act} \approx B \cdot S \cdot d_{model} \cdot L \cdot k \cdot b" />
        <p><Eq tex="k" /> is roughly 10 to 20 depending on what the implementation saves. Note that
        this grows linearly with both batch and sequence length — and attention adds a term that
        grows with <Eq tex="S^2" /> unless you are using a fused kernel that avoids materialising
        the score matrix.</p>
        <h3>What sharding gives you</h3>
        <Eq block tex="M_{\text{ZeRO-3}} = \frac{P(b_w + b_g + 12)}{G} + M_{act}" />
        <p>Activations are not sharded by ZeRO — they belong to the data on that GPU. That is why
        at large scale activation memory ends up dominating, and why sequence or context parallelism
        exists to split that term too.</p>
      </>
    ),
    code: () => (
      <>
        <Code lang="python">{`def training_memory_gb(params, batch, seq, d_model, layers,
                       dtype_bytes=2, optimizer="adamw", checkpointing=False):
    weights = params * dtype_bytes
    grads   = params * dtype_bytes
    master  = params * 4
    opt     = params * (8 if optimizer == "adamw" else 4)

    acts = batch * seq * d_model * layers * 16 * dtype_bytes
    if checkpointing:
        acts = acts / (layers ** 0.5)      # roughly sqrt(L) with optimal placement

    total = weights + grads + master + opt + acts
    return {k: v / 1e9 for k, v in
            dict(weights=weights, grads=grads, master=master,
                 optimizer=opt, activations=acts, total=total).items()}

print(training_memory_gb(8e9, batch=4, seq=4096, d_model=4096, layers=32))`}</Code>
        <Code lang="python">{`import torch

# find out where it actually went, rather than guessing
print(torch.cuda.memory_summary())

# the big levers, in the order worth trying
model.gradient_checkpointing_enable()            # activations
opt = bnb.optim.AdamW8bit(model.parameters())    # optimizer state: 8 bytes -> 2
# then FSDP / ZeRO-3 for everything else`}</Code>
        <Note kind="info" title="Fragmentation is real">
          <p><code>memory_allocated</code> can be well below <code>memory_reserved</code> and you
          still get an out-of-memory error. Variable sequence lengths cause this. Setting
          <code> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True</code> often fixes it outright.</p>
        </Note>
      </>
    ),
    systems: () => (
      <>
        <h3>The order to reach for things</h3>
        <ol className="ml-4 list-decimal space-y-1">
          <li>Smaller batch per GPU, with gradient accumulation to keep the effective batch. Free.</li>
          <li>Activation checkpointing. Costs about 30% throughput, large memory win.</li>
          <li>8-bit optimizer states. Saves 6 bytes per parameter, small quality risk.</li>
          <li>ZeRO stage 1, then 2, then 3. Progressively more communication.</li>
          <li>Tensor parallel, if you have fast interconnect within the node.</li>
          <li>CPU or NVMe offload. Works, but slow enough that it is usually a last resort.</li>
        </ol>
        <h3>Why the fp32 master copy exists</h3>
        <p>bf16 has 7 mantissa bits. A weight of 1.0 and an update of 1e-4 add to exactly 1.0 in
        bf16 — the update rounds away entirely. Keeping the master copy in fp32 means those small
        updates accumulate instead of vanishing. Removing it to save memory breaks training in a way
        that is very hard to diagnose.</p>
      </>
    ),
    failure: () => (
      <>
        <table className="w-full text-[13px]">
          <tbody>
            {[['Out of memory on step 1', 'Static memory. Batch size will not help. Shard or shrink the model.'],
              ['Out of memory after several hundred steps', 'Fragmentation, or something accumulating. Check for tensors kept in a list for logging.'],
              ['Fine at seq 2048, dies at 4096', 'Activations scale linearly with sequence, and attention scores quadratically without a fused kernel.'],
              ['Checkpointing enabled but memory barely moved', 'Applied to the wrong modules, or the memory was never in activations to begin with.'],
              ['FSDP uses more memory than expected', 'Weights are gathered per layer. Peak includes one full layer un-sharded.'],
              ['Loss stops improving after switching to pure bf16', 'The fp32 master copy was removed. Small updates are rounding to zero.']].map(r => (
              <tr key={r[0]} className="border-t border-[var(--line)] align-top">
                <td className="py-2 pr-4 text-[#c7cdd8]">{r[0]}</td>
                <td className="py-2 text-[var(--dim)]">{r[1]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </>
    ),
  },
}
