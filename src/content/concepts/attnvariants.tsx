import KVBudgetLab from '../../labs/KVBudgetLab'
import { Code, Eq, Note } from '../../components/ui'
import type { Concept } from '../types'

export const attentionVariants: Concept = {
  id: 'attention-variants',
  title: 'MHA, MQA, GQA, MLA',
  tagline: 'Four ways to shrink the KV cache. The cache, not the weights, is what decides how many users fit on a GPU.',
  track: 'transformer',
  difficulty: 3,
  status: 'stable',
  hasLab: true,
  prereqs: ['attention', 'decode-cost'],
  papers: [
    { title: 'Fast Transformer Decoding: One Write-Head is All You Need', year: 2019,
      url: 'https://arxiv.org/abs/1911.02150',
      idea: 'Share one key and value head across every query head. The cache shrinks by the head count.' },
    { title: 'GQA: Training Generalized Multi-Query Transformer Layers', year: 2023,
      url: 'https://arxiv.org/abs/2305.13245',
      idea: 'MQA loses too much quality. Share in groups instead, and you keep almost all of it for almost all of the saving.' },
    { title: 'DeepSeek-V2', year: 2024, url: 'https://arxiv.org/abs/2405.04434',
      idea: 'Do not store keys and values at all. Store a small latent vector and project it back up when needed.' },
  ],
  lenses: {
    intuition: () => (
      <>
        <p>During generation, every token you have produced so far has to keep its keys and values
        around, because every new token attends to all of them. That is the KV cache.</p>
        <p>It grows with every token, every layer, and every user at once. At long context it gets
        bigger than the model itself. So the obvious question is: does every attention head really
        need its own keys and values?</p>
        <p>It turns out no. Query heads can share. How much they share is the whole design space.</p>
        <ul>
          <li><b>MHA</b> — 32 query heads, 32 KV heads. Nothing shared.</li>
          <li><b>GQA</b> — 32 query heads, 8 KV heads. Groups of 4 share. Four times smaller cache.</li>
          <li><b>MQA</b> — 32 query heads, 1 KV head. Everything shared. 32 times smaller, and you
            can feel the quality drop.</li>
          <li><b>MLA</b> — store a compressed latent and reconstruct keys and values from it.</li>
        </ul>
      </>
    ),
    mechanism: () => (
      <>
        <p>Set a realistic serving workload and compare:</p>
        <KVBudgetLab />
        <h3>How sharing actually works</h3>
        <p>Nothing changes about the attention math. You still compute <Eq tex="QK^\top" /> per head.
        The difference is that several query heads point at the same K and V tensors. In code it is
        a repeat or an expand, which costs nothing in memory because it does not copy.</p>
        <h3>What MLA does differently</h3>
        <p>Instead of caching K and V, it caches a single smaller vector per token. When attention
        runs, K and V are projected back up from that latent. You trade a bit of extra compute for a
        much smaller cache — and because the up-projection can be folded into the query projection,
        most of that extra compute disappears.</p>
      </>
    ),
    math: () => (
      <>
        <h3>Cache size</h3>
        <Eq block tex="\text{bytes} = 2 \times L \times H_{kv} \times d_{head} \times S \times B \times b" />
        <p>Only <Eq tex="H_{kv}" /> changes between schemes. Everything else is fixed by the model
        and the workload, which is why this one number is worth so much.</p>
        <h3>A concrete case</h3>
        <p>Llama 3 8B, 32k context, one request, bf16:</p>
        <ul>
          <li>MHA (32 KV heads): 2 × 32 × 32 × 128 × 32768 × 2 = <b>17.2 GB</b></li>
          <li>GQA (8 KV heads): <b>4.3 GB</b></li>
          <li>MQA (1 KV head): <b>0.54 GB</b></li>
        </ul>
        <p>The weights are 16 GB. With MHA a single long request would use more memory for its cache
        than the entire model. With GQA you can serve several. That is the whole argument.</p>
        <h3>What sharing costs in quality</h3>
        <p>Each KV head is a subspace the model can look things up in. Fewer of them means less
        independent lookup capacity. The GQA paper found 8 groups loses very little on most tasks,
        while MQA's single head is a measurable drop — which is why nearly everyone settled on 8.</p>
      </>
    ),
    code: () => (
      <>
        <Code lang="python">{`import torch, torch.nn as nn, torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.n_rep = n_heads // n_kv_heads          # how many queries share one KV head
        self.d_head = d_model // n_heads

        self.wq = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)   # smaller
        self.wv = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)   # smaller
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cache=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        if cache is not None:
            k, v = cache.update(k, v)      # only n_kv_heads worth gets stored

        # expand, do not copy: each KV head is viewed by n_rep query heads
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None)
        return self.wo(out.transpose(1, 2).reshape(B, T, -1))`}</Code>
        <Note kind="info" title="repeat_interleave is not a copy here">
          <p>With a fused attention kernel the expansion is folded into the kernel's indexing, so no
          extra memory is used. If you write attention manually with explicit matmuls, it <em>will</em>
          materialise and you lose the benefit you came for.</p>
        </Note>
      </>
    ),
    systems: () => (
      <>
        <h3>This is a serving decision made at training time</h3>
        <p>You cannot switch a trained MHA model to GQA without retraining, though you can distil
        one into the other by mean-pooling the KV heads and fine-tuning briefly — which is what the
        GQA paper proposed and what several model conversions have done since.</p>
        <h3>What it buys in practice</h3>
        <p>More requests per GPU, directly. KV cache is usually the binding constraint on batch
        size, and batch size is the only real lever on decode throughput. A 4× smaller cache is
        close to 4× the concurrency at the same context length.</p>
        <h3>It also speeds up the decode step itself</h3>
        <p>Decode is memory bound and the cache is part of what gets read every step. Shrinking it
        shrinks the bytes moved, so the step gets faster too — not just more concurrent.</p>
        <h3>Where MLA wins and loses</h3>
        <p>MLA gets the smallest cache of all, which matters most at very long context. The cost is
        a more complicated kernel and an awkward interaction with RoPE, which DeepSeek handled by
        splitting the head into a part that gets rotation and a part that does not.</p>
      </>
    ),
    failure: () => (
      <>
        <table className="w-full text-[13px]">
          <tbody>
            {[['Out of memory at long context, weights fit fine', 'KV cache. Check the per-request cache size before assuming you need a bigger GPU.'],
              ['Throughput collapsed when users sent longer prompts', 'Cache grew, fewer requests admitted. This is a capacity problem, not a speed problem.'],
              ['Converted a model to GQA and quality dropped sharply', 'Mean-pool the KV heads rather than dropping them, and fine-tune afterwards.'],
              ['GQA model is slower than expected', 'repeat_interleave materialising a real tensor. Use a kernel that supports grouped heads.'],
              ['MLA implementation gives wrong results with RoPE', 'Rotation must be applied to the un-compressed part only. This is the standard MLA trap.'],
              ['Cache quantisation to int8 hurt long context badly', 'Error accumulates over sequence length. Keep the most recent tokens at full precision.']].map(r => (
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
