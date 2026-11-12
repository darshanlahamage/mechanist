import AttentionLab from '../../labs/AttentionLab'
import { Code, Eq, Note } from '../../components/ui'
import type { Concept } from '../types'

export const attention: Concept = {
  id: 'attention',
  title: 'Scaled dot-product attention',
  tagline: 'Every token asks every other token a question, and mixes the answers by how well they match.',
  track: 'transformer',
  difficulty: 2,
  status: 'stable',
  hasLab: true,
  prereqs: ['softmax-numerics'],
  papers: [
    { title: 'Attention Is All You Need', year: 2017, url: 'https://arxiv.org/abs/1706.03762',
      idea: 'Drop recurrence entirely. If every position can read every other position in one step, the whole sequence trains in parallel.' },
    { title: 'Neural Machine Translation by Jointly Learning to Align and Translate', year: 2014,
      url: 'https://arxiv.org/abs/1409.0473',
      idea: 'The first attention. Added so a decoder could look back at the encoder instead of relying on one fixed vector.' },
  ],
  lenses: {
    intuition: () => (
      <>
        <p>A token on its own means very little. "it" means nothing until you know what it points to.
        So each token needs a way to pull in information from the tokens around it.</p>

        <p>Attention does this with a lookup. Every token produces three vectors:</p>
        <ul>
          <li><b>Query</b> — what am I looking for?</li>
          <li><b>Key</b> — what do I contain?</li>
          <li><b>Value</b> — what will I hand over if you pick me?</li>
        </ul>

        <p>To update one token, you compare its query against every key. Where they match, you take
        more of that token's value. Where they do not, you take less. Then you add it all up.</p>

        <p>That is it. The comparison is a dot product, the "how much" is a softmax, and the mixing is
        a weighted sum. Everything else in a transformer is built around this one operation.</p>

        <Note kind="key" title="The part people miss">
          <p>Attention has no idea about word order. Shuffle the tokens and you get the same answers
          in a shuffled order. Position has to be injected separately — that is a different concept,
          and it exists only because attention does not care about order.</p>
        </Note>
      </>
    ),

    mechanism: () => (
      <>
        <p>Start with the input, one row per token:</p>
        <Eq block tex="X \in \mathbb{R}^{n \times d_{model}}" />
        <p>Three learned matrices turn each token into a query, a key and a value:</p>
        <Eq block tex="Q = XW_Q,\quad K = XW_K,\quad V = XW_V" />
        <p>Compare every query with every key. That is one matrix multiply, and it gives an
        <code>n × n</code> grid of scores:</p>
        <Eq block tex="S = \frac{QK^\top}{\sqrt{d_{head}}}" />
        <p>Block the future, if this is a decoder. Set anything a token should not see to negative
        infinity, so softmax gives it exactly zero:</p>
        <Eq block tex="S_{ij} \leftarrow -\infty \quad \text{for } j > i" />
        <p>Turn each row of scores into weights that sum to 1, then mix the values:</p>
        <Eq block tex="\text{Attention}(Q,K,V) = \text{softmax}(S)V" />

        <p>Play with it. Change the head size and turn the <code>√d</code> off:</p>
        <AttentionLab />

        <h3>Why divide by √d</h3>
        <p>Q and K rows are roughly unit-variance random vectors of length <code>d_head</code>. Their
        dot product is a sum of <code>d_head</code> such products, so its standard deviation grows
        like <code>√d_head</code>. With a head size of 128, raw scores swing by ±11 or more. Softmax
        on numbers that far apart returns almost exactly one-hot, the gradient becomes almost exactly
        zero, and the layer stops learning. Dividing by <code>√d_head</code> pulls the spread back to
        around 1 no matter how big the head is.</p>
      </>
    ),

    math: () => (
      <>
        <h3>Shapes, since this is where people get lost</h3>
        <table className="w-full font-mono text-[12.5px]">
          <tbody>
            {[['X', 'n × d_model', 'input tokens'],
              ['W_Q, W_K, W_V', 'd_model × d_head', 'learned projections'],
              ['Q, K, V', 'n × d_head', 'per-token vectors'],
              ['QKᵀ', 'n × n', 'every token vs every token'],
              ['softmax(S)', 'n × n', 'rows sum to 1'],
              ['output', 'n × d_head', 'same shape as V']].map(r => (
              <tr key={r[0]} className="border-t border-[var(--line)]">
                <td className="py-1.5 text-[var(--accent)]">{r[0]}</td>
                <td className="py-1.5 text-[#c7cdd8]">{r[1]}</td>
                <td className="py-1.5 text-[var(--dim)]">{r[2]}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <h3>Where the √d comes from</h3>
        <p>Take <Eq tex="q, k \in \mathbb{R}^{d}" /> with independent entries of mean 0 and variance 1.
        The dot product is</p>
        <Eq block tex="q \cdot k = \sum_{i=1}^{d} q_i k_i" />
        <p>Each term has mean 0 and variance 1, and they are independent, so variances add:</p>
        <Eq block tex="\mathrm{Var}(q \cdot k) = d \quad\Rightarrow\quad \mathrm{sd}(q\cdot k) = \sqrt{d}" />
        <p>Divide by <Eq tex="\sqrt{d}" /> and the standard deviation is back to 1, independent of head
        size. This is the entire argument — it is a variance fix, not a modelling idea.</p>

        <h3>The gradient of softmax, and why saturation kills it</h3>
        <Eq block tex="\frac{\partial p_i}{\partial s_j} = p_i(\delta_{ij} - p_j)" />
        <p>If one <Eq tex="p_i \approx 1" /> and the rest are near 0, every term here is near 0. No
        gradient flows back to the scores, so <Eq tex="W_Q" /> and <Eq tex="W_K" /> stop updating.
        Saturated attention is a dead layer.</p>

        <h3>Cost</h3>
        <p>The score matrix is <Eq tex="n \times n" />. Both compute and memory grow with
        <Eq tex="n^2" />, which is the single fact that drives Flash Attention, sliding windows,
        linear attention and every long-context trick.</p>
        <Eq block tex="\text{FLOPs} \approx 4nd^2 + 2n^2d" />
      </>
    ),

    code: () => (
      <>
        <h3>From scratch, no library magic</h3>
        <Code lang="python">{`import numpy as np

def attention(X, Wq, Wk, Wv, causal=True):
    Q = X @ Wq                      # (n, d_head)
    K = X @ Wk
    V = X @ Wv
    d_head = Q.shape[-1]

    S = Q @ K.T / np.sqrt(d_head)   # (n, n) -- the quadratic part

    if causal:
        n = S.shape[0]
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        S = np.where(mask, -np.inf, S)

    S = S - S.max(axis=-1, keepdims=True)   # numerical safety
    P = np.exp(S)
    P = P / P.sum(axis=-1, keepdims=True)

    return P @ V                    # (n, d_head)`}</Code>

        <h3>Multi-head, the way real code does it</h3>
        <Code lang="python">{`import torch, torch.nn as nn, torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)  # one matmul, not three
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_heads, T, d_head): heads become a batch dimension
        q, k, v = (t.view(B, T, self.n_heads, self.d_head).transpose(1, 2) for t in (q, k, v))

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # fused kernel

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)`}</Code>

        <Note kind="info" title="Two things worth noticing">
          <p>The three projections are fused into one <code>nn.Linear</code>. One bigger matmul beats
          three small ones on a GPU.</p>
          <p><code>scaled_dot_product_attention</code> does not build the n×n matrix in memory at all.
          It calls a Flash Attention kernel. Same math, very different memory behaviour.</p>
        </Note>
      </>
    ),

    systems: () => (
      <>
        <h3>The two halves scale differently</h3>
        <p>Attention is really two different workloads glued together:</p>
        <ul>
          <li>The four projections (Q, K, V, output) are plain matmuls. Cost grows as
            <Eq tex="n \cdot d^2" /> — linear in sequence length.</li>
          <li>The score and mixing matmuls cost <Eq tex="n^2 \cdot d" /> — quadratic.</li>
        </ul>
        <p>Short sequences are dominated by the projections. Long sequences are dominated by the
        quadratic part. The crossover is around <Eq tex="n \approx 2d" />, so for a model with
        d_model 4096 the quadratic term only takes over past roughly 8k tokens. Below that, "attention
        is quadratic" is technically true and practically irrelevant.</p>

        <h3>Memory is the real problem, not FLOPs</h3>
        <p>A naive implementation writes the full <Eq tex="n \times n" /> score matrix to memory, then
        reads it back for softmax, then again for the value multiply. At 8k tokens with 32 heads in
        bf16 that is 8192² × 32 × 2 bytes = <b>4.3 GB</b> for a single layer of a single sequence.
        Flash Attention produces identical numbers while never writing that matrix at all — it keeps
        tiles in on-chip memory and carries a running max and sum.</p>

        <h3>At inference time it changes shape again</h3>
        <p>When generating, there is exactly one new query and the keys and values come from cache.
        The n×n matrix becomes a 1×n vector. Attention stops being a matmul problem and becomes a
        memory-reading problem — you are streaming the whole KV cache to produce one token.</p>
      </>
    ),

    failure: () => (
      <>
        <h3>Things that actually go wrong</h3>
        <table className="w-full text-[13px]">
          <thead><tr className="text-[var(--dim)]">
            <th className="py-2 text-left font-normal">What you see</th>
            <th className="py-2 text-left font-normal">What it usually is</th>
          </tr></thead>
          <tbody>
            {[['Loss stuck, attention looks one-hot', 'Scores not divided by √d, or logits drifting large. Try QK-norm.'],
              ['NaN loss right after a long sequence', 'A fully masked row. Softmax over all −∞ gives 0/0.'],
              ['Model repeats or copies the prompt', 'Mask bug — a causal mask applied off by one, letting a token see itself ahead.'],
              ['Works in training, broken in generation', 'Padding mask dropped at inference, so the model attends to pad tokens.'],
              ['Fine up to 4k tokens, garbage after', 'Not attention — that is the position encoding running past what it was trained on.'],
              ['Out of memory only on long batches', 'Naive attention materialising n² scores. Switch to a fused kernel.']].map(r => (
              <tr key={r[0]} className="border-t border-[var(--line)] align-top">
                <td className="py-2 pr-4 text-[#c7cdd8]">{r[0]}</td>
                <td className="py-2 text-[var(--dim)]">{r[1]}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <h3>The fully-masked row</h3>
        <p>This one bites everybody. If a row is masked everywhere, every score is −∞, so
        <code>exp</code> gives all zeros and you divide zero by zero:</p>
        <Code lang="python">{`S = np.array([-np.inf, -np.inf, -np.inf])
e = np.exp(S - S.max())   # S.max() is -inf  ->  -inf - -inf = nan
print(e / e.sum())        # [nan nan nan]`}</Code>
        <p>It happens with padded batches where one sequence has length zero, or with a sliding window
        that slides off the start. Fix it by masking with a large negative number instead of true
        infinity (<code>-1e9</code>), or by skipping empty rows.</p>

        <Note kind="warn" title="Try it in the lab">
          <p>Set head size to 64 and turn off "divide by √d". The peak weight jumps near 1.0 and the
          entropy collapses. That is a layer that has stopped learning.</p>
        </Note>
      </>
    ),
  },
}
