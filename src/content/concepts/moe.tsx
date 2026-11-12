import MoELab from '../../labs/MoELab'
import { Code, Eq, Note } from '../../components/ui'
import type { Concept } from '../types'

export const moe: Concept = {
  id: 'moe',
  title: 'Mixture of experts',
  tagline: 'Hold a lot of parameters, use a few of them per token. The hard part is stopping the router from picking favourites.',
  track: 'llm',
  difficulty: 4,
  status: 'stable',
  hasLab: true,
  prereqs: ['attention'],
  papers: [
    { title: 'Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer', year: 2017,
      url: 'https://arxiv.org/abs/1701.06538',
      idea: 'Route each example to a few sub-networks out of thousands, and add a loss term that keeps the routing balanced.' },
    { title: 'Switch Transformers', year: 2021, url: 'https://arxiv.org/abs/2101.03961',
      idea: 'Route to exactly one expert. Simpler, faster, and it still works if you handle capacity and precision carefully.' },
    { title: 'Mixtral of Experts', year: 2024, url: 'https://arxiv.org/abs/2401.04088',
      idea: '8 experts, top-2 routing, in an open model. 47B parameters held, 13B used per token.' },
    { title: 'DeepSeekMoE', year: 2024, url: 'https://arxiv.org/abs/2401.06066',
      idea: 'Many small experts plus a few always-on shared ones, so specialists stay specialised and common knowledge is not duplicated.' },
  ],
  lenses: {
    intuition: () => (
      <>
        <p>A dense model uses every parameter for every token. That is wasteful — the weights that
        help with Python syntax are not the ones that help with French grammar, but you pay for both
        on every token.</p>
        <p>A mixture of experts replaces the feed-forward block with many copies of it, and a small
        router that picks two of them per token. You store all the experts but only compute the
        selected ones.</p>
        <p>So you get the knowledge capacity of a very large model at the compute cost of a small
        one. Mixtral holds 47B parameters and uses about 13B per token.</p>
        <Note kind="bad" title="The problem that makes this hard">
          <p>Nothing forces the router to use its experts evenly. An expert that gets picked more
          gets more gradient, gets better, and gets picked more. Left alone, a few experts take
          everything and the rest are dead weight you still pay to store.</p>
        </Note>
      </>
    ),
    mechanism: () => (
      <>
        <p>Turn the balancing loss off and watch the entropy fall:</p>
        <MoELab />
        <h3>One token's journey</h3>
        <ul>
          <li>The token's vector goes through a small linear layer, giving one score per expert.</li>
          <li>Softmax over those scores gives routing weights.</li>
          <li>Take the top k (usually 1 or 2). Those experts run on this token.</li>
          <li>Their outputs are combined, weighted by the router's scores.</li>
        </ul>
        <p>Because the router's weights multiply the output, gradient flows back into the router.
        That is the only thing teaching it to route sensibly — and it is also the feedback loop that
        causes collapse.</p>
        <h3>Capacity and dropping</h3>
        <p>For the batch to run efficiently every expert needs a fixed-size buffer. If more tokens
        pick an expert than fit, the extras are <b>dropped</b> — they skip the layer entirely and
        pass through on the residual connection. Silently. Capacity factor is how much headroom you
        give over the even split.</p>
      </>
    ),
    math: () => (
      <>
        <h3>The layer</h3>
        <Eq block tex="g = \text{softmax}(W_r x), \qquad \mathcal{T} = \text{top-}k(g)" />
        <Eq block tex="y = \sum_{i \in \mathcal{T}} \frac{g_i}{\sum_{j \in \mathcal{T}} g_j} \cdot E_i(x)" />
        <h3>The auxiliary loss</h3>
        <p>Let <Eq tex="f_i" /> be the fraction of tokens routed to expert <Eq tex="i" />, and
        <Eq tex="P_i" /> the mean router probability for it. The balancing loss is:</p>
        <Eq block tex="L_{aux} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i" />
        <p>This is minimised when both are uniform. Multiplying them is deliberate: <Eq tex="f_i" />
        is a hard count with no gradient, so the gradient arrives through <Eq tex="P_i" />, pushing
        down the router's enthusiasm for whichever experts are currently overloaded.</p>
        <p><Eq tex="\alpha" /> is usually around 0.01. Too small and you get collapse; too large and
        the router balances instead of routing well, which costs quality.</p>
        <h3>What you actually save</h3>
        <Eq block tex="\text{FLOPs per token} \propto k \cdot d_{ffn}, \qquad \text{memory} \propto N \cdot d_{ffn}" />
        <p>Compute scales with k. Memory scales with the number of experts. That asymmetry is the
        entire design — and it is also why MoE is harder to serve than to train.</p>
      </>
    ),
    code: () => (
      <>
        <Code lang="python">{`import torch, torch.nn as nn, torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, d_model, d_ffn, n_experts=8, top_k=2):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([FFN(d_model, d_ffn) for _ in range(n_experts)])
        self.top_k, self.n_experts = top_k, n_experts

    def forward(self, x):
        B, T, D = x.shape
        flat = x.view(-1, D)

        logits = self.router(flat)                       # (tokens, n_experts)
        probs = logits.softmax(-1)
        weights, idx = probs.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)

        out = torch.zeros_like(flat)
        for e in range(self.n_experts):
            mask = (idx == e)
            if not mask.any():
                continue
            token_ids, slot = mask.nonzero(as_tuple=True)
            out[token_ids] += weights[token_ids, slot, None] * self.experts[e](flat[token_ids])

        # the loss that stops one expert eating everything
        f = torch.zeros(self.n_experts, device=x.device)
        f.scatter_add_(0, idx.flatten(), torch.ones_like(idx.flatten(), dtype=f.dtype))
        f = f / idx.numel()
        aux = self.n_experts * (f * probs.mean(0)).sum()

        return out.view(B, T, D), aux`}</Code>
        <Code lang="python">{`# total loss. forgetting this term is the single most common MoE bug
loss = cross_entropy(logits, targets) + 0.01 * sum(aux_losses)`}</Code>
      </>
    ),
    systems: () => (
      <>
        <h3>Serving MoE is harder than serving dense</h3>
        <p>You need every expert in memory even though you use two per token. Mixtral 8x7B needs
        about 93 GB in bf16 to serve, while computing like a 13B model. Memory capacity, not compute,
        is the constraint.</p>
        <h3>Expert parallelism and the all-to-all</h3>
        <p>At scale experts live on different GPUs. Every token must be sent to whichever devices
        hold its chosen experts, and the results sent back. That is two all-to-all collectives per
        MoE layer, and they are usually the slowest part of the step. Imbalanced routing makes it
        worse: the batch finishes when the busiest device finishes.</p>
        <h3>Batching behaves differently</h3>
        <p>With a large enough batch every expert gets work and the hardware stays busy. With a small
        batch you are running many small matmuls, which is terrible for a GPU. MoE needs scale to pay
        off, which is why it shows up in frontier models and rarely in small deployments.</p>
      </>
    ),
    failure: () => (
      <>
        <table className="w-full text-[13px]">
          <tbody>
            {[['A few experts take most tokens', 'Auxiliary loss missing, too weak, or not added to the total loss. Check the routing entropy, not the training loss.'],
              ['Loss looks fine but quality is poor', 'Tokens being dropped at capacity. They skip the layer silently and nothing in the loss tells you.'],
              ['Throughput much worse than the FLOPs suggest', 'Routing imbalance. The batch runs at the speed of the busiest device.'],
              ['Training unstable compared to a dense model', 'Router logits in bf16. Compute the router in fp32 — it is tiny and the precision matters.'],
              ['Works at batch 1024, falls apart at batch 8', 'Not enough tokens for the experts to form useful batches. MoE needs scale.'],
              ['Fine-tuning an MoE collapses the routing', 'Narrow fine-tuning data makes one expert correct for everything. Keep the aux loss on.']].map(r => (
              <tr key={r[0]} className="border-t border-[var(--line)] align-top">
                <td className="py-2 pr-4 text-[#c7cdd8]">{r[0]}</td>
                <td className="py-2 text-[var(--dim)]">{r[1]}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <Note kind="warn" title="Try it in the lab">
          <p>Turn off the balancing loss and raise the training steps. Entropy falls, one bar takes
          over, and the dropped-token counter starts climbing. All three are symptoms of the same
          feedback loop.</p>
        </Note>
      </>
    ),
  },
}
