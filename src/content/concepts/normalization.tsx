import NormLab from '../../labs/NormLab'
import { Code, Eq, Note } from '../../components/ui'
import type { Concept } from '../types'

export const normalization: Concept = {
  id: 'normalization',
  title: 'LayerNorm, RMSNorm, BatchNorm',
  tagline: 'They all divide by a standard deviation. The only question is which direction you measure it in.',
  track: 'nn-core',
  difficulty: 2,
  status: 'stable',
  hasLab: true,
  papers: [
    { title: 'Layer Normalization', year: 2016, url: 'https://arxiv.org/abs/1607.06450',
      idea: 'Normalise across a single example\'s features instead of across the batch, so batch size stops mattering.' },
    { title: 'Root Mean Square Layer Normalization', year: 2019, url: 'https://arxiv.org/abs/1910.07467',
      idea: 'Skip the mean subtraction. The rescaling was doing the work; centring was not.' },
    { title: 'On Layer Normalization in the Transformer Architecture', year: 2020,
      url: 'https://arxiv.org/abs/2002.04745',
      idea: 'Putting the norm inside the residual branch instead of after it removes the need for learning rate warmup.' },
  ],
  lenses: {
    intuition: () => (
      <>
        <p>Deep networks have a scale problem. Each layer multiplies its input by weights, and small
        differences in scale compound. Twenty layers later some activations are enormous and others
        are tiny, and gradients are useless.</p>
        <p>Normalisation resets the scale. Take a group of numbers, subtract their mean, divide by
        their standard deviation, then let the network learn a scale and shift back if it wants one.</p>
        <p>The only real difference between the methods is which group you take.</p>
        <ul>
          <li><b>LayerNorm</b> — all the features of one token.</li>
          <li><b>RMSNorm</b> — same group, but it skips the mean subtraction.</li>
          <li><b>BatchNorm</b> — one feature, across every example in the batch.</li>
        </ul>
      </>
    ),
    mechanism: () => (
      <>
        <p>Switch between them and watch which table goes flat:</p>
        <NormLab />
        <h3>Where it sits in a transformer block</h3>
        <p>Two placements exist, and the difference is not cosmetic:</p>
        <ul>
          <li><b>Post-norm</b> (original transformer): <code>x = norm(x + attn(x))</code>. The
            residual path goes through the norm, so the gradient gets rescaled at every layer.
            Needs careful warmup or it diverges.</li>
          <li><b>Pre-norm</b> (everything modern): <code>x = x + attn(norm(x))</code>. The residual
            path is untouched, so gradients flow straight from the loss to the first layer. Much
            more stable, at the cost of the residual stream growing in magnitude with depth.</li>
        </ul>
      </>
    ),
    math: () => (
      <>
        <h3>LayerNorm</h3>
        <Eq block tex="\mu = \frac{1}{d}\sum_i x_i, \qquad \sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2" />
        <Eq block tex="y_i = \gamma_i \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i" />
        <h3>RMSNorm</h3>
        <Eq block tex="y_i = \gamma_i \cdot \frac{x_i}{\sqrt{\frac{1}{d}\sum_j x_j^2 + \epsilon}}" />
        <p>No mean, no shift. That removes one pass over the data and two learned vectors.
        The paper's finding was that the rescaling is what stabilises training, and the recentring
        contributes almost nothing — which held up, since Llama, Qwen, Gemma and most recent models
        use RMSNorm.</p>
        <h3>Why the gradient behaves better</h3>
        <p>Normalising makes the output scale-invariant with respect to the input:</p>
        <Eq block tex="\text{LN}(\alpha x) = \text{LN}(x)" />
        <p>So if a weight update accidentally scales activations up by 10, the next layer does not
        notice. That is the stabilising effect, and it is also why weight decay interacts oddly with
        normalised layers — shrinking the weights before a norm changes nothing except through the
        epsilon.</p>
      </>
    ),
    code: () => (
      <>
        <Code lang="python">{`import torch, torch.nn as nn

class RMSNorm(nn.Module):
    """Six lines. This is the whole thing."""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        # accumulate in fp32 even when x is bf16 -- the square can overflow otherwise
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight`}</Code>
        <Code lang="python">{`# pre-norm block: the residual path never passes through a norm
def block(x, attn, ffn, n1, n2):
    x = x + attn(n1(x))
    x = x + ffn(n2(x))
    return x

# post-norm block: it does, and that is why it needs warmup
def block_post(x, attn, ffn, n1, n2):
    x = n1(x + attn(x))
    x = n2(x + ffn(x))
    return x`}</Code>
      </>
    ),
    systems: () => (
      <>
        <h3>Memory bound, always fused</h3>
        <p>Normalisation does a handful of operations per element against the full cost of reading
        and writing the tensor. Run on its own it is pure bandwidth. Every production implementation
        fuses it with whatever comes next, which is why "write a fused layernorm" is the standard
        second exercise in every Triton tutorial.</p>
        <h3>RMSNorm is measurably cheaper</h3>
        <p>One pass instead of two, no mean, no bias vector. On its own that is a small win, but it
        runs twice per layer in every forward and backward pass, so it adds up.</p>
        <h3>BatchNorm does not fit language models at all</h3>
        <p>It needs batch statistics, which means the result for one sequence depends on the others
        in the batch. With variable-length text, batch size 1 at inference, and sequence parallelism
        splitting the batch across devices, it is unusable. It also forces a train/eval mode
        difference that is a permanent source of bugs.</p>
      </>
    ),
    failure: () => (
      <>
        <table className="w-full text-[13px]">
          <tbody>
            {[['Training diverges in the first few hundred steps', 'Post-norm without warmup. Switch to pre-norm or warm up the learning rate.'],
              ['Model works in training, garbage at inference', 'BatchNorm left in train mode, or running statistics never updated.'],
              ['Loss spikes late in a long run', 'Residual stream magnitude grew with depth under pre-norm. QK-norm or a final norm helps.'],
              ['NaN inside the norm in bf16', 'The sum of squares overflowed. Cast to fp32 before squaring.'],
              ['Replacing LayerNorm with RMSNorm hurt quality', 'Usually the epsilon or the fp32 cast, not the method. Check both before blaming RMSNorm.'],
              ['Weight decay has no visible effect on some layers', 'Expected. Normalised layers are scale-invariant, so shrinking those weights does almost nothing.']].map(r => (
              <tr key={r[0]} className="border-t border-[var(--line)] align-top">
                <td className="py-2 pr-4 text-[#c7cdd8]">{r[0]}</td>
                <td className="py-2 text-[var(--dim)]">{r[1]}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <Note kind="warn" title="Try it in the lab">
          <p>Set the drift high with BatchNorm selected. The per-feature table goes flat but the
          per-token one does not — and in a language model each token is an independent thing you
          care about, so that is the wrong table to fix.</p>
        </Note>
      </>
    ),
  },
}
