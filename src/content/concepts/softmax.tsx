import SoftmaxLab from '../../labs/SoftmaxLab'
import { Code, Eq, Note } from '../../components/ui'
import type { Concept } from '../types'

export const softmaxNumerics: Concept = {
  id: 'softmax-numerics',
  title: 'Softmax and floating point',
  tagline: 'The formula in the paper and the formula in the kernel are not the same, and the difference is the whole lesson.',
  track: 'foundations',
  difficulty: 2,
  status: 'stable',
  hasLab: true,
  papers: [
    { title: 'Online normalizer calculation for softmax', year: 2018, url: 'https://arxiv.org/abs/1805.02867',
      idea: 'You can compute softmax in a single pass, carrying a running max and a running sum. Flash Attention is built on this.' },
  ],
  lenses: {
    intuition: () => (
      <>
        <p>Softmax turns any list of numbers into a probability distribution. Big numbers get big
        shares, small numbers get small shares, everything sums to 1.</p>
        <Eq block tex="p_i = \frac{e^{x_i}}{\sum_j e^{x_j}}" />
        <p>Written like that, it does not work. <code>exp</code> grows absurdly fast. In float32 the
        largest number you can hold is about 3.4 × 10³⁸, and <code>exp(89)</code> already passes it.
        Model logits routinely reach into the hundreds.</p>
        <p>The fix is to subtract the largest value first. It cancels out of the fraction exactly, so
        the answer is unchanged, but now the biggest thing you ever exponentiate is
        <code>exp(0) = 1</code>.</p>
      </>
    ),
    mechanism: () => (
      <>
        <p>Turn the logits up and switch the max subtraction off. Watch the exact moment it breaks:</p>
        <SoftmaxLab />
        <h3>Why the subtraction is free</h3>
        <p>Multiply top and bottom by the same constant <Eq tex="e^{-m}" />:</p>
        <Eq block tex="\frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i}e^{-m}}{\sum_j e^{x_j}e^{-m}} = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}" />
        <p>Any <Eq tex="m" /> works. Picking <Eq tex="m = \max_j x_j" /> makes the largest exponent
        exactly 0, so no term can overflow. Terms far below the max underflow to 0 instead — which is
        harmless, because they were going to round to 0 anyway.</p>
      </>
    ),
    math: () => (
      <>
        <h3>One pass instead of three</h3>
        <p>The safe version reads the data three times: once for the max, once for the sum, once to
        divide. On a GPU, reading memory three times is the expensive part. Online softmax does it in
        one pass by fixing up the running total whenever a new maximum appears.</p>
        <p>Keep a running max <Eq tex="m" /> and running sum <Eq tex="\ell" />. For a new value
        <Eq tex="x" />:</p>
        <Eq block tex="m_{new} = \max(m, x)" />
        <Eq block tex="\ell_{new} = \ell \cdot e^{m - m_{new}} + e^{x - m_{new}}" />
        <p>The factor <Eq tex="e^{m - m_{new}}" /> rescales everything you already summed so it is
        consistent with the new maximum. If the max did not change, that factor is 1 and nothing
        happens.</p>
        <Note kind="key" title="Why this matters far beyond softmax">
          <p>This rescaling trick is exactly what lets Flash Attention process attention in tiles.
          Each tile updates a running max and sum, and the partial outputs get rescaled the same way.
          Without online softmax there is no Flash Attention.</p>
        </Note>
      </>
    ),
    code: () => (
      <>
        <Code lang="python">{`import numpy as np

def softmax_naive(x):
    e = np.exp(x)                 # overflows above ~89 in float32
    return e / e.sum()

def softmax_safe(x):
    x = x - x.max()               # largest exponent is now exactly 0
    e = np.exp(x)
    return e / e.sum()

def softmax_online(x):
    """One pass. This is the shape every attention kernel uses."""
    m, l = -np.inf, 0.0
    for v in x:
        m_new = max(m, v)
        l = l * np.exp(m - m_new) + np.exp(v - m_new)
        m = m_new
    return np.exp(x - m) / l`}</Code>
        <Code lang="python">{`# log-softmax: do not take a log of a softmax, fold it in
def log_softmax(x):
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())

# cross entropy loss straight from logits, never from probabilities
loss = -log_softmax(logits)[target]`}</Code>
        <Note kind="warn" title="Never do this">
          <p><code>np.log(softmax(x))</code> loses precision badly for small probabilities, and gives
          <code>-inf</code> where softmax underflowed to 0. Every framework has a
          <code>log_softmax</code> and a <code>cross_entropy</code> that take raw logits for this reason.
          Using them is not a style preference.</p>
        </Note>
      </>
    ),
    systems: () => (
      <>
        <h3>Softmax is memory bound, always</h3>
        <p>Per element it does one subtract, one exp and one divide — a handful of operations against
        4 or 2 bytes of traffic. The arithmetic intensity is tiny, so the hardware is waiting on memory
        the whole time. Peak FLOPs numbers are meaningless here; bandwidth is everything.</p>
        <p>That is why softmax is almost never run alone in production. It gets fused into the
        operation before it or after it, so the data is touched once while it is already in registers.
        A fused softmax in Triton is one of the standard first kernels people write for exactly this
        reason.</p>
        <h3>Float range, in numbers</h3>
        <table className="w-full font-mono text-[12.5px]">
          <tbody>
            {[['float32', '≈ 3.4e38', 'exp overflows above x ≈ 88.7'],
              ['float16', '≈ 65504', 'exp overflows above x ≈ 11.1'],
              ['bfloat16', '≈ 3.4e38', 'same range as fp32, far less precision']].map(r => (
              <tr key={r[0]} className="border-t border-[var(--line)]">
                <td className="py-1.5 text-[var(--accent)]">{r[0]}</td>
                <td className="py-1.5 text-[#3e392f]">{r[1]}</td>
                <td className="py-1.5 text-[var(--dim)]">{r[2]}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p>float16 overflows at a logit of 11. This is the main reason attention and softmax
        accumulate in fp32 even when the weights are fp16, and a large part of why bf16 took over
        for training.</p>
      </>
    ),
    failure: () => (
      <>
        <h3>The NaN that eats your run</h3>
        <p>A NaN in softmax does not stay local. It flows into the attention output, into the residual
        stream, into every later layer, and into every parameter through the gradient. One bad row at
        step 40,000 and the whole model is NaN by step 40,001.</p>
        <ul>
          <li><b>All-masked row</b> — every score is −∞, so you compute <code>-inf - -inf = nan</code>.
            Mask with <code>-1e9</code> rather than true infinity.</li>
          <li><b>fp16 overflow</b> — logits above ~11 are enough. Accumulate softmax in fp32.</li>
          <li><b>Log of a zero probability</b> — use <code>log_softmax</code>, not <code>log(softmax(x))</code>.</li>
          <li><b>Temperature near zero</b> — dividing logits by 0.0 gives ±inf before softmax even starts.</li>
        </ul>
        <Code lang="python">{`# find which layer first produced the NaN
torch.autograd.set_detect_anomaly(True)

# or, cheaper, check as you go
assert torch.isfinite(scores).all(), f"non-finite scores in layer {i}"`}</Code>
      </>
    ),
  },
}
