import OptimizerLab from '../../labs/OptimizerLab'
import { Code, Eq, Note } from '../../components/ui'
import type { Concept } from '../types'

export const optimizers: Concept = {
  id: 'optimizers',
  title: 'Optimizers',
  tagline: 'Every one of them is a rule for turning a gradient into a step. The differences are about scale and memory.',
  track: 'training',
  difficulty: 3,
  status: 'stable',
  hasLab: true,
  prereqs: ['optimization-basics'],
  papers: [
    { title: 'Adam: A Method for Stochastic Optimization', year: 2014, url: 'https://arxiv.org/abs/1412.6980',
      idea: 'Keep a running average of the gradient and of its square, and divide one by the other. Each parameter gets its own effective step size.' },
    { title: 'Decoupled Weight Decay Regularization', year: 2017, url: 'https://arxiv.org/abs/1711.05101',
      idea: 'Adam was applying weight decay through the adaptive denominator, which is not weight decay. Apply it to the weights directly instead.' },
    { title: 'Muon: An optimizer for hidden layers', year: 2024, url: 'https://arxiv.org/abs/2502.16982',
      idea: 'Treat weight matrices as matrices. Orthogonalise the momentum update rather than scaling it element by element.' },
  ],
  lenses: {
    intuition: () => (
      <>
        <p>Gradient descent says: move downhill. Everything else is dealing with the fact that
        "downhill" points in a direction whose length you cannot trust.</p>
        <p>Real loss surfaces are stretched. Some directions are steep and some are nearly flat, and
        the same step size cannot suit both. Too big and you bounce off the steep walls; too small
        and you crawl along the flat floor.</p>
        <p><b>Momentum</b> fixes this by averaging recent gradients, so the bouncing cancels and the
        consistent direction builds up. <b>Adam</b> goes further and divides each direction by how
        big its gradients have recently been, so every parameter gets a step size that suits it.</p>
        <Note kind="key" title="What Adam actually costs">
          <p>Two extra numbers per parameter, in fp32. For a 7B model that is 56 GB of optimizer
          state before you store a single gradient. Adam is not free — it is bought with memory.</p>
        </Note>
      </>
    ),
    mechanism: () => (
      <>
        <p>Same surface, same start, same learning rate. Switch to the ravine and watch SGD bounce:</p>
        <OptimizerLab />
        <h3>What each one is doing</h3>
        <ul>
          <li><b>SGD</b> — step straight along the gradient. Direction is right, scale is not.</li>
          <li><b>Momentum</b> — accumulate a velocity. Oscillations cancel, consistent directions add.</li>
          <li><b>RMSProp</b> — divide by the running size of each direction's gradients. Fixes scale,
            ignores direction history.</li>
          <li><b>Adam</b> — both at once: momentum on the gradient, and division by the running
            square.</li>
          <li><b>AdamW</b> — Adam, but weight decay applied directly to the weights rather than
            pushed through the adaptive denominator. This is what everyone actually uses.</li>
        </ul>
      </>
    ),
    math: () => (
      <>
        <h3>Adam, written out</h3>
        <Eq block tex="m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t" />
        <Eq block tex="v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2" />
        <Eq block tex="\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}" />
        <Eq block tex="\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}" />
        <p>The bias correction matters at the start. Both averages begin at zero, so for the first
        few hundred steps they underestimate badly. Without correction the early steps are far too
        small, which is a large part of why warmup exists.</p>
        <h3>The decoupling fix</h3>
        <p>Original Adam added weight decay into the gradient:</p>
        <Eq block tex="g_t \leftarrow g_t + \lambda\theta \quad \text{then divided by } \sqrt{\hat v_t}" />
        <p>So parameters with large gradients got less decay — the opposite of what you want. AdamW
        applies it separately:</p>
        <Eq block tex="\theta_t = \theta_{t-1} - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\theta_{t-1}\right)" />
        <h3>Memory, per parameter</h3>
        <table className="w-full font-mono text-[12.5px]">
          <tbody>
            {[['SGD', '0 bytes', 'nothing to keep'],
              ['SGD + momentum', '4 bytes', 'one velocity'],
              ['Adam / AdamW', '8 bytes', 'two states, both fp32'],
              ['Adafactor', '~0.5 bytes', 'factored second moment'],
              ['8-bit Adam', '2 bytes', 'quantised states']].map(r => (
              <tr key={r[0]} className="border-t border-[var(--line)]">
                <td className="py-1.5 text-[var(--accent)]">{r[0]}</td>
                <td className="py-1.5 text-[#3e392f]">{r[1]}</td>
                <td className="py-1.5 text-right text-[var(--dim)]">{r[2]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </>
    ),
    code: () => (
      <>
        <Code lang="python">{`def adamw_step(p, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
    m = b1 * m + (1 - b1) * g           # average of the gradient
    v = b2 * v + (1 - b2) * g * g       # average of its square

    m_hat = m / (1 - b1 ** t)           # bias correction -- matters early
    v_hat = v / (1 - b2 ** t)

    p = p - lr * (m_hat / (v_hat ** 0.5 + eps) + wd * p)
    return p, m, v`}</Code>
        <Code lang="python">{`import torch

# do not decay biases and norm weights -- they are not the thing weight decay is for
decay, no_decay = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    (no_decay if p.ndim < 2 else decay).append(p)

opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.1},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=3e-4, betas=(0.9, 0.95), eps=1e-8,   # beta2=0.95 is common for LLMs
)`}</Code>
        <Note kind="info" title="beta2 = 0.95, not 0.999">
          <p>Large language model training usually lowers beta2. The default 0.999 averages over
          roughly a thousand steps, which reacts too slowly when the loss spikes. 0.95 averages over
          about twenty and recovers faster.</p>
        </Note>
      </>
    ),
    systems: () => (
      <>
        <h3>Optimizer state is usually the biggest thing you hold</h3>
        <p>Weights in bf16 are 2 bytes per parameter. Adam state plus an fp32 master copy is 12.
        The optimizer is six times the model. That is what ZeRO stage 1 shards first, because it is
        the cheapest large win available.</p>
        <h3>The step itself is memory bound</h3>
        <p>Reading and writing several arrays the size of the model to do a handful of operations
        each. Fused and multi-tensor implementations exist for this reason, and they matter more at
        small model sizes where the step is a visible fraction of the time.</p>
        <h3>Newer optimizers trade differently</h3>
        <p>Muon and Shampoo treat weight matrices as matrices rather than as flat lists of numbers,
        which gives better-conditioned updates at the cost of extra work per step. They have shown
        real speedups in wall-clock time to a loss target, which is the measure that matters —
        not steps.</p>
      </>
    ),
    failure: () => (
      <>
        <table className="w-full text-[13px]">
          <tbody>
            {[['Loss explodes in the first few hundred steps', 'No warmup. Adam\'s second moment estimate is unreliable early, so the effective step is huge.'],
              ['Training fine then suddenly diverges', 'A large gradient poisoned the second moment. Lower beta2 and clip gradients.'],
              ['Weight decay seems to do nothing', 'Using Adam rather than AdamW, so decay is being divided away by the adaptive denominator.'],
              ['Out of memory with Adam, fine with SGD', 'Expected. 8 extra bytes per parameter. Try 8-bit Adam, Adafactor, or ZeRO.'],
              ['Switching optimizer made everything worse', 'The learning rate does not transfer. SGD and Adam need learning rates orders of magnitude apart.'],
              ['Model trains but will not generalise', 'Decay applied to biases and norm parameters. Exclude anything with fewer than 2 dimensions.']].map(r => (
              <tr key={r[0]} className="border-t border-[var(--line)] align-top">
                <td className="py-2 pr-4 text-[#3e392f]">{r[0]}</td>
                <td className="py-2 text-[var(--dim)]">{r[1]}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <Note kind="warn" title="Try it in the lab">
          <p>Pick the ravine and raise the learning rate until SGD diverges. Adam and RMSProp carry
          on, because they rescale the steep direction down. Then switch to the plain bowl — now
          everything works and Adam's extra memory bought you nothing.</p>
        </Note>
      </>
    ),
  },
}
