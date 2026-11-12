import ScalingLawLab from '../../labs/ScalingLawLab'
import { Code, Eq, Note } from '../../components/ui'
import type { Concept } from '../types'

export const scalingLaws: Concept = {
  id: 'scaling-laws',
  title: 'Scaling laws',
  tagline: 'Given a compute budget, how big should the model be and how much data should it see? There is an answer, and it is not "as big as possible".',
  track: 'training',
  difficulty: 4,
  status: 'stable',
  hasLab: true,
  papers: [
    { title: 'Scaling Laws for Neural Language Models', year: 2020, url: 'https://arxiv.org/abs/2001.08361',
      idea: 'Loss follows a clean power law in model size, data and compute. Their conclusion — spend most of a budget on size — turned out to be wrong.' },
    { title: 'Training Compute-Optimal Large Language Models', year: 2022, url: 'https://arxiv.org/abs/2203.15556',
      idea: 'Chinchilla. Redo the experiment with the learning rate schedule matched to each run, and model size and data should scale together, roughly 20 tokens per parameter.' },
    { title: 'Beyond Chinchilla-Optimal', year: 2023, url: 'https://arxiv.org/abs/2401.00448',
      idea: 'Compute-optimal ignores the cost of running the model. If you will serve it a lot, train a smaller one for longer.' },
  ],
  lenses: {
    intuition: () => (
      <>
        <p>You have a fixed number of GPU-hours. You can spend them on a big model that sees little
        data, or a small model that sees a lot. Both extremes are bad.</p>
        <p>Too big, and the model runs out of tokens before it has learned much — it has capacity it
        never fills. Too small, and it sees plenty of data but cannot hold what it reads.</p>
        <p>Chinchilla measured where the balance sits: roughly <b>20 tokens for every parameter</b>.
        GPT-3 was trained at about 1.7 tokens per parameter, which means a model a third of its size
        trained on the same compute would have been better.</p>
        <Note kind="key" title="The part that gets missed">
          <p>The curve near its minimum is very flat. A model three times smaller trained three times
          longer is almost as good — and then costs a third as much every single time you run it.
          Nearly every deployed model is deliberately trained past the compute-optimal point for
          exactly that reason.</p>
        </Note>
      </>
    ),
    mechanism: () => (
      <>
        <p>Set a budget and see how flat the bottom of that curve really is:</p>
        <ScalingLawLab />
        <h3>How these are measured</h3>
        <p>Train many models at many sizes and data amounts, record the final loss for each, and fit
        a curve. The important methodological point from Chinchilla is that each run needs a learning
        rate schedule matched to its own length. Kaplan's earlier work used one schedule across runs,
        which penalised the short ones and made large models look better than they were.</p>
      </>
    ),
    math: () => (
      <>
        <h3>The fitted form</h3>
        <Eq block tex="L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}" />
        <p>With Chinchilla's values: <Eq tex="E = 1.69" />, <Eq tex="A = 406.4" />,
        <Eq tex="\alpha = 0.34" />, <Eq tex="B = 410.7" />, <Eq tex="\beta = 0.28" />.</p>
        <ul>
          <li><Eq tex="E" /> is the floor — the entropy of language itself. No amount of compute
            gets below it.</li>
          <li>The second term is what you lose for having too few parameters.</li>
          <li>The third is what you lose for having seen too little data.</li>
        </ul>
        <h3>The compute constraint</h3>
        <Eq block tex="C \approx 6ND" />
        <p>Six FLOPs per parameter per token: two for the forward multiply-add, four for the
        backward pass. Minimise <Eq tex="L" /> subject to that constraint and both exponents being
        close means <Eq tex="N" /> and <Eq tex="D" /> should grow at about the same rate:</p>
        <Eq block tex="N^* \propto C^{0.5}, \qquad D^* \propto C^{0.5}" />
        <h3>Inference-aware scaling</h3>
        <p>Add the cost of serving. If you expect to generate <Eq tex="D_{inf}" /> tokens over the
        model's life, total cost is:</p>
        <Eq block tex="C_{total} = 6ND_{train} + 2ND_{inf}" />
        <p>Minimising that pushes <Eq tex="N" /> down and <Eq tex="D_{train}" /> up. Llama 3 8B was
        trained on 15T tokens — nearly 2000 tokens per parameter, a hundred times past
        compute-optimal. That was a deliberate choice about inference cost.</p>
      </>
    ),
    code: () => (
      <>
        <Code lang="python">{`import numpy as np
from scipy.optimize import minimize_scalar

E, A, ALPHA, B, BETA = 1.69, 406.4, 0.3392, 410.7, 0.2849

def loss(N, D):
    return E + A / N ** ALPHA + B / D ** BETA

def compute_optimal(C):
    """Best split of a compute budget C = 6ND."""
    def f(log_n):
        N = 10 ** log_n
        return loss(N, C / (6 * N))
    best = minimize_scalar(f, bounds=(7, 13), method="bounded")
    N = 10 ** best.x
    return N, C / (6 * N), f(best.x)

N, D, L = compute_optimal(1e23)
print(f"{N/1e9:.1f}B params, {D/1e12:.1f}T tokens, {D/N:.0f} tokens/param, loss {L:.3f}")`}</Code>
        <Code lang="python">{`# what a budget actually costs
def run_cost(flops, peak=989e12, mfu=0.4, dollars_per_hour=2.5):
    gpu_hours = flops / (peak * mfu) / 3600
    return gpu_hours, gpu_hours * dollars_per_hour

print(run_cost(1e23))     # hours, dollars`}</Code>
      </>
    ),
    systems: () => (
      <>
        <h3>What the numbers mean in practice</h3>
        <p>At 10²³ FLOPs on H100s at 40% MFU, you are looking at roughly 175,000 GPU-hours. On 1024
        GPUs that is about a week, and at typical rental prices a few hundred thousand dollars.
        Scaling laws are how that number gets justified before anyone presses start.</p>
        <h3>Where real runs deviate on purpose</h3>
        <ul>
          <li><b>Inference cost</b> — train smaller and longer, as above.</li>
          <li><b>Data limits</b> — high quality text is finite. Past a few epochs, repeating data
            stops helping.</li>
          <li><b>Memory limits</b> — a compute-optimal model that does not fit your cluster is not
            optimal for you.</li>
          <li><b>Distillation</b> — a small model trained on a big model's outputs beats these
            curves, because the curves assume you are learning from raw text.</li>
        </ul>
        <h3>Hyperparameter transfer</h3>
        <p>μP (maximal update parameterisation) makes the best learning rate roughly independent of
        model width, so you can tune on a small model and transfer the setting to a large one. That
        matters enormously when a single large run costs six figures and you only get one attempt.</p>
      </>
    ),
    failure: () => (
      <>
        <table className="w-full text-[13px]">
          <tbody>
            {[['Predicted loss did not match the run', 'Coefficients were fitted on one dataset and tokenizer. Fit your own on small runs.'],
              ['Lower loss but the model is not better at the task', 'Loss is not capability. The relationship is real but loose, and it breaks per task.'],
              ['Scaled up and quality went backwards', 'Hyperparameters did not transfer. Learning rate and batch size both need rescaling, or use μP.'],
              ['Training on repeated data stopped helping', 'Expected past a few epochs. The data term assumes fresh tokens.'],
              ['Small model beat the predicted curve', 'Data quality or distillation. These laws assume ordinary web text.'],
              ['Kaplan and Chinchilla disagree', 'Chinchilla matched the schedule to each run length. That one methodological fix changed the conclusion.']].map(r => (
              <tr key={r[0]} className="border-t border-[var(--line)] align-top">
                <td className="py-2 pr-4 text-[#3e392f]">{r[0]}</td>
                <td className="py-2 text-[var(--dim)]">{r[1]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </>
    ),
  },
}
