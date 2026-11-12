import LoRALab from '../../labs/LoRALab'
import { Code, Eq, Note } from '../../components/ui'
import type { Concept } from '../types'

export const lora: Concept = {
  id: 'lora',
  title: 'LoRA and QLoRA',
  tagline: 'Do not change the weights. Add a thin correction to them, and only train that.',
  track: 'posttraining',
  difficulty: 3,
  status: 'stable',
  hasLab: true,
  papers: [
    { title: 'LoRA: Low-Rank Adaptation of Large Language Models', year: 2021,
      url: 'https://arxiv.org/abs/2106.09685',
      idea: 'The change fine-tuning makes to a weight matrix has low rank, so represent it as a product of two thin matrices and train only those.' },
    { title: 'QLoRA: Efficient Finetuning of Quantized LLMs', year: 2023,
      url: 'https://arxiv.org/abs/2305.14314',
      idea: 'Freeze the base model in 4-bit and train the adapters in bf16. A 65B model fine-tunes on a single 48 GB GPU.' },
    { title: 'DoRA: Weight-Decomposed Low-Rank Adaptation', year: 2024,
      url: 'https://arxiv.org/abs/2402.09353',
      idea: 'Split the weight into magnitude and direction, and apply the low-rank update only to the direction.' },
  ],
  lenses: {
    intuition: () => (
      <>
        <p>Full fine-tuning updates every weight. For a 70B model that means 70B gradients, 70B
        momentum values, 70B variance values — hundreds of gigabytes before you have stored a single
        activation.</p>
        <p>LoRA starts from an observation: the <b>change</b> fine-tuning makes to a weight matrix is
        usually close to low rank. The model is not learning a whole new matrix, it is learning a
        small adjustment that lives in a few directions.</p>
        <p>So freeze the original weight and learn two thin matrices beside it. If the weight is
        4096×4096 and you pick rank 16, you train 131k values instead of 16.8M — under 1%.</p>
        <Eq block tex="W' = W + BA, \qquad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k}" />
      </>
    ),
    mechanism: () => (
      <>
        <p>Slide the rank and see how much of a real weight change gets captured. Then switch the
        change to full-rank noise and watch LoRA stop working:</p>
        <LoRALab />
        <h3>Why B starts at zero</h3>
        <p>A is initialised randomly and B is initialised to zeros, so <Eq tex="BA = 0" /> at step
        zero and the model behaves exactly like the base model. Training then moves away from that
        point gradually. Initialising both randomly would shove the model somewhere arbitrary before
        it has learned anything.</p>
        <h3>Where to attach it</h3>
        <p>The original paper only adapted the attention projections. Later work found adapting all
        linear layers, including the feed-forward ones, works better for the same parameter budget —
        which is what most tooling now does by default.</p>
      </>
    ),
    math: () => (
      <>
        <h3>Forward pass</h3>
        <Eq block tex="h = Wx + \frac{\alpha}{r} BAx" />
        <p>The <Eq tex="\alpha/r" /> factor exists so that changing the rank does not change the
        effective size of the update. Without it, doubling the rank would roughly double the
        magnitude of the correction and you would have to retune the learning rate every time.</p>
        <h3>What rank can represent</h3>
        <p>Any matrix has a singular value decomposition:</p>
        <Eq block tex="\Delta W = \sum_{i=1}^{\min(d,k)} \sigma_i u_i v_i^\top" />
        <p>A rank-r product can represent exactly the first r terms, and that truncation is provably
        the best possible rank-r approximation in Frobenius norm. So the question is only whether
        the singular values drop off quickly. For behaviour changes — tone, format, instruction
        following — they do. For new factual knowledge they do not, which is the real limit.</p>
        <h3>Memory saved</h3>
        <p>Optimizer state scales with trainable parameters, not total parameters:</p>
        <Eq block tex="\text{Adam state} = 12 \text{ bytes} \times r(d + k) \text{ instead of } 12 \times dk" />
        <p>At rank 16 on a 4096×4096 matrix that is 1.6 MB instead of 201 MB.</p>
      </>
    ),
    code: () => (
      <>
        <Code lang="python">{`import math, torch, torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False              # the original weight never moves

        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # starts at zero
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.drop(x) @ self.A.T @ self.B.T * self.scale

    def merge(self):
        """Fold the adapter into the base weight. Zero overhead afterwards."""
        self.base.weight.data += (self.B @ self.A) * self.scale`}</Code>
        <Code lang="python">{`from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,                   # scale = alpha / r = 2
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],   # all of them, not just attention
    lora_dropout=0.05,
)
model = get_peft_model(base_model, config)
model.print_trainable_parameters()   # trainable: 0.62% of total`}</Code>
      </>
    ),
    systems: () => (
      <>
        <h3>Training memory, not training speed</h3>
        <p>LoRA saves optimizer state and gradient memory. It does <em>not</em> make the forward pass
        faster — you still run the full base model, plus two extra small matmuls. A LoRA step is
        slightly slower per token than a frozen forward pass, and much lighter on memory.</p>
        <h3>QLoRA</h3>
        <p>Quantise the frozen base to 4-bit (it never gets updated, so quantisation error is fixed
        and tolerable) and keep the adapters in bf16. The base weights are dequantised on the fly
        inside the kernel. This is what makes single-GPU fine-tuning of large models possible.</p>
        <h3>Serving many adapters at once</h3>
        <p>Because the base is shared, one GPU can serve hundreds of different fine-tunes. Either
        merge the adapter into the weights for zero overhead (one model per deployment), or keep it
        separate and batch requests for different adapters together (many tenants, slightly slower).
        Most serving stacks support the second.</p>
      </>
    ),
    failure: () => (
      <>
        <table className="w-full text-[13px]">
          <tbody>
            {[['Learns the format but not the facts', 'Expected. Low rank captures behaviour changes well and new knowledge poorly. Use RAG or full fine-tuning for facts.'],
              ['Raising the rank did not help', 'The singular values already flattened. Check where the captured-change curve levels off before paying for more rank.'],
              ['Merged adapter behaves differently from the unmerged one', 'Dropout is active in the unmerged path during training only, or the scale was applied twice.'],
              ['Training loss will not go down', 'Only attention projections adapted. Add the feed-forward layers.'],
              ['QLoRA result much worse than LoRA', 'Quantisation of the base is too aggressive, or a layer that should stay in higher precision got quantised.'],
              ['Stacking two adapters gives nonsense', 'They were each trained against the unmodified base. Their updates are not designed to compose.']].map(r => (
              <tr key={r[0]} className="border-t border-[var(--line)] align-top">
                <td className="py-2 pr-4 text-[#c7cdd8]">{r[0]}</td>
                <td className="py-2 text-[var(--dim)]">{r[1]}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <Note kind="warn" title="Try it in the lab">
          <p>Switch the update type to "full rank (noise)". The captured-change number stays low no
          matter how far you push the rank. That is the picture of LoRA failing to teach a model
          something genuinely new.</p>
        </Note>
      </>
    ),
  },
}
