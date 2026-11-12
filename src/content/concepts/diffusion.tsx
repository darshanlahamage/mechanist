import DiffusionLab from '../../labs/DiffusionLab'
import { Code, Eq, Note } from '../../components/ui'
import type { Concept } from '../types'

export const diffusionForward: Concept = {
  id: 'diffusion-forward',
  title: 'Diffusion: the forward process',
  tagline: 'Destroying an image is one line of algebra. The network only learns to undo one step of it.',
  track: 'diffusion',
  difficulty: 3,
  status: 'stable',
  hasLab: true,
  papers: [
    { title: 'Denoising Diffusion Probabilistic Models', year: 2020, url: 'https://arxiv.org/abs/2006.11239',
      idea: 'Train a network to predict the noise instead of the image. The loss collapses to a plain mean squared error.' },
    { title: 'Improved Denoising Diffusion Probabilistic Models', year: 2021, url: 'https://arxiv.org/abs/2102.09672',
      idea: 'The linear noise schedule wastes the back half of training. A cosine schedule fixes it.' },
    { title: 'Denoising Diffusion Implicit Models', year: 2020, url: 'https://arxiv.org/abs/2010.02502',
      idea: 'The reverse process does not have to be random. Make it deterministic and you can skip most steps.' },
  ],
  lenses: {
    intuition: () => (
      <>
        <p>Generating an image from nothing is hard. Removing a little noise from a slightly noisy
        image is easy. Diffusion turns the first problem into many copies of the second.</p>
        <p>Training works backwards from real data. Take a real image, add a measured amount of noise,
        and ask the network one question: <b>what noise did I just add?</b> That is a plain regression
        problem with a known answer, so there is nothing clever in the loss at all.</p>
        <p>To generate, start from pure noise and ask that question over and over, removing a bit of
        the predicted noise each time. After enough steps you are left with an image.</p>
        <Note kind="key" title="The thing that surprises people">
          <p>The forward noising process has no learned parameters. It is fixed algebra, decided
          before training starts. The network never sees it — it only ever sees a noisy image and a
          timestep, and guesses the noise.</p>
        </Note>
      </>
    ),
    mechanism: () => (
      <>
        <p>Noise is added gradually, with a schedule <Eq tex="\beta_t" /> that says how much at each
        step. Define:</p>
        <Eq block tex="\alpha_t = 1 - \beta_t, \qquad \bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s" />
        <p>The useful part is that you never have to run the steps one by one. You can jump straight
        to any timestep in closed form:</p>
        <Eq block tex="x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)" />
        <p>So <Eq tex="\sqrt{\bar\alpha_t}" /> is how much of the picture is left, and
        <Eq tex="\sqrt{1-\bar\alpha_t}" /> is how much noise is on top. Training picks a random
        <Eq tex="t" /> per example, builds <Eq tex="x_t" /> with that line, and asks the model for
        <Eq tex="\epsilon" />.</p>
        <p>Rearranging the same line gives you the image back from the predicted noise:</p>
        <Eq block tex="\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\hat{\epsilon}}{\sqrt{\bar{\alpha}_t}}" />
        <p>Slide the timestep, then add some error to the model's noise prediction and watch what
        happens to the reconstruction at high t:</p>
        <DiffusionLab />
      </>
    ),
    math: () => (
      <>
        <h3>Why the closed form exists</h3>
        <p>Each step adds Gaussian noise:</p>
        <Eq block tex="q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(\sqrt{\alpha_t}\,x_{t-1},\ (1-\alpha_t)I\right)" />
        <p>Sums of independent Gaussians are Gaussian, and the variances add. Compose <Eq tex="t" />
        steps and the means multiply while the variances accumulate, which gives exactly:</p>
        <Eq block tex="q(x_t \mid x_0) = \mathcal{N}\!\left(\sqrt{\bar{\alpha}_t}\,x_0,\ (1-\bar{\alpha}_t)I\right)" />
        <p>This is the only reason diffusion training is cheap. Without it you would have to simulate
        hundreds of steps for every training example.</p>

        <h3>The loss, after all the cancellation</h3>
        <p>The proper derivation goes through a variational bound with KL terms between Gaussians.
        Once you substitute the closed forms and drop constants, almost everything cancels and you are
        left with:</p>
        <Eq block tex="L = \mathbb{E}_{t, x_0, \epsilon}\left[\left\| \epsilon - \epsilon_\theta(x_t, t) \right\|^2\right]" />
        <p>Mean squared error against the noise you added. That is the entire training objective.</p>

        <h3>Error amplification</h3>
        <p>Look at the reconstruction formula again. The denominator is <Eq tex="\sqrt{\bar\alpha_t}" />,
        which goes to nearly zero at high <Eq tex="t" />. An error <Eq tex="\delta" /> in the predicted
        noise becomes an error in the image of:</p>
        <Eq block tex="\frac{\sqrt{1-\bar{\alpha}_t}}{\sqrt{\bar{\alpha}_t}}\,\delta" />
        <p>At t = 900 on a cosine schedule that factor is well over 20. This is the mathematical
        reason you cannot jump from pure noise to a finished image in one step, and why every fast
        sampler is a careful argument about controlling this amplification.</p>
      </>
    ),
    code: () => (
      <>
        <Code lang="python">{`import torch

def cosine_alpha_bar(T, s=0.008):
    t = torch.arange(T + 1) / T
    f = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
    return f / f[0]

alpha_bar = cosine_alpha_bar(1000)

def training_step(model, x0):
    B = x0.shape[0]
    t = torch.randint(0, 1000, (B,), device=x0.device)      # random timestep per image
    eps = torch.randn_like(x0)

    a = alpha_bar[t].view(-1, 1, 1, 1)
    x_t = a.sqrt() * x0 + (1 - a).sqrt() * eps              # the whole forward process

    eps_pred = model(x_t, t)
    return ((eps - eps_pred) ** 2).mean()                   # that is the loss`}</Code>
        <Code lang="python">{`@torch.no_grad()
def ddim_sample(model, shape, steps=50, eta=0.0):
    """Deterministic sampling. Skips most of the 1000 steps."""
    x = torch.randn(shape)
    ts = torch.linspace(999, 0, steps).long()

    for i, t in enumerate(ts):
        a_t = alpha_bar[t]
        a_prev = alpha_bar[ts[i + 1]] if i + 1 < steps else torch.tensor(1.0)

        eps = model(x, t.expand(shape[0]))
        x0_pred = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)                      # stops error blowing up

        x = a_prev.sqrt() * x0_pred + (1 - a_prev).sqrt() * eps
    return x`}</Code>
        <Note kind="info" title="The clamp is not cosmetic">
          <p>That <code>clamp(-1, 1)</code> is doing real work. Images live in [-1, 1], so anything
          outside is known to be wrong. Clipping it stops amplified error from feeding into the next
          step. Remove it and early steps produce saturated garbage that never recovers.</p>
        </Note>
      </>
    ),
    systems: () => (
      <>
        <h3>Cost is steps × model, and the model runs twice</h3>
        <p>Generation cost is the number of sampling steps times one forward pass. With classifier-free
        guidance you run the model twice per step — once conditioned, once unconditioned — so the real
        cost is <code>2 × steps</code> forward passes. 50 steps with guidance is 100 passes for one
        image.</p>
        <p>This is why so much work goes into cutting steps: DDIM took 1000 down to 50, distillation
        methods took 50 down to 4, and adversarial distillation reaches 1. Each one is trading quality
        for a direct multiple of the cost.</p>
        <h3>Latent diffusion changes the arithmetic</h3>
        <p>Running diffusion on 1024×1024 pixels is expensive because attention over a million
        positions is hopeless. Latent diffusion compresses to 128×128 with a VAE first — 64× fewer
        positions — runs diffusion there, then decodes once at the end. Nearly all modern image models
        work this way, and the VAE decode is often a noticeable part of total time at low step counts.</p>
        <h3>Unlike LLM decoding, this is compute bound</h3>
        <p>Each step is a full forward pass over a convolution or transformer stack with real batch
        dimensions in the spatial axes. The math units are busy, not idle. So the levers are different
        from LLM serving: batching helps less, and lower precision or fewer steps helps more.</p>
      </>
    ),
    failure: () => (
      <>
        <table className="w-full text-[13px]">
          <thead><tr className="text-[var(--dim)]">
            <th className="py-2 text-left font-normal">Symptom</th>
            <th className="py-2 text-left font-normal">Cause</th>
          </tr></thead>
          <tbody>
            {[['Outputs are grey mush', 'Too few steps for the schedule, or x0 prediction not clamped.'],
              ['Oversaturated, burned-looking colours', 'Guidance scale too high. It pushes x0 far outside the valid range.'],
              ['Training loss drops but samples stay noise', 'Timestep embedding not actually reaching the network.'],
              ['Good at 512px, broken at 1024px', 'Signal-to-noise ratio shifts with resolution. The schedule needs shifting too.'],
              ['Last third of training does nothing', 'Linear schedule. Those steps are already pure noise. Switch to cosine.'],
              ['Fine detail missing in every sample', 'VAE reconstruction limit, not the diffusion model. Check the VAE alone first.']].map(r => (
              <tr key={r[0]} className="border-t border-[var(--line)] align-top">
                <td className="py-2 pr-4 text-[#c7cdd8]">{r[0]}</td>
                <td className="py-2 text-[var(--dim)]">{r[1]}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <Note kind="warn" title="Try it in the lab">
          <p>Set the timestep to 950 and raise the model's noise error to 0.2. The reconstruction
          falls apart completely, while the same error at t=200 is barely visible. Same error, wildly
          different outcome — that is the amplification factor doing it.</p>
        </Note>
      </>
    ),
  },
}
