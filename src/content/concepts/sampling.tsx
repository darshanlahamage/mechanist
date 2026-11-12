import SamplingLab from '../../labs/SamplingLab'
import { Code, Eq, Note } from '../../components/ui'
import type { Concept } from '../types'

export const sampling: Concept = {
  id: 'sampling',
  title: 'Sampling the next token',
  tagline: 'The model gives you a distribution. How you pick from it changes the output more than most prompt tweaks.',
  track: 'foundations',
  difficulty: 2,
  status: 'stable',
  hasLab: true,
  prereqs: ['softmax-numerics'],
  papers: [
    { title: 'The Curious Case of Neural Text Degeneration', year: 2019, url: 'https://arxiv.org/abs/1904.09751',
      idea: 'Introduced top-p. Maximising likelihood gives repetitive text; cutting the unreliable tail works better than cutting a fixed number of options.' },
    { title: 'Locally Typical Sampling', year: 2022, url: 'https://arxiv.org/abs/2202.00666',
      idea: 'Pick tokens whose surprise is close to the expected surprise, rather than the most likely ones.' },
  ],
  lenses: {
    intuition: () => (
      <>
        <p>A language model does not output a word. It outputs a score for every token in its
        vocabulary. Softmax turns those into probabilities. Something then has to pick one.</p>
        <p>Always picking the highest gives repetitive, flat text — the model gets into a loop
        because the safest next word keeps being the safest next word. Picking randomly in
        proportion to probability gives more natural text, but occasionally picks something from
        the long tail that makes no sense.</p>
        <p>So the common strategies all do one of two things: reshape the distribution
        (temperature), or cut off the unreliable tail (top-k, top-p, min-p).</p>
      </>
    ),
    mechanism: () => (
      <>
        <p>Same logits, every strategy, side by side:</p>
        <SamplingLab />
        <h3>What each control does</h3>
        <ul>
          <li><b>Temperature</b> divides the logits before softmax. Below 1 sharpens the
            distribution, above 1 flattens it. It reshapes everything, including the tail.</li>
          <li><b>Top-k</b> keeps the k highest and renormalises. Fixed count, regardless of how
            confident the model is.</li>
          <li><b>Top-p (nucleus)</b> keeps the smallest set whose probabilities add up to p. When
            the model is confident that might be 1 token; when it is unsure it might be 200.</li>
          <li><b>Min-p</b> keeps anything above a fraction of the top token's probability. Adapts
            like top-p but is cheaper and behaves better at high temperature.</li>
        </ul>
        <p>They compose. A typical setting applies temperature first, then a cutoff, then
        renormalises what is left.</p>
      </>
    ),
    math: () => (
      <>
        <h3>Temperature</h3>
        <Eq block tex="p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}" />
        <p>As <Eq tex="T \to 0" /> this becomes one-hot on the largest logit, which is greedy
        decoding. As <Eq tex="T \to \infty" /> it becomes uniform. Note it is applied to the logits,
        before softmax — applying it after would not give the same result.</p>
        <h3>Top-p, written out</h3>
        <p>Sort probabilities descending. Keep the smallest prefix such that:</p>
        <Eq block tex="\sum_{i \in S} p_i \geq p" />
        <p>then renormalise over <Eq tex="S" />. The adaptive size is the point: it keeps many
        options when the model is genuinely unsure and very few when it is not.</p>
        <h3>Why temperature and cutoffs are not interchangeable</h3>
        <p>Temperature changes the <em>ratios</em> between all tokens. A cutoff leaves ratios alone
        and only removes options. Raising temperature makes a bad token more likely relative to a
        good one; top-p removes the bad token entirely while leaving the good ones in proportion.
        That is why people use both.</p>
      </>
    ),
    code: () => (
      <>
        <Code lang="python">{`import torch

def sample(logits, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0):
    if temperature <= 0:
        return logits.argmax(-1)              # greedy

    logits = logits / temperature

    if top_k > 0:
        kth = torch.topk(logits, top_k).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if min_p > 0:
        probs = logits.softmax(-1)
        floor = probs.max(-1, keepdim=True).values * min_p
        logits = logits.masked_fill(probs < floor, float("-inf"))

    if top_p < 1.0:
        sorted_logits, idx = logits.sort(descending=True)
        cum = sorted_logits.softmax(-1).cumsum(-1)
        remove = cum - sorted_logits.softmax(-1) > top_p   # keep the one that crosses p
        logits = logits.masked_fill(remove.scatter(-1, idx, remove), float("-inf"))

    return torch.multinomial(logits.softmax(-1), 1)`}</Code>
        <Note kind="info" title="Repetition penalties are a different thing">
          <p>Frequency and presence penalties subtract from the logits of tokens already used. They
          are a blunt fix for looping, and they damage text that legitimately repeats — code,
          tables, lists, names. Reach for better sampling settings first.</p>
        </Note>
      </>
    ),
    systems: () => (
      <>
        <h3>Sampling is cheap, but not free</h3>
        <p>The logits tensor is <Eq tex="\text{batch} \times V" />, and with a 128k vocabulary a
        sort for top-p over a large batch is a real kernel, not a rounding error. Serving stacks fuse
        the whole sampling pipeline into one kernel and avoid the full sort where they can — top-k
        selection does not need a complete ordering.</p>
        <h3>It affects reproducibility</h3>
        <p>Temperature 0 is not always deterministic in practice. Batching changes reduction order,
        which changes the logits by tiny amounts, which can flip which token is highest when two are
        nearly tied. If you need reproducible output, fix the seed, fix the batch composition, and
        expect it to still drift across library versions.</p>
        <h3>Structured output changes the picture</h3>
        <p>When output has to be valid JSON or match a grammar, a mask is applied to the logits
        before sampling so invalid tokens are impossible. That mask has to be recomputed every step
        from the current parser state, which is why constrained decoding has a real cost.</p>
      </>
    ),
    failure: () => (
      <>
        <table className="w-full text-[13px]">
          <tbody>
            {[['Output loops the same phrase', 'Temperature too low, or greedy. The model is stuck in its own highest-probability path.'],
              ['Text drifts into nonsense mid-paragraph', 'Temperature too high with no cutoff. One unlucky tail token derails everything after it.'],
              ['Code generation has subtle syntax errors', 'Sampling at all. For code, low temperature or greedy is usually correct.'],
              ['Same prompt gives very different quality', 'High variance settings. Measure with several samples before concluding anything about the model.'],
              ['top_p=0.9 behaves differently across libraries', 'Off-by-one in whether the token that crosses p is kept. Both exist in the wild.'],
              ['Tuned settings stopped working after a model update', 'Logit scale changed. Temperature is not comparable across models.']].map(r => (
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
