import QuantizationLab from '../../labs/QuantizationLab'
import { Code, Eq, Note } from '../../components/ui'
import type { Concept } from '../types'

export const quantization: Concept = {
  id: 'quantization',
  title: 'Quantisation',
  tagline: 'Store weights on a coarse grid. The whole game is choosing how many weights share one scale.',
  track: 'gpu',
  difficulty: 4,
  status: 'stable',
  hasLab: true,
  prereqs: ['float-formats', 'decode-cost'],
  papers: [
    { title: 'LLM.int8()', year: 2022, url: 'https://arxiv.org/abs/2208.07339',
      idea: 'A handful of activation channels have huge values and break int8. Keep those in fp16 and quantise the rest.' },
    { title: 'GPTQ', year: 2022, url: 'https://arxiv.org/abs/2210.17323',
      idea: 'Quantise one column at a time and adjust the remaining columns to compensate for the error you just made.' },
    { title: 'SmoothQuant', year: 2022, url: 'https://arxiv.org/abs/2211.10438',
      idea: 'Move the difficulty from activations into weights with a per-channel rescale, since weights are easier to quantise.' },
    { title: 'AWQ: Activation-aware Weight Quantization', year: 2023, url: 'https://arxiv.org/abs/2306.00978',
      idea: 'Not all weights matter equally. Protect the ~1% that multiply large activations and the rest can go to 4 bits.' },
  ],
  lenses: {
    intuition: () => (
      <>
        <p>Weights are stored as 16-bit floats. Most of them are small numbers clustered near zero.
        You do not need 16 bits of detail to say "about 0.03".</p>
        <p>Quantisation picks a grid. With 4 bits you get 16 levels. You choose a scale that maps
        your weights onto those levels, round each weight to the nearest one, and store the level
        number instead of the number itself. Four times smaller.</p>
        <p>The catch is that the scale is set by the largest value in the group. One outlier weight
        that is 20 times bigger than the rest stretches the grid across a range that is mostly empty,
        and every ordinary weight lands on a coarse step.</p>
        <Note kind="key" title="The whole subject in one sentence">
          <p>Quantisation quality is not about bit count. It is about how far apart the biggest and
          smallest value in one scaling group are.</p>
        </Note>
      </>
    ),
    mechanism: () => (
      <>
        <p>Turn outliers on, then switch between per-tensor, per-channel and per-group scaling:</p>
        <QuantizationLab />
        <h3>The three choices that matter</h3>
        <ul>
          <li><b>How many bits</b> — 8 is nearly free, 4 is the usual sweet spot, 3 and below needs
            real care.</li>
          <li><b>How many weights share a scale</b> — one per tensor is cheapest to store and worst
            for quality. One per group of 64 or 128 is what almost everything uses.</li>
          <li><b>Weights only, or activations too</b> — weight-only quantisation is easy because the
            weights are fixed and you can look at them. Activations change with every input and have
            far worse outliers.</li>
        </ul>
      </>
    ),
    math: () => (
      <>
        <h3>Symmetric quantisation</h3>
        <Eq block tex="s = \frac{\max|w|}{2^{b-1} - 1}, \qquad q = \text{round}\!\left(\frac{w}{s}\right), \qquad \hat{w} = q \cdot s" />
        <p>The error for each weight is at most <Eq tex="s/2" />, so the error is set directly by the
        largest absolute value in the group. That is the mathematical statement of the outlier
        problem.</p>
        <h3>Why smaller groups help so much</h3>
        <p>With group size <Eq tex="g" />, the scale is the max over <Eq tex="g" /> values instead of
        over the whole tensor. The cost is one extra scale per group:</p>
        <Eq block tex="\text{bits per weight} = b + \frac{16}{g}" />
        <p>At 4 bits with group size 128 that is 4.125 bits per weight — the bookkeeping is almost
        free, and the quality difference is large.</p>
        <h3>Why activations are harder</h3>
        <p>Transformer activations develop a small number of channels with values 100 times larger
        than the rest, and they appear in the same channels consistently. Per-tensor activation
        quantisation is destroyed by them. SmoothQuant divides activations by a per-channel factor
        and multiplies the weights by it, moving the problem to where it is easier:</p>
        <Eq block tex="Y = (X \oslash s)(\text{diag}(s) W)" />
      </>
    ),
    code: () => (
      <>
        <Code lang="python">{`import torch

def quantize_groupwise(w, bits=4, group=128):
    """Symmetric, per-group. This is what almost every 4-bit format does."""
    shape = w.shape
    w = w.reshape(-1, group)

    levels = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=1, keepdim=True) / levels
    scale = scale.clamp(min=1e-8)

    q = (w / scale).round().clamp(-levels - 1, levels)
    return q.to(torch.int8), scale, shape

def dequantize(q, scale, shape):
    return (q.float() * scale).reshape(shape)

# measure what you lost, per channel -- averages hide the damage
w = model.layers[0].mlp.down_proj.weight
q, s, shape = quantize_groupwise(w, bits=4, group=128)
err = (w - dequantize(q, s, shape)).abs()
print(err.mean(dim=1).topk(5))      # the channels that got hurt`}</Code>
        <Note kind="warn" title="Perplexity is not enough">
          <p>A 4-bit model often shows almost no perplexity change while being noticeably worse at
          long-context recall, code and multi-step reasoning. Evaluate on the tasks you care about,
          not on the metric that is easy to compute.</p>
        </Note>
      </>
    ),
    systems: () => (
      <>
        <h3>The speedup comes from bytes, not maths</h3>
        <p>Decode is memory bound. Reading 4-bit weights instead of 16-bit means a quarter of the
        traffic, so roughly four times the tokens per second — as long as the dequantisation happens
        inside the kernel. If dequantisation runs as a separate step that writes bf16 back to memory,
        you move <em>more</em> bytes than before and gain nothing.</p>
        <h3>Where each method fits</h3>
        <table className="w-full text-[13px]">
          <tbody>
            {[['GPTQ', 'Weight-only, 3-4 bit, needs a calibration set. Good quality, slow to produce.'],
              ['AWQ', 'Weight-only, 4 bit, protects salient channels. Fast kernels, widely deployed.'],
              ['bitsandbytes NF4', 'Weight-only 4 bit, no calibration. What QLoRA uses.'],
              ['SmoothQuant', 'Weights and activations at int8. Needs the per-channel rescale folded in.'],
              ['FP8', 'Hardware-native on Hopper and later. Fast, and usable for training as well.']].map(r => (
              <tr key={r[0]} className="border-t border-[var(--line)] align-top">
                <td className="py-2 pr-4 font-medium text-[#3e392f]">{r[0]}</td>
                <td className="py-2 text-[var(--dim)]">{r[1]}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <h3>The KV cache can be quantised too</h3>
        <p>At long context the cache is bigger than the weights. int8 KV halves it. The error
        accumulates over the sequence though, so quality falls off with length — a common compromise
        is to keep the most recent tokens at full precision.</p>
      </>
    ),
    failure: () => (
      <>
        <table className="w-full text-[13px]">
          <tbody>
            {[['int8 destroyed the model, int8 weight-only was fine', 'Activation outliers. Use SmoothQuant or keep outlier channels in higher precision.'],
              ['4-bit model is not faster', 'Dequantisation running as a separate kernel. Check that it is fused.'],
              ['Perplexity barely moved but the model got noticeably dumber', 'Perplexity is insensitive to exactly the failures quantisation causes. Evaluate on real tasks.'],
              ['Quality fell off a cliff below 4 bits', 'Expected without a method that compensates. GPTQ or AWQ, and a smaller group size.'],
              ['Works on short prompts, degrades on long ones', 'KV cache quantisation. Error accumulates with sequence length.'],
              ['One layer ruins everything', 'Usually the first or last. Many recipes deliberately keep those at higher precision.']].map(r => (
              <tr key={r[0]} className="border-t border-[var(--line)] align-top">
                <td className="py-2 pr-4 text-[#3e392f]">{r[0]}</td>
                <td className="py-2 text-[var(--dim)]">{r[1]}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <Note kind="warn" title="Try it in the lab">
          <p>Outliers on, per-tensor, 4 bits. Then switch to per-group. Same bit count, same weights,
          and the error drops by a large factor — all from changing how many weights share a scale.</p>
        </Note>
      </>
    ),
  },
}
