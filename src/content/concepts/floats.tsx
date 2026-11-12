import FloatLab from '../../labs/FloatLab'
import { Code, Eq, Note } from '../../components/ui'
import type { Concept } from '../types'

export const floatFormats: Concept = {
  id: 'float-formats',
  title: 'Float formats',
  tagline: 'Exponent bits buy range. Mantissa bits buy precision. Every format is a bet on which one you need.',
  track: 'foundations',
  difficulty: 2,
  status: 'stable',
  hasLab: true,
  papers: [
    { title: 'Mixed Precision Training', year: 2017, url: 'https://arxiv.org/abs/1710.03740',
      idea: 'Train in fp16 but keep an fp32 master copy and scale the loss, so small gradients do not vanish.' },
    { title: 'FP8 Formats for Deep Learning', year: 2022, url: 'https://arxiv.org/abs/2209.05433',
      idea: 'Two 8-bit formats, one favouring range and one favouring precision, used at different points in the network.' },
  ],
  lenses: {
    intuition: () => (
      <>
        <p>A floating point number is scientific notation in binary: a sign, an exponent, and the
        digits. The format decides how many bits go to each.</p>
        <p>More exponent bits means you can represent very large and very small numbers. More
        mantissa bits means the numbers you can represent are packed closer together.</p>
        <p>float16 and bfloat16 are both 16 bits and they split them differently. float16 has 5
        exponent bits and 10 mantissa bits — precise but it runs out of range at 65504. bfloat16 has
        8 exponent bits and 7 mantissa bits — the same range as float32 but much coarser steps.</p>
        <Note kind="key" title="Why training chose bfloat16">
          <p>Running out of range gives you infinity, and infinity poisons everything downstream.
          Running out of precision gives you a slightly wrong number, which gradient descent does
          not care about. So for training, range is worth more than precision.</p>
        </Note>
      </>
    ),
    mechanism: () => (
      <>
        <p>Set a value and watch the bits. Then look at what each format does with the same number:</p>
        <FloatLab />
        <h3>How a value is stored</h3>
        <Eq block tex="v = (-1)^{s} \times (1 + m/2^{M}) \times 2^{(e - \text{bias})}" />
        <p>The leading 1 is implied and not stored, which buys one extra bit of precision for free.
        The bias lets the exponent field be unsigned while still representing negative exponents.</p>
        <h3>The gap between neighbours</h3>
        <p>Representable numbers are not evenly spaced. Near 1.0 the gap is <Eq tex="2^{-M}" />.
        Near 1000 it is a thousand times larger. This is why adding a tiny number to a large one can
        do nothing at all — the result rounds back to where you started.</p>
      </>
    ),
    math: () => (
      <>
        <h3>The numbers worth knowing</h3>
        <table className="w-full font-mono text-[12.5px]">
          <thead><tr className="text-[var(--dim)]">
            <th className="py-1 text-left font-normal">format</th>
            <th className="py-1 text-right font-normal">bits (s/e/m)</th>
            <th className="py-1 text-right font-normal">max</th>
            <th className="py-1 text-right font-normal">precision</th>
          </tr></thead>
          <tbody>
            {[['float32', '1/8/23', '3.4e38', '~7 digits'],
              ['tf32', '1/8/10', '3.4e38', '~3 digits'],
              ['bfloat16', '1/8/7', '3.4e38', '~2 digits'],
              ['float16', '1/5/10', '65504', '~3 digits'],
              ['fp8 e4m3', '1/4/3', '448', '~1 digit'],
              ['fp8 e5m2', '1/5/2', '57344', '<1 digit']].map(r => (
              <tr key={r[0]} className="border-t border-[var(--line)]">
                <td className="py-1.5 text-[var(--accent)]">{r[0]}</td>
                <td className="py-1.5 text-right text-[#c7cdd8]">{r[1]}</td>
                <td className="py-1.5 text-right text-[#c7cdd8]">{r[2]}</td>
                <td className="py-1.5 text-right text-[var(--dim)]">{r[3]}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <h3>Why summation order matters</h3>
        <p>Floating point addition is not associative. Add a million small numbers to a large
        running total and each addition rounds; add them in pairs and the error is far smaller.
        This is why a reduction on a GPU gives a slightly different answer depending on how many
        threads took part, and why "deterministic" training costs performance.</p>
        <Eq block tex="(a + b) + c \neq a + (b + c)" />
      </>
    ),
    code: () => (
      <>
        <Code lang="python">{`import numpy as np

# the gap between neighbouring representable numbers
print(np.spacing(np.float32(1.0)))      # 1.19e-07
print(np.spacing(np.float32(1000.0)))   # 6.10e-05   <- a thousand times coarser

# adding something smaller than the gap does nothing
x = np.float32(1000.0)
print(x + np.float32(1e-5) == x)        # True

# fp16 runs out of range long before fp32
print(np.float16(70000))                # inf
print(np.float32(70000))                # 70000.0`}</Code>
        <Code lang="python">{`import torch

# bf16 keeps the range, loses the digits
v = torch.tensor([1.0001], dtype=torch.float32)
print(v.to(torch.bfloat16).item())      # 1.0      -- gone
print(v.to(torch.float16).item())       # 1.0      -- also gone at this precision

big = torch.tensor([1e30])
print(big.to(torch.bfloat16).item())    # 1.0e30   -- fine
print(big.to(torch.float16).item())     # inf      -- overflowed`}</Code>
      </>
    ),
    systems: () => (
      <>
        <h3>Bytes moved is the real cost</h3>
        <p>Halving the format halves the memory traffic, and most of deep learning is memory bound.
        That is the actual reason low precision is fast — not the arithmetic, the bytes.</p>
        <h3>Accumulate wider than you multiply</h3>
        <p>Tensor cores multiply in bf16 or fp8 but accumulate the result in fp32. Thousands of
        products added together in low precision would lose everything to rounding. Every serious
        kernel does this, and it is why fp8 training is viable at all.</p>
        <h3>fp8 needs scaling</h3>
        <p>With a maximum of 448, fp8 cannot hold raw activations. Each tensor gets a scaling factor
        chosen from recent history so the values land in the usable range. Getting that scaling
        wrong is the main way fp8 training runs diverge.</p>
      </>
    ),
    failure: () => (
      <>
        <table className="w-full text-[13px]">
          <tbody>
            {[['Loss becomes NaN in fp16 but is fine in fp32', 'Overflow. Softmax, attention scores or the loss itself passed 65504.'],
              ['Gradients are all zero in fp16', 'Underflow. Small gradients fell below the smallest representable value. This is what loss scaling fixes.'],
              ['Model converges slower in bf16 than fp16', 'bf16 has 3 fewer mantissa bits. Usually fine, but tiny weight updates can round away entirely.'],
              ['Results differ between runs with the same seed', 'Reduction order changed. Different batch size or GPU count means different summation order.'],
              ['Validation loss does not match training loss exactly', 'Different batch shapes produce different accumulation paths. Small differences here are normal.'],
              ['fp8 run diverged after thousands of steps', 'Scaling factors drifted out of range. Check the amax history, not the model.']].map(r => (
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
