import RooflineLab from '../../labs/RooflineLab'
import { Code, Eq, Note } from '../../components/ui'
import type { Concept } from '../types'

export const roofline: Concept = {
  id: 'roofline',
  title: 'The roofline model',
  tagline: 'One number tells you whether to optimise the maths or the memory. Almost nobody computes it.',
  track: 'gpu',
  difficulty: 3,
  status: 'stable',
  hasLab: true,
  papers: [
    { title: 'Roofline: An Insightful Visual Performance Model', year: 2009,
      url: 'https://dl.acm.org/doi/10.1145/1498765.1498785',
      idea: 'Plot achievable performance against arithmetic intensity. Two straight lines tell you what is limiting you and what to do about it.' },
  ],
  lenses: {
    intuition: () => (
      <>
        <p>A GPU can do two things: move bytes and do arithmetic. For any given operation, one of
        those finishes last, and that one is your time.</p>
        <p>The deciding number is <b>arithmetic intensity</b>: how many operations you perform per
        byte you move. Compare it against the hardware's own ratio of peak compute to bandwidth.</p>
        <p>An H100 does about 989 TFLOP/s in bf16 with 3.35 TB/s of bandwidth. That is roughly 295
        operations per byte. If your operation does fewer than 295, no amount of extra compute will
        help you — the math units will be waiting for data regardless.</p>
        <Note kind="key" title="Why this is worth learning">
          <p>It tells you which optimisations are pointless before you spend a week on them. If you
          are memory bound, a faster kernel will not help; fusing two kernels together will.</p>
        </Note>
      </>
    ),
    mechanism: () => (
      <>
        <p>Every point here is computed from its own FLOP and byte counts, not placed by hand:</p>
        <RooflineLab />
        <h3>Reading the chart</h3>
        <ul>
          <li>The slanted line is bandwidth. Below the ridge point, your speed is intensity times
            bandwidth, and nothing else matters.</li>
          <li>The flat line is peak compute. Past the ridge, more intensity buys nothing.</li>
          <li>The corner where they meet is the ridge point. It is a property of the hardware alone.</li>
        </ul>
        <h3>Flash attention versus plain attention</h3>
        <p>Both do identical arithmetic. Plain attention writes the full n×n score matrix to memory
        and reads it back twice. Flash attention keeps tiles on chip and never writes it. Same FLOPs,
        far fewer bytes, so the intensity shoots up and the operation moves from the memory-bound
        side to the compute-bound side. That is the entire idea of the paper, visible as a jump along
        one axis.</p>
      </>
    ),
    math: () => (
      <>
        <h3>The model</h3>
        <Eq block tex="I = \frac{\text{FLOPs}}{\text{bytes moved}}" />
        <Eq block tex="\text{attainable FLOP/s} = \min(\text{peak FLOP/s},\ I \times \text{bandwidth})" />
        <Eq block tex="\text{ridge point} = \frac{\text{peak FLOP/s}}{\text{bandwidth}}" />
        <h3>Worked example: a matmul</h3>
        <p>For <Eq tex="C = AB" /> with all matrices <Eq tex="n \times n" /> in bf16:</p>
        <Eq block tex="\text{FLOPs} = 2n^3, \qquad \text{bytes} = 3 \cdot 2n^2" />
        <Eq block tex="I = \frac{2n^3}{6n^2} = \frac{n}{3}" />
        <p>So intensity grows linearly with size. At n = 512 it is 170 — memory bound on an H100.
        At n = 4096 it is 1365 — comfortably compute bound. Same operation, different regime, purely
        because of shape. This is why small matmuls are so inefficient and why batching works.</p>
        <h3>Ridge points worth remembering</h3>
        <table className="w-full font-mono text-[12.5px]">
          <tbody>
            {[['H100 bf16', '989 TF / 3.35 TB/s', '295'],
              ['A100 bf16', '312 TF / 2.0 TB/s', '153'],
              ['RTX 4090', '165 TF / 1.0 TB/s', '164'],
              ['M3 Max', '28 TF / 0.4 TB/s', '70']].map(r => (
              <tr key={r[0]} className="border-t border-[var(--line)]">
                <td className="py-1.5 text-[var(--accent)]">{r[0]}</td>
                <td className="py-1.5 text-[var(--dim)]">{r[1]}</td>
                <td className="py-1.5 text-right text-[#c7cdd8]">{r[2]} FLOPs/byte</td>
              </tr>
            ))}
          </tbody>
        </table>
      </>
    ),
    code: () => (
      <>
        <Code lang="python">{`def roofline(flops, bytes_moved, peak_flops, bandwidth):
    intensity = flops / bytes_moved
    attainable = min(peak_flops, intensity * bandwidth)
    return {
        "intensity": intensity,
        "ridge": peak_flops / bandwidth,
        "bound": "memory" if intensity < peak_flops / bandwidth else "compute",
        "best_time_ms": flops / attainable * 1000,
    }

# one decode step of an 8B model on an H100
print(roofline(flops=2 * 8e9, bytes_moved=16e9,
               peak_flops=989e12, bandwidth=3.35e12))
# intensity 1.0 against a ridge of 295 -- memory bound by a factor of 295`}</Code>
        <Code lang="python">{`# measure the real one instead of trusting the spec sheet
import torch, time

def measured_bandwidth(n=1 << 28):
    x = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    y = torch.empty_like(x)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(20):
        y.copy_(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 20
    return 2 * x.numel() * 2 / dt / 1e9       # GB/s, read + write

print(measured_bandwidth())   # usually 80-90% of the spec sheet number`}</Code>
      </>
    ),
    systems: () => (
      <>
        <h3>What to do on each side of the ridge</h3>
        <table className="w-full text-[13px]">
          <tbody>
            {[['Memory bound', 'Fuse kernels so data is touched once. Use a smaller dtype. Increase batch size. Improve reuse with tiling. Do NOT look for faster arithmetic.'],
              ['Compute bound', 'Use tensor cores. Pick shapes that match the hardware MMA sizes. Lower precision. Check you are not losing time to poor occupancy or divergence.'],
              ['Neither — far below the roof', 'Something else is wrong: launch overhead, synchronisation, bad access patterns, or the GPU is idle waiting on the CPU.']].map(r => (
              <tr key={r[0]} className="border-t border-[var(--line)] align-top">
                <td className="py-2 pr-4 font-medium text-[#c7cdd8]">{r[0]}</td>
                <td className="py-2 text-[var(--dim)]">{r[1]}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <h3>Peak is a lie, but a useful one</h3>
        <p>You will never reach peak FLOP/s or peak bandwidth. Real matmuls reach 70-85% of peak
        compute; real memory streams reach 80-90% of peak bandwidth. The roofline still tells you
        which ceiling you are under, which is the decision it is for.</p>
        <h3>The hierarchy has more than one roof</h3>
        <p>The simple version uses HBM bandwidth. A fuller version adds a roof for L2 and one for
        shared memory, each with much higher bandwidth. An operation that is memory bound against
        HBM can be compute bound against shared memory — which is exactly what tiling achieves, and
        what the matmul optimisation ladder is doing step by step.</p>
      </>
    ),
    failure: () => (
      <>
        <table className="w-full text-[13px]">
          <tbody>
            {[['Optimised the maths, nothing got faster', 'You were memory bound. Compute the intensity first, every time.'],
              ['Model runs at 5% of peak FLOPs', 'Almost certainly normal for decode. Check against the roofline before calling it a bug.'],
              ['Kernel is far below both roofs', 'Not a roofline problem. Look at launch overhead, synchronisation or occupancy.'],
              ['Intensity looks high but performance is poor', 'Bytes counted as ideal traffic, not actual. Strided or uncoalesced access moves far more than you think.'],
              ['Fusing two kernels gave a large speedup', 'Correct and expected — you removed a round trip to memory without changing the arithmetic.'],
              ['Half precision gave no speedup on a compute-bound kernel', 'Tensor cores not being used. Check the shapes and alignment the hardware requires.']].map(r => (
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
