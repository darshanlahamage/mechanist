import RopeLab from '../../labs/RopeLab'
import { Code, Eq, Note } from '../../components/ui'
import type { Concept } from '../types'

export const rope: Concept = {
  id: 'rope',
  title: 'RoPE — rotary position embeddings',
  tagline: 'Position is not added to the vector. The vector is rotated, and the angle is the position.',
  track: 'transformer',
  difficulty: 3,
  status: 'stable',
  hasLab: true,
  prereqs: ['attention'],
  papers: [
    { title: 'RoFormer: Enhanced Transformer with Rotary Position Embedding', year: 2021,
      url: 'https://arxiv.org/abs/2104.09864',
      idea: 'Rotate query and key by an angle set by position. The dot product then depends only on the distance between tokens, for free.' },
    { title: 'Extending Context Window via Position Interpolation', year: 2023,
      url: 'https://arxiv.org/abs/2306.15595',
      idea: 'Squeeze positions into the trained range instead of extending past it. A few hundred fine-tuning steps then buy you 8x the context.' },
    { title: 'YaRN: Efficient Context Window Extension', year: 2023, url: 'https://arxiv.org/abs/2309.00071',
      idea: 'Do not stretch every frequency equally. High frequency dims carry local detail and should be left alone.' },
  ],
  lenses: {
    intuition: () => (
      <>
        <p>Attention sees a bag of tokens. It has no idea which came first. So you have to tell it.</p>
        <p>The old way was to add a position vector to the token vector. It works, but it mixes two
        unrelated things into the same numbers, and it does not generalise past the lengths you
        trained on.</p>
        <p>RoPE does something cleaner. Take the query and key vectors, chop them into pairs of
        numbers, and treat each pair as a point on a plane. Then rotate each point by an angle
        proportional to the token's position.</p>
        <p>Here is why that is clever: when you later take the dot product of a query at position
        <code>m</code> and a key at position <code>n</code>, the two rotations partly cancel. What is
        left depends only on <code>m − n</code>, the distance between them. You injected absolute
        position, and attention automatically reads out relative position.</p>
        <Note kind="key" title="One sentence version">
          <p>Rotating both sides by their own position makes the dot product care only about the gap
          between them.</p>
        </Note>
      </>
    ),
    mechanism: () => (
      <>
        <p>Each pair of dimensions gets its own rotation speed. Pair <Eq tex="i" /> out of
        <Eq tex="d/2" /> rotates at:</p>
        <Eq block tex="\theta_i = \text{base}^{-2i/d}, \quad \text{base} = 10000" />
        <p>Pair 0 spins fastest. The last pair spins so slowly that its full cycle is longer than any
        document you will ever feed it. So the early dimensions encode "right next to me" and the late
        ones encode "roughly where in the text".</p>
        <p>At position <Eq tex="m" />, the pair is rotated by <Eq tex="m\theta_i" />:</p>
        <Eq block tex="\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}" />
        <p>Drag the position and watch the similarity curve. Then push the tested length past the
        trained length and see what goes wrong:</p>
        <RopeLab />
      </>
    ),
    math: () => (
      <>
        <h3>The relative position property</h3>
        <p>Write the rotation as a matrix <Eq tex="R_m" />. Rotation matrices have the property
        <Eq tex="R_m^\top R_n = R_{n-m}" />. So:</p>
        <Eq block tex="(R_m q)^\top (R_n k) = q^\top R_m^\top R_n k = q^\top R_{n-m} k" />
        <p>The absolute positions have disappeared. Only <Eq tex="n - m" /> survives. Nothing was
        trained to make this happen — it is a property of rotation.</p>

        <h3>Complex number form</h3>
        <p>Pair up dimensions as complex numbers and rotation is just multiplication by a phase:</p>
        <Eq block tex="q_m = q \odot e^{im\theta}, \quad k_n = k \odot e^{in\theta}" />
        <Eq block tex="\langle q_m, k_n \rangle = \mathrm{Re}\left[ \sum_i q_i \bar{k_i} e^{i(m-n)\theta_i} \right]" />
        <p>This is why implementations keep a complex-number version around — it is the same thing
        written shorter and it runs faster.</p>

        <h3>What breaks past the trained length</h3>
        <p>Nothing in the math breaks. The rotations keep working at any position. What breaks is the
        model: at position 10,000 the fast dimensions have spun through angles the model has never
        seen, and it has no learned behaviour for them.</p>
        <p>The fixes all slow the rotation down:</p>
        <ul>
          <li><b>Linear interpolation (PI)</b> — divide every position by the stretch factor. All
            frequencies squeeze equally. Simple, but it crushes the fine local detail.</li>
          <li><b>NTK-aware</b> — raise the base instead. High frequencies barely change, low
            frequencies stretch a lot. Often works with no fine-tuning at all.</li>
          <li><b>YaRN</b> — do it per frequency band: leave the fast ones alone, interpolate the slow
            ones, and adjust attention temperature to compensate.</li>
        </ul>
      </>
    ),
    code: () => (
      <>
        <Code lang="python">{`import torch

def rope_frequencies(d_head, max_len, base=10000.0, device="cpu"):
    # one angle per PAIR of dimensions
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    pos = torch.arange(max_len, device=device).float()
    angles = torch.outer(pos, inv_freq)        # (max_len, d_head/2)
    return angles.cos(), angles.sin()

def apply_rope(x, cos, sin):
    """x: (batch, heads, seq, d_head)"""
    x1, x2 = x[..., 0::2], x[..., 1::2]        # even and odd dims form the pairs
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin       # plain 2D rotation
    out[..., 1::2] = x1 * sin + x2 * cos
    return out`}</Code>
        <Note kind="info" title="Where it goes in the model">
          <p>RoPE is applied to Q and K only — never to V, and never to the residual stream. V carries
          content, not position. Applying it to V is a real bug people ship.</p>
          <p>It is also applied <em>inside</em> every attention layer, not once at the input. That is
          why the KV cache stores keys that already have rotation baked in.</p>
        </Note>
        <Code lang="python">{`# context extension, both variants, in four lines
def extend(base, d_head, factor, method):
    if method == "linear":   # squeeze positions
        return base, lambda pos: pos / factor
    if method == "ntk":      # slow every frequency by raising the base
        return base * factor ** (d_head / (d_head - 2)), lambda pos: pos`}</Code>
      </>
    ),
    systems: () => (
      <>
        <h3>It costs almost nothing</h3>
        <p>RoPE touches Q and K only, with a few multiplies and adds per element. Compared with the
        projections and the attention matmul it is noise — well under 1% of layer time. The cos and
        sin tables are computed once and cached.</p>
        <p>The real systems question is where to apply it. Applying RoPE as a separate kernel means
        reading and writing Q and K to memory again, which for a memory-bound decode step is pure
        waste. Every serious implementation fuses it into the attention kernel or into the QKV
        projection epilogue.</p>
        <h3>It changes what you can cache</h3>
        <p>Keys are stored rotated. That is fine for normal decoding, but it means you cannot move a
        cached prefix to a different position without re-rotating it. Prefix caching systems that
        share a cached system prompt across requests rely on that prefix always sitting at position
        zero. Break that assumption and the cache is silently wrong.</p>
        <h3>Base choice at scale</h3>
        <p>Models trained for long context raise the base from 10,000 to 500,000 or a million during a
        long-context training phase. That is a deliberate decision: a bigger base slows every rotation,
        giving the slow dimensions room to describe much longer documents.</p>
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
            {[['Perfect to 4k tokens, nonsense at 5k', 'Past the trained length. Rotation angles the model has never seen.'],
              ['Context extension ruined short prompts', 'Linear interpolation squeezed the high frequencies too. Use NTK or YaRN.'],
              ['Quality drops after adding RoPE to a working model', 'Applied to V as well as Q and K.'],
              ['Shared prefix cache gives wrong answers', 'Cached keys are rotated for their original positions and got reused at different ones.'],
              ['Fine in fp32, degraded in fp16 at long context', 'cos/sin table precision. Build the table in fp32 and cast after.'],
              ['Model ignores long range structure', 'Base too small for the target length. Raise it during long-context training.']].map(r => (
              <tr key={r[0]} className="border-t border-[var(--line)] align-top">
                <td className="py-2 pr-4 text-[#c7cdd8]">{r[0]}</td>
                <td className="py-2 text-[var(--dim)]">{r[1]}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <Note kind="warn" title="Try it in the lab">
          <p>Set trained length to 512 and tested length to 16384 with no scaling. Look at the
          "fastest dim turns" number — it will be in the thousands. Then switch to NTK-aware and watch
          it drop.</p>
        </Note>
      </>
    ),
  },
}
