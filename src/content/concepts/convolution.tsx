import ConvLab from '../../labs/ConvLab'
import { Code, Eq, Note } from '../../components/ui'
import type { Concept } from '../types'

export const convolution: Concept = {
  id: 'convolution',
  title: 'Convolution',
  tagline: 'A tiny dot product, repeated at every position, with the same weights everywhere.',
  track: 'vision',
  difficulty: 2,
  status: 'stable',
  hasLab: true,
  papers: [
    { title: 'ImageNet Classification with Deep CNNs (AlexNet)', year: 2012,
      url: 'https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html',
      idea: 'Convolutions on GPUs with ReLU and dropout. The result that started the modern era.' },
    { title: 'Deep Residual Learning (ResNet)', year: 2015, url: 'https://arxiv.org/abs/1512.03385',
      idea: 'Very deep stacks were getting worse, not overfitting. Skip connections made 100+ layers trainable.' },
    { title: 'A ConvNet for the 2020s', year: 2022, url: 'https://arxiv.org/abs/2201.03545',
      idea: 'Apply the transformer era\'s training recipe and design choices to a plain ConvNet and it matches transformers again.' },
  ],
  lenses: {
    intuition: () => (
      <>
        <p>A fully connected layer on a 224×224 image would need 50,000 weights per output unit, and
        it would have to learn what an edge looks like separately for every position in the image.</p>
        <p>A convolution uses a small window — often 3×3 — of weights, and slides it over every
        position. Nine weights, reused everywhere. If those nine weights detect a vertical edge in
        the top left, they detect one in the bottom right too, for free.</p>
        <p>That reuse is the whole idea. It is called translation equivariance: move the object in
        the image, and the response moves with it.</p>
        <Note kind="key" title="Depth builds size">
          <p>One 3×3 layer sees 3 pixels. Stack two and each output sees 5. Stack ten and it sees 21.
          The receptive field grows with depth, so early layers find edges and later layers find
          shapes — not because anyone designed it that way, but because that is what the geometry
          allows.</p>
        </Note>
      </>
    ),
    mechanism: () => (
      <>
        <p>Hover the output to see exactly which input pixels produced each value:</p>
        <ConvLab />
        <h3>The three knobs</h3>
        <ul>
          <li><b>Kernel size</b> — how much the window sees at once. 3×3 is almost always the answer,
            because two stacked 3×3 layers see as much as one 5×5 with fewer weights and an extra
            nonlinearity.</li>
          <li><b>Stride</b> — how far the window jumps. Stride 2 halves the output size and is the
            usual way to downsample.</li>
          <li><b>Padding</b> — zeros added around the edge. Padding of 1 with a 3×3 kernel keeps the
            output the same size as the input.</li>
        </ul>
      </>
    ),
    math: () => (
      <>
        <h3>The operation</h3>
        <Eq block tex="y[i,j] = \sum_{m}\sum_{n} x[i \cdot s + m - p,\ j \cdot s + n - p] \cdot w[m,n]" />
        <p>Strictly this is cross-correlation, not convolution — a true convolution flips the kernel.
        Since the weights are learned it makes no difference, and every framework does it this way
        while calling it convolution.</p>
        <h3>Output size</h3>
        <Eq block tex="H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1" />
        <h3>Receptive field</h3>
        <p>After <Eq tex="L" /> layers of kernel <Eq tex="k" /> with stride 1:</p>
        <Eq block tex="r_L = 1 + L(k-1)" />
        <p>With stride, each layer multiplies the growth, so downsampling expands the receptive
        field far faster than adding layers does.</p>
        <h3>Cost, with channels</h3>
        <Eq block tex="\text{FLOPs} = 2 \cdot H_{out} W_{out} \cdot C_{in} C_{out} \cdot k^2" />
        <p>Note <Eq tex="C_{in} \times C_{out}" />. Channel count dominates, which is why depthwise
        separable convolutions — doing the spatial part and the channel part separately — cut cost
        by roughly the kernel area.</p>
      </>
    ),
    code: () => (
      <>
        <Code lang="python">{`import numpy as np

def conv2d(x, w, stride=1, pad=0):
    """The literal definition. Four nested loops."""
    x = np.pad(x, pad)
    k = w.shape[0]
    out_h = (x.shape[0] - k) // stride + 1
    out_w = (x.shape[1] - k) // stride + 1

    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            window = x[i*stride : i*stride + k, j*stride : j*stride + k]
            out[i, j] = (window * w).sum()
    return out`}</Code>
        <Code lang="python">{`def conv2d_as_matmul(x, w, k=3):
    """How it is really computed: unfold into columns, then one big matmul.
       Same answer, hundreds of times faster, because GEMM is what hardware is good at."""
    patches = np.lib.stride_tricks.sliding_window_view(x, (k, k))
    patches = patches.reshape(-1, k * k)        # (positions, k*k)
    return (patches @ w.reshape(-1)).reshape(
        x.shape[0] - k + 1, x.shape[1] - k + 1)`}</Code>
        <Note kind="info" title="This is why convolution is fast">
          <p>Nobody runs the loop version. The data is rearranged so every sliding window becomes a
          row of a matrix, and the whole layer becomes one matrix multiply. Everything the hardware
          is good at — tensor cores, tiling, coalesced reads — applies to a matmul and not to a loop.</p>
        </Note>
      </>
    ),
    systems: () => (
      <>
        <h3>im2col trades memory for speed</h3>
        <p>Unfolding duplicates every pixel up to <Eq tex="k^2" /> times. For a 3×3 kernel that is
        nine times the memory for the unfolded copy. Fast implementations do it tile by tile inside
        the kernel so the expanded form never fully exists — which is the same trick as Flash
        Attention, for the same reason.</p>
        <h3>Channels decide the arithmetic intensity</h3>
        <p>A convolution with few channels is memory bound; with many channels it is a proper matmul
        and runs near peak. This is why the first layer of a CNN, with 3 input channels, is always
        inefficient, and why networks widen quickly after it.</p>
        <h3>Depthwise convolutions are slower than their FLOPs suggest</h3>
        <p>MobileNet-style depthwise separable convolutions cut FLOPs by roughly 8 or 9 times, but
        rarely run 8 times faster. They have very low arithmetic intensity, so they are memory bound
        — a clean example of why FLOPs are the wrong unit to optimise.</p>
      </>
    ),
    failure: () => (
      <>
        <table className="w-full text-[13px]">
          <tbody>
            {[['Output keeps shrinking through the network', 'No padding. Use padding = (k-1)/2 to keep the size.'],
              ['Model cannot see the whole object', 'Receptive field too small. Add depth, stride, or dilation.'],
              ['Edges of the image predicted badly', 'Zero padding makes the border look like a dark edge. Try reflect or replicate padding.'],
              ['Depthwise version has far fewer FLOPs but is barely faster', 'Memory bound. FLOPs were never the limit.'],
              ['Works on 224px, fails on other sizes', 'A fixed-size flatten before the classifier. Use global pooling instead.'],
              ['First layer dominates the profile', 'Three input channels gives terrible arithmetic intensity. Expected, and usually not worth fixing.']].map(r => (
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
