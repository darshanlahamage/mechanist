import { useEffect, useMemo, useRef, useState } from 'react'
import { rng } from '../lib/tensor'
import { Lab, Line, Panel, Select, Slider, Stat } from '../components/ui'

const N = 48          // image is N x N
const T = 1000        // total diffusion steps

/** A small scene, made with code so it is identical for everyone. */
function makeImage(): Float32Array {
  const img = new Float32Array(N * N * 3)
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const i = (y * N + x) * 3
      let r: number, g: number, b: number
      if (y > N * 0.68) { r = 0.22; g = 0.42; b = 0.20 }                 // ground
      else { const t = y / (N * 0.68); r = 0.25 + 0.4 * t; g = 0.45 + 0.35 * t; b = 0.85 - 0.1 * t }
      const dx = x - N * 0.72, dy = y - N * 0.26
      if (dx * dx + dy * dy < 54) { r = 1.0; g = 0.88; b = 0.35 }        // sun
      const mx = Math.abs(x - N * 0.34)
      if (y > N * 0.68 - (N * 0.34 - mx) * 0.9 && y < N * 0.7) { r = 0.42; g = 0.38; b = 0.44 }
      img[i] = r * 2 - 1; img[i + 1] = g * 2 - 1; img[i + 2] = b * 2 - 1  // scale to [-1,1]
    }
  }
  return img
}

function gaussian(seed: number, n: number): Float32Array {
  const u = rng(seed), out = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const a = Math.max(u(), 1e-12), b = u()
    out[i] = Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b)
  }
  return out
}

/** Both schedules, computed the way the papers define them. */
function alphaBar(t: number, kind: 'linear' | 'cosine'): number {
  if (kind === 'cosine') {
    const f = (s: number) => Math.cos(((s / T + 0.008) / 1.008) * Math.PI / 2) ** 2
    return f(t) / f(0)
  }
  const b0 = 1e-4, b1 = 0.02
  let logProd = 0
  for (let i = 0; i <= t; i++) logProd += Math.log(1 - (b0 + (b1 - b0) * (i / T)))
  return Math.exp(logProd)
}

function ImgCanvas({ data, label, px = 4 }: { data: Float32Array; label: string; px?: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const cv = ref.current!; cv.width = N; cv.height = N
    const g = cv.getContext('2d')!
    const im = g.createImageData(N, N)
    for (let i = 0; i < N * N; i++) {
      for (let c = 0; c < 3; c++) {
        const v = (data[i * 3 + c] + 1) / 2
        im.data[i * 4 + c] = Math.max(0, Math.min(255, Math.round(v * 255)))
      }
      im.data[i * 4 + 3] = 255
    }
    g.putImageData(im, 0, 0)
  }, [data])
  return (
    <div>
      <canvas ref={ref} style={{ width: N * px, height: N * px, imageRendering: 'pixelated' }}
              className="rounded-md border border-[var(--line)]" />
      <div className="mt-1.5 text-[11px] text-[var(--dim)]">{label}</div>
    </div>
  )
}

export default function DiffusionLab() {
  const [t, setT] = useState(300)
  const [schedule, setSchedule] = useState<'cosine' | 'linear'>('cosine')
  const [predError, setPredError] = useState(0.1)

  const x0 = useMemo(makeImage, [])
  const eps = useMemo(() => gaussian(99, N * N * 3), [])
  const errNoise = useMemo(() => gaussian(1234, N * N * 3), [])

  const ab = alphaBar(t, schedule)
  const sa = Math.sqrt(ab), s1a = Math.sqrt(1 - ab)

  const { xt, x0pred } = useMemo(() => {
    const xt = new Float32Array(x0.length)
    const x0pred = new Float32Array(x0.length)
    for (let i = 0; i < x0.length; i++) {
      xt[i] = sa * x0[i] + s1a * eps[i]
      const epsPred = eps[i] + predError * errNoise[i]   // the model is never exact
      x0pred[i] = (xt[i] - s1a * epsPred) / sa
    }
    return { xt, x0pred }
  }, [x0, eps, errNoise, sa, s1a, predError])

  const curves = useMemo(() => {
    const pts = (k: 'linear' | 'cosine') => {
      const p: [number, number][] = []
      for (let s = 0; s <= T; s += 20) p.push([s, alphaBar(s, k)])
      return p
    }
    return [
      { name: 'cosine', color: '#0b6f66', points: pts('cosine') },
      { name: 'linear', color: '#6a46a3', points: pts('linear') },
    ]
  }, [])

  const amplification = s1a / sa
  const signalLeft = sa

  return (
    <Lab title="Diffusion: noising, and getting the image back"
      takeaway={<>The forward step is one line of algebra, not a neural network. The model only ever
        predicts the noise. Notice the <b>error amplification</b> number: at high t you divide by a tiny
        √ᾱ, so a small mistake in the predicted noise becomes a huge mistake in the predicted image.
        That is why sampling takes many small steps instead of one big jump.</>}>
      <div className="grid gap-4 lg:grid-cols-[230px_1fr]">
        <div className="space-y-3">
          <Panel title="Controls">
            <div className="space-y-3">
              <Slider label="timestep t" min={0} max={T} step={5} value={t} onChange={setT} />
              <Select label="noise schedule" value={schedule} options={['cosine', 'linear'] as const}
                      onChange={setSchedule} />
              <Slider label="model's noise error" min={0} max={0.5} step={0.01} value={predError}
                      onChange={setPredError} fmt={v => v.toFixed(2)} />
            </div>
          </Panel>
          <Stat label="signal left (√ᾱ)" value={signalLeft.toFixed(4)}
                tone={signalLeft < 0.1 ? 'bad' : signalLeft < 0.4 ? 'warn' : 'good'} />
          <Stat label="error amplification" value={`${amplification.toFixed(1)}×`}
                sub="√(1−ᾱ) / √ᾱ" tone={amplification > 10 ? 'bad' : 'normal'} />
        </div>

        <div className="space-y-4">
          <Panel title={`Step ${t} of ${T}`}>
            <div className="flex flex-wrap items-start gap-5">
              <ImgCanvas data={x0} label="x₀ — the real image" />
              <div className="self-center text-[18px] text-[var(--dim)]">→</div>
              <ImgCanvas data={xt} label={`x_t — ${((1 - signalLeft) * 100).toFixed(0)}% noise`} />
              <div className="self-center text-[18px] text-[var(--dim)]">→</div>
              <ImgCanvas data={x0pred} label="x̂₀ — model's guess" />
            </div>
            <div className="mt-4 font-mono text-[12px] leading-relaxed text-[var(--dim)]">
              x_t = {sa.toFixed(3)}·x₀ + {s1a.toFixed(3)}·ε<br />
              x̂₀ = (x_t − {s1a.toFixed(3)}·ε̂) / {sa.toFixed(3)}
            </div>
          </Panel>

          <Panel title="How fast each schedule destroys the image">
            <Line series={curves} markers={[{ x: t, label: 'now' }]}
                  xLabel="timestep" yLabel="ᾱ (signal kept)" height={150} />
            <div className="mt-2 text-[12.5px] leading-relaxed text-[var(--dim)]">
              The linear schedule flattens out early — the last third of training steps see images
              that are already pure noise, so those steps teach the model almost nothing. The cosine
              schedule spreads the destruction out. That single change was the point of the paper
              that introduced it.
            </div>
          </Panel>
        </div>
      </div>
    </Lab>
  )
}
