import { useEffect, useMemo, useRef } from 'react'
import { rng } from '../lib/tensor'
import { Scene, ease, type Frame, type Step } from './Scene'

const N = 56, T = 1000

const STEPS: Step[] = [
  { label: 'Start from a real image', say: 'Training never starts from noise. It starts from a picture you already have.', hold: 2.2 },
  { label: 'Destroy it, a little at a time', say: 'Mix in gaussian noise according to a fixed schedule. No network is involved — this is one line of algebra, and you can jump straight to any step.', hold: 5 },
  { label: 'Nothing left', say: 'By the last step the picture is gone. Every image in the dataset ends up at the same place: pure noise.', hold: 2.4 },
  { label: 'The only thing the model learns', say: 'Show it a noisy image and the timestep, and ask one question: what noise did I add? You know the answer, so it is plain regression.', hold: 3.6 },
  { label: 'Now run it backwards', say: 'Start from noise. Predict the noise, subtract a bit of it, repeat. The picture is not drawn — it is uncovered.', hold: 6.5 },
  { label: 'Why it cannot be done in one jump', say: 'Recovering the image divides by √ᾱ, which is tiny early on. A small error in the predicted noise gets multiplied by a large number. Many small steps keep that error under control.', hold: 5 },
]

function makeImage(): Float32Array {
  const img = new Float32Array(N * N * 3)
  for (let y = 0; y < N; y++)
    for (let x = 0; x < N; x++) {
      const i = (y * N + x) * 3
      let r: number, g: number, b: number
      if (y > N * 0.66) { r = 0.20; g = 0.42; b = 0.19 }
      else { const t = y / (N * 0.66); r = 0.24 + 0.42 * t; g = 0.44 + 0.36 * t; b = 0.86 - 0.12 * t }
      const dx = x - N * 0.73, dy = y - N * 0.24
      if (dx * dx + dy * dy < 62) { r = 1.0; g = 0.87; b = 0.32 }
      const mx = Math.abs(x - N * 0.33)
      if (y > N * 0.66 - (N * 0.33 - mx) * 0.95 && y < N * 0.68) { r = 0.40; g = 0.36; b = 0.43 }
      img[i] = r * 2 - 1; img[i + 1] = g * 2 - 1; img[i + 2] = b * 2 - 1
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

const alphaBar = (t: number) => {
  const f = (s: number) => Math.cos(((s / T + 0.008) / 1.008) * Math.PI / 2) ** 2
  return f(t) / f(0)
}

function Canvas({ data, size = 150, label, dim }:
  { data: Float32Array; size?: number; label?: string; dim?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const cv = ref.current!; cv.width = N; cv.height = N
    const g = cv.getContext('2d')!
    const im = g.createImageData(N, N)
    for (let i = 0; i < N * N; i++) {
      for (let c = 0; c < 3; c++)
        im.data[i * 4 + c] = Math.max(0, Math.min(255, Math.round(((data[i * 3 + c] + 1) / 2) * 255)))
      im.data[i * 4 + 3] = 255
    }
    g.putImageData(im, 0, 0)
  }, [data])
  return (
    <div className="text-center" style={{ opacity: dim ? 0.35 : 1 }}>
      <canvas ref={ref} className="rounded-md border border-[#2b3340]"
        style={{ width: size, height: size, imageRendering: 'pixelated' }} />
      {label && <div className="mt-1.5 font-mono text-[10.5px] text-[var(--dim)]">{label}</div>}
    </div>
  )
}

function Stage({ step, t }: Frame) {
  const x0 = useMemo(makeImage, [])
  const eps = useMemo(() => gaussian(99, N * N * 3), [])
  const errNoise = useMemo(() => gaussian(4242, N * N * 3), [])
  const e = ease(t)

  // which timestep this frame is showing
  const tt =
    step === 0 ? 0
    : step === 1 ? Math.round(e * T)
    : step === 2 ? T
    : step === 3 ? 420
    : step === 4 ? Math.round(T - e * T)
    : Math.round(900 - e * 880)

  const ab = alphaBar(tt), sa = Math.sqrt(ab), s1a = Math.sqrt(1 - ab)

  const xt = useMemo(() => {
    const o = new Float32Array(x0.length)
    for (let i = 0; i < x0.length; i++) o[i] = sa * x0[i] + s1a * eps[i]
    return o
  }, [x0, eps, sa, s1a])

  // what the model would hand back, with a realistic amount of error
  const guess = useMemo(() => {
    const err = step === 5 ? 0.12 : 0.04
    const o = new Float32Array(x0.length)
    for (let i = 0; i < x0.length; i++) {
      const epsHat = eps[i] + err * errNoise[i]
      o[i] = Math.max(-1, Math.min(1, (xt[i] - s1a * epsHat) / Math.max(sa, 1e-4)))
    }
    return o
  }, [xt, eps, errNoise, sa, s1a, step])

  const noiseOnly = useMemo(() => {
    const o = new Float32Array(x0.length)
    for (let i = 0; i < x0.length; i++) o[i] = eps[i] * 0.55
    return o
  }, [eps])

  const amp = s1a / Math.max(sa, 1e-4)

  return (
    <div className="flex flex-col items-center justify-center gap-4 px-4 py-6" style={{ minHeight: 330 }}>
      <div className="flex items-center justify-center gap-6">
        {step <= 2 && (
          <>
            <Canvas data={x0} label="x₀  the real image" dim={step === 2} />
            <div className="text-[18px] text-[var(--dim)]">→</div>
            <Canvas data={xt} label={`x_t   t = ${tt}`} />
          </>
        )}

        {step === 3 && (
          <>
            <Canvas data={xt} label={`x_t   t = ${tt}`} />
            <div className="text-center text-[12px] text-[var(--accent)]">
              <div className="text-[18px]">→</div>
              <div className="mt-1">model</div>
            </div>
            <Canvas data={noiseOnly} label="ε̂   its guess at the noise" />
            <div className="text-[18px] text-[var(--dim)]">≈</div>
            <Canvas data={noiseOnly} label="ε   the noise you added" />
          </>
        )}

        {step === 4 && (
          <>
            <Canvas data={xt} label={`x_t   t = ${tt}`} size={168} />
            <div className="text-center text-[12px] text-[var(--dim)]">
              <div className="text-[18px]">→</div>
              <div className="mt-1">subtract a little<br />predicted noise</div>
            </div>
            <Canvas data={guess} label="what it looks like underneath" size={168} />
          </>
        )}

        {step === 5 && (
          <>
            <Canvas data={xt} label={`x_t   t = ${tt}`} />
            <div className="text-center">
              <div className="text-[18px] text-[var(--dim)]">→</div>
              <div className="mt-1 font-mono text-[11px] text-[var(--warn)]">
                error × {amp.toFixed(1)}
              </div>
            </div>
            <Canvas data={guess} label="one jump straight to x₀" />
          </>
        )}
      </div>

      {/* the schedule, with a marker where we are */}
      <div className="w-full max-w-[560px]">
        <div className="relative h-[42px]">
          <svg viewBox="0 0 560 42" className="w-full">
            <polyline fill="none" stroke="#2b3340" strokeWidth="1.5"
              points={Array.from({ length: 56 }, (_, i) => {
                const s = (i / 55) * T
                return `${(i / 55) * 560},${38 - alphaBar(s) * 32}`
              }).join(' ')} />
            <circle cx={(tt / T) * 560} cy={38 - ab * 32} r="4" fill="#5eead4" />
          </svg>
        </div>
        <div className="flex justify-between font-mono text-[10px] text-[var(--dim)]">
          <span>t = 0 · all picture</span>
          <span className="text-[var(--accent)]">√ᾱ = {sa.toFixed(3)} signal left</span>
          <span>t = 1000 · all noise</span>
        </div>
      </div>
    </div>
  )
}

export default function DiffusionFlow() {
  return <Scene steps={STEPS} height={330} autoPlay>{f => <Stage {...f} />}</Scene>
}
