import { useEffect, useMemo, useRef, useState } from 'react'
import { SURFACES, type OptName, type Surface, runOptimizer } from '../lib/optim'
import { Lab, Panel, Select, Slider, Stat } from '../components/ui'

const OPTS: { name: OptName; color: string }[] = [
  { name: 'SGD', color: '#f87171' },
  { name: 'SGD + momentum', color: '#fbbf24' },
  { name: 'RMSProp', color: '#60a5fa' },
  { name: 'Adam', color: '#5eead4' },
  { name: 'AdamW', color: '#a78bfa' },
]

export default function OptimizerLab() {
  const [surface, setSurface] = useState<Surface>('ravine')
  const [lr, setLr] = useState(0.1)
  const [steps, setSteps] = useState(60)
  const ref = useRef<HTMLCanvasElement>(null)

  const paths = useMemo(
    () => OPTS.map(o => ({ ...o, path: runOptimizer(o.name, surface, lr, steps) })),
    [surface, lr, steps])

  const W = 460, H = 320, R = 2.2

  useEffect(() => {
    const cv = ref.current!
    const dpr = window.devicePixelRatio || 1
    cv.width = W * dpr; cv.height = H * dpr
    cv.style.width = `${W}px`; cv.style.height = `${H}px`
    const g = cv.getContext('2d')!
    g.setTransform(dpr, 0, 0, dpr, 0, 0)

    const { f } = SURFACES[surface]
    const toPx = (x: number, y: number): [number, number] =>
      [((x + R) / (2 * R)) * W, H - ((y + R) / (2 * R)) * H]

    // contour field, drawn from the real function
    const img = g.createImageData(W, H)
    let lo = Infinity, hi = -Infinity
    const vals = new Float32Array(W * H)
    for (let py = 0; py < H; py++)
      for (let px = 0; px < W; px++) {
        const x = (px / W) * 2 * R - R, y = ((H - py) / H) * 2 * R - R
        const v = Math.log1p(Math.max(f(x, y), 0))
        vals[py * W + px] = v
        if (v < lo) lo = v; if (v > hi) hi = v
      }
    for (let i = 0; i < W * H; i++) {
      const t = (vals[i] - lo) / (hi - lo || 1)
      const band = Math.sin(t * 34) > 0.86 ? 26 : 0      // contour lines
      img.data[i * 4] = 14 + t * 34 + band
      img.data[i * 4 + 1] = 17 + t * 26 + band
      img.data[i * 4 + 2] = 22 + t * 52 + band
      img.data[i * 4 + 3] = 255
    }
    g.putImageData(img, 0, 0)

    // the minimum
    const [mx, my] = toPx(surface === 'rosenbrock' ? 1 : 0, surface === 'rosenbrock' ? 1 : 0)
    g.strokeStyle = 'rgba(255,255,255,0.5)'; g.lineWidth = 1
    g.beginPath(); g.moveTo(mx - 5, my); g.lineTo(mx + 5, my)
    g.moveTo(mx, my - 5); g.lineTo(mx, my + 5); g.stroke()

    for (const p of paths) {
      g.strokeStyle = p.color; g.lineWidth = 1.6; g.beginPath()
      p.path.forEach(([x, y], i) => {
        const [px, py] = toPx(x, y)
        i === 0 ? g.moveTo(px, py) : g.lineTo(px, py)
      })
      g.stroke()
      const last = p.path[p.path.length - 1]
      const [lx, ly] = toPx(last[0], last[1])
      if (Math.abs(last[0]) < R && Math.abs(last[1]) < R) {
        g.fillStyle = p.color; g.beginPath(); g.arc(lx, ly, 3, 0, Math.PI * 2); g.fill()
      }
    }
  }, [paths, surface])

  const { f, note } = SURFACES[surface]

  return (
    <Lab title="Every optimizer, same surface, same start"
      takeaway={<>Adam is not "better" — it rescales each direction by how big that direction's
        gradients have been. On the ravine that is exactly what you need, because the two directions
        have wildly different scales. On a round bowl it buys you nothing and costs you two extra
        copies of the model in memory. Raise the learning rate until SGD diverges and watch which
        ones survive.</>}>
      <div className="grid gap-4 lg:grid-cols-[230px_1fr]">
        <div className="space-y-3">
          <Panel title="Controls">
            <div className="space-y-3">
              <Select label="loss surface" value={surface}
                      options={['bowl', 'ravine', 'saddle', 'rosenbrock'] as const} onChange={setSurface} />
              <Slider label="learning rate" min={0.005} max={0.6} step={0.005} value={lr}
                      onChange={setLr} fmt={v => v.toFixed(3)} />
              <Slider label="steps" min={5} max={300} value={steps} onChange={setSteps} />
            </div>
          </Panel>
          <div className="rounded-lg border border-[var(--line)] bg-[var(--panel-2)] px-3 py-2.5 text-[12px] leading-relaxed text-[var(--dim)]">
            {note}
          </div>
          <Panel title="Where each one ended">
            <div className="space-y-1.5">
              {paths.map(p => {
                const last = p.path[p.path.length - 1]
                const loss = f(last[0], last[1])
                const diverged = !Number.isFinite(loss) || Math.abs(last[0]) > 10
                return (
                  <div key={p.name} className="flex items-center gap-2">
                    <span className="h-[2px] w-4 shrink-0" style={{ background: p.color }} />
                    <span className="flex-1 truncate text-[12px] text-[#c7cdd8]">{p.name}</span>
                    <span className={`font-mono text-[11px] ${diverged ? 'text-[var(--bad)]' : 'text-[var(--accent)]'}`}>
                      {diverged ? 'diverged' : loss.toFixed(4)}
                    </span>
                  </div>
                )
              })}
            </div>
          </Panel>
        </div>

        <Panel title="The path each optimizer takes">
          <canvas ref={ref} className="w-full max-w-[460px] rounded-md border border-[var(--line)]" />
          <div className="mt-2.5 text-[12.5px] leading-relaxed text-[var(--dim)]">
            Brighter means higher loss. The cross is the minimum. Each line is one optimizer walking
            downhill from the same starting point with the same learning rate.
          </div>
        </Panel>
      </div>
    </Lab>
  )
}
