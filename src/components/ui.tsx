import { useEffect, useRef, useState, type ReactNode } from 'react'
import katex from 'katex'
import type { Mat } from '../lib/tensor'
import { minMax } from '../lib/tensor'

export function Eq({ tex, block = false }: { tex: string; block?: boolean }) {
  const html = katex.renderToString(tex, { throwOnError: false, displayMode: block })
  return block
    ? <div className="katex-display" dangerouslySetInnerHTML={{ __html: html }} />
    : <span className="mx-[0.15em]" dangerouslySetInnerHTML={{ __html: html }} />
}

export function Panel({ title, right, children, className = '' }:
  { title?: string; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <div className={`rounded-xl border border-[var(--line)] bg-[var(--panel)] ${className}`}>
      {title && (
        <div className="flex items-center justify-between gap-3 border-b border-[var(--line)] px-4 py-2.5">
          <h4 className="text-[13px] font-semibold tracking-wide text-[var(--text)]">{title}</h4>
          {right}
        </div>
      )}
      <div className="p-4">{children}</div>
    </div>
  )
}

export function Note({ kind = 'info', title, children }:
  { kind?: 'info' | 'warn' | 'bad' | 'key'; title?: string; children: ReactNode }) {
  const c = {
    info: ['#5eead4', 'rgba(94,234,212,0.07)'],
    key:  ['#a78bfa', 'rgba(167,139,250,0.08)'],
    warn: ['#fbbf24', 'rgba(251,191,36,0.07)'],
    bad:  ['#f87171', 'rgba(248,113,113,0.07)'],
  }[kind]
  return (
    <div className="my-4 rounded-lg border-l-2 px-4 py-3 text-[14px] leading-relaxed"
         style={{ borderColor: c[0], background: c[1] }}>
      {title && <div className="mb-1 font-semibold" style={{ color: c[0] }}>{title}</div>}
      <div className="text-[#c7cdd8] [&_p]:my-1.5">{children}</div>
    </div>
  )
}

export function Code({ children, lang = 'python' }: { children: string; lang?: string }) {
  return (
    <div className="my-4 overflow-hidden rounded-lg border border-[var(--line)] bg-[#0e1116]">
      <div className="border-b border-[var(--line)] px-3 py-1.5 text-[11px] uppercase tracking-wider text-[var(--dim)]">{lang}</div>
      <pre className="overflow-x-auto p-3 text-[12.5px] leading-[1.65]"><code className="font-mono text-[#cdd6e4]">{children.trim()}</code></pre>
    </div>
  )
}

export function Slider({ label, value, min, max, step = 1, onChange, fmt }:
  { label: string; value: number; min: number; max: number; step?: number
    onChange: (v: number) => void; fmt?: (v: number) => string }) {
  return (
    <label className="block">
      <div className="mb-1.5 flex items-baseline justify-between">
        <span className="text-[12px] text-[var(--dim)]">{label}</span>
        <span className="font-mono text-[12px] text-[var(--accent)]">{fmt ? fmt(value) : value}</span>
      </div>
      <input type="range" className="w-full" min={min} max={max} step={step} value={value}
             onChange={e => onChange(Number(e.target.value))} />
    </label>
  )
}

export function Select<T extends string>({ label, value, options, onChange }:
  { label: string; value: T; options: readonly T[]; onChange: (v: T) => void }) {
  return (
    <label className="block">
      <div className="mb-1.5 text-[12px] text-[var(--dim)]">{label}</div>
      <select value={value} onChange={e => onChange(e.target.value as T)}
        className="w-full rounded-md border border-[var(--line)] bg-[var(--panel-2)] px-2.5 py-1.5 text-[13px] text-[var(--text)] outline-none focus:border-[var(--accent)]">
        {options.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
    </label>
  )
}

export function Toggle({ label, value, onChange, hint }:
  { label: string; value: boolean; onChange: (v: boolean) => void; hint?: string }) {
  return (
    <button onClick={() => onChange(!value)}
      className="flex w-full items-center justify-between gap-3 rounded-md border border-[var(--line)] bg-[var(--panel-2)] px-2.5 py-2 text-left transition hover:border-[#38404c]">
      <span>
        <span className="block text-[13px] text-[var(--text)]">{label}</span>
        {hint && <span className="block text-[11px] text-[var(--dim)]">{hint}</span>}
      </span>
      <span className={`h-5 w-9 shrink-0 rounded-full p-0.5 transition ${value ? 'bg-[var(--accent)]' : 'bg-[#2a313b]'}`}>
        <span className={`block h-4 w-4 rounded-full bg-[#0b0d10] transition ${value ? 'translate-x-4' : ''}`} />
      </span>
    </button>
  )
}

export function Stat({ label, value, sub, tone = 'normal' }:
  { label: string; value: string; sub?: string; tone?: 'normal' | 'good' | 'warn' | 'bad' }) {
  const col = { normal: 'var(--text)', good: 'var(--accent)', warn: 'var(--warn)', bad: 'var(--bad)' }[tone]
  return (
    <div className="rounded-lg border border-[var(--line)] bg-[var(--panel-2)] px-3 py-2.5">
      <div className="text-[11px] uppercase tracking-wider text-[var(--dim)]">{label}</div>
      <div className="mt-0.5 font-mono text-[15px] font-semibold" style={{ color: col }}>{value}</div>
      {sub && <div className="mt-0.5 text-[11px] text-[var(--dim)]">{sub}</div>}
    </div>
  )
}

/** Colour ramp: blue (low) -> dark (mid) -> teal (high). Readable for most colour vision. */
function ramp(t: number): string {
  const x = Math.max(0, Math.min(1, t))
  if (x < 0.5) {
    const u = x * 2
    return `rgb(${Math.round(38 + u * 10)}, ${Math.round(70 - u * 40)}, ${Math.round(130 - u * 80)})`
  }
  const u = (x - 0.5) * 2
  return `rgb(${Math.round(48 + u * 46)}, ${Math.round(30 + u * 204)}, ${Math.round(50 + u * 162)})`
}

/**
 * Draws a real matrix on a canvas. Hover a cell to read its exact value.
 * There is a table fallback below it so the numbers are never trapped in a picture.
 */
export function Heat({ m, rowLabels, colLabels, cell = 26, title, showValues = false }:
  { m: Mat; rowLabels?: string[]; colLabels?: string[]; cell?: number; title?: string; showValues?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null)
  const [lo, hi] = minMax(m)
  const rows = m.length, cols = m[0]?.length ?? 0

  useEffect(() => {
    const cv = ref.current
    if (!cv) return
    const dpr = window.devicePixelRatio || 1
    cv.width = cols * cell * dpr
    cv.height = rows * cell * dpr
    cv.style.width = `${cols * cell}px`
    cv.style.height = `${rows * cell}px`
    const g = cv.getContext('2d')!
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    g.clearRect(0, 0, cols * cell, rows * cell)
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const v = m[i][j]
        if (!Number.isFinite(v)) {
          g.fillStyle = '#15181d'
        } else {
          g.fillStyle = ramp(hi === lo ? 0.5 : (v - lo) / (hi - lo))
        }
        g.fillRect(j * cell, i * cell, cell - 1, cell - 1)
        if (showValues && cell >= 30 && Number.isFinite(v)) {
          g.fillStyle = 'rgba(255,255,255,0.85)'
          g.font = '10px ui-monospace, monospace'
          g.textAlign = 'center'
          g.fillText(v.toFixed(2), j * cell + cell / 2 - 0.5, i * cell + cell / 2 + 3)
        }
      }
    }
  }, [m, cell, lo, hi, rows, cols, showValues])

  const onMove = (e: React.MouseEvent) => {
    const r = ref.current!.getBoundingClientRect()
    const j = Math.floor((e.clientX - r.left) / cell)
    const i = Math.floor((e.clientY - r.top) / cell)
    setHover(i >= 0 && i < rows && j >= 0 && j < cols ? { i, j } : null)
  }

  const hv = hover ? m[hover.i][hover.j] : null

  return (
    <div>
      {title && <div className="mb-1.5 text-[12px] text-[var(--dim)]">{title}</div>}
      <div className="flex gap-2">
        {rowLabels && (
          <div className="shrink-0 pt-0" style={{ fontSize: 10 }}>
            {rowLabels.map((l, i) => (
              <div key={i} className="flex items-center justify-end pr-1 font-mono text-[var(--dim)]"
                   style={{ height: cell }}>{l}</div>
            ))}
          </div>
        )}
        <div>
          <canvas ref={ref} onMouseMove={onMove} onMouseLeave={() => setHover(null)}
                  className="rounded-sm" style={{ imageRendering: 'pixelated', cursor: 'crosshair' }} />
          {colLabels && (
            <div className="flex" style={{ fontSize: 10 }}>
              {colLabels.map((l, j) => (
                <div key={j} className="overflow-hidden text-center font-mono text-[var(--dim)]"
                     style={{ width: cell }}>{l}</div>
              ))}
            </div>
          )}
        </div>
      </div>
      <div className="mt-2 h-4 font-mono text-[11px] text-[var(--dim)]">
        {hover && hv !== null
          ? <>[{hover.i},{hover.j}] = <span className="text-[var(--accent)]">{Number.isFinite(hv) ? hv.toFixed(4) : '-inf (masked)'}</span></>
          : <>range {lo.toFixed(2)} … {hi.toFixed(2)} — hover a cell to read it</>}
      </div>
    </div>
  )
}

export function Bars({ values, labels, height = 90, highlight }:
  { values: number[]; labels?: string[]; height?: number; highlight?: number }) {
  const max = Math.max(...values, 1e-9)
  return (
    <div>
      <div className="flex items-end gap-[3px]" style={{ height }}>
        {values.map((v, i) => (
          <div key={i} className="group relative flex-1" style={{ height: '100%' }}>
            <div className="absolute bottom-0 w-full rounded-t-[2px] transition-all"
                 style={{ height: `${(v / max) * 100}%`,
                          background: i === highlight ? 'var(--accent-2)' : 'var(--accent)',
                          opacity: i === highlight ? 1 : 0.75 }} />
            <div className="pointer-events-none absolute bottom-full left-1/2 z-10 mb-1 hidden -translate-x-1/2 whitespace-nowrap rounded bg-[#0b0d10] px-1.5 py-0.5 font-mono text-[10px] text-[var(--accent)] group-hover:block">
              {v.toFixed(4)}
            </div>
          </div>
        ))}
      </div>
      {labels && (
        <div className="mt-1 flex gap-[3px]">
          {labels.map((l, i) => (
            <div key={i} className="flex-1 overflow-hidden text-center font-mono text-[10px] text-[var(--dim)]">{l}</div>
          ))}
        </div>
      )}
    </div>
  )
}

export function Lab({ title, children, takeaway }:
  { title: string; children: ReactNode; takeaway?: ReactNode }) {
  return (
    <div className="my-6 overflow-hidden rounded-xl border border-[#2b3340] bg-[#0f1319]">
      <div className="flex items-center gap-2 border-b border-[#2b3340] bg-[#131821] px-4 py-2.5">
        <span className="rounded bg-[var(--accent)]/15 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-[var(--accent)]">Lab</span>
        <span className="text-[13px] font-semibold">{title}</span>
        <span className="ml-auto text-[11px] text-[var(--dim)]">runs live in your browser</span>
      </div>
      <div className="p-4">{children}</div>
      {takeaway && (
        <div className="border-t border-[#2b3340] bg-[#0c1015] px-4 py-3 text-[13px] text-[#c7cdd8]">
          <span className="font-semibold text-[var(--accent)]">What to take away: </span>{takeaway}
        </div>
      )}
    </div>
  )
}

/** Simple SVG line chart. Multiple series, shared axes, honest tick labels. */
export function Line({ series, xLabel, yLabel, height = 170, xTicks, yFmt, markers }:
  { series: { name: string; color: string; points: [number, number][]; dashed?: boolean }[]
    xLabel?: string; yLabel?: string; height?: number
    xTicks?: number; yFmt?: (v: number) => string; markers?: { x: number; label: string }[] }) {
  const all = series.flatMap(s => s.points)
  if (all.length === 0) return null
  const xs = all.map(p => p[0]), ys = all.map(p => p[1])
  const x0 = Math.min(...xs), x1 = Math.max(...xs)
  let y0 = Math.min(...ys), y1 = Math.max(...ys)
  if (y1 - y0 < 1e-9) { y1 = y0 + 1 }
  const pad = (y1 - y0) * 0.08
  y0 -= pad; y1 += pad
  const W = 560, H = height, L = 46, B = 26, R = 10, T = 10
  const px = (x: number) => L + ((x - x0) / (x1 - x0 || 1)) * (W - L - R)
  const py = (y: number) => T + (1 - (y - y0) / (y1 - y0)) * (H - T - B)
  const fmt = yFmt ?? ((v: number) => (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(1) : v.toFixed(2)))
  const nT = xTicks ?? 5

  return (
    <div className="w-full overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ minWidth: 380 }}>
        {[0, 0.25, 0.5, 0.75, 1].map(f => {
          const y = y0 + f * (y1 - y0)
          return (
            <g key={f}>
              <line x1={L} x2={W - R} y1={py(y)} y2={py(y)} stroke="#242a33" strokeWidth="1" />
              <text x={L - 6} y={py(y) + 3} textAnchor="end" fontSize="9" fill="#6b7483" fontFamily="ui-monospace, monospace">{fmt(y)}</text>
            </g>
          )
        })}
        {Array.from({ length: nT + 1 }, (_, i) => {
          const x = x0 + (i / nT) * (x1 - x0)
          return <text key={i} x={px(x)} y={H - 8} textAnchor="middle" fontSize="9" fill="#6b7483" fontFamily="ui-monospace, monospace">
            {Math.abs(x) >= 1000 ? x.toExponential(0) : (Number.isInteger(x) ? x : x.toFixed(1))}
          </text>
        })}
        {markers?.map(m => (
          <g key={m.label}>
            <line x1={px(m.x)} x2={px(m.x)} y1={T} y2={H - B} stroke="#a78bfa" strokeWidth="1" strokeDasharray="3 3" />
            <text x={px(m.x) + 4} y={T + 10} fontSize="9" fill="#a78bfa">{m.label}</text>
          </g>
        ))}
        {series.map(s => (
          <polyline key={s.name} fill="none" stroke={s.color} strokeWidth="1.8"
            strokeDasharray={s.dashed ? '4 3' : undefined}
            points={s.points.map(p => `${px(p[0])},${py(p[1])}`).join(' ')} />
        ))}
        {yLabel && <text x={4} y={12} fontSize="9" fill="#6b7483">{yLabel}</text>}
        {xLabel && <text x={W - R} y={H - 8} textAnchor="end" fontSize="9" fill="#6b7483">{xLabel}</text>}
      </svg>
      {series.length > 1 && (
        <div className="mt-1 flex flex-wrap gap-3">
          {series.map(s => (
            <span key={s.name} className="flex items-center gap-1.5 text-[11px] text-[var(--dim)]">
              <span className="inline-block h-[2px] w-4" style={{ background: s.color }} />{s.name}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
