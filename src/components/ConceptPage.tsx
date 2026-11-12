import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { CONCEPTS, trackOf } from '../content/registry'
import { LENSES, type LensId } from '../content/types'

export default function ConceptPage() {
  const { id } = useParams()
  const c = CONCEPTS.find(x => x.id === id)
  const [lens, setLens] = useState<LensId>('intuition')

  if (!c) return <div className="text-[var(--dim)]">Not found. <Link className="text-[var(--accent)]" to="/">Go home</Link></div>

  const track = trackOf(c.id)
  const available = LENSES.filter(l => c.lenses?.[l.id])
  const active = c.lenses?.[lens] ? lens : available[0]?.id
  const Body = active ? c.lenses?.[active] : undefined

  return (
    <article>
      <div className="mb-1 flex items-center gap-2 text-[12px]">
        <Link to={`/track/${track?.id}`} className="text-[var(--accent)] hover:underline">{track?.title}</Link>
        <span className="text-[#39414e]">/</span>
        <span className="text-[var(--dim)]">difficulty {'●'.repeat(c.difficulty)}<span className="text-[#39414e]">{'●'.repeat(5 - c.difficulty)}</span></span>
      </div>

      <h1 className="text-[30px] font-bold leading-tight tracking-tight">{c.title}</h1>
      <p className="mt-2 text-[16px] leading-relaxed text-[var(--dim)]">{c.tagline}</p>

      {c.prereqs && c.prereqs.length > 0 && (
        <div className="mt-4 flex flex-wrap items-center gap-2 text-[12px]">
          <span className="text-[var(--dim)]">You need first:</span>
          {c.prereqs.map(p => {
            const pc = CONCEPTS.find(x => x.id === p)
            return pc
              ? <Link key={p} to={`/c/${p}`} className="rounded-full border border-[var(--line)] px-2.5 py-0.5 text-[var(--text)] hover:border-[var(--accent)] hover:text-[var(--accent)]">{pc.title}</Link>
              : null
          })}
        </div>
      )}

      {available.length === 0 ? (
        <div className="mt-8 rounded-xl border border-dashed border-[var(--line)] p-8 text-center">
          <div className="text-[15px] font-semibold">Not written yet</div>
          <p className="mx-auto mt-2 max-w-[420px] text-[13px] leading-relaxed text-[var(--dim)]">
            This concept is on the map but has no pages yet. Each one needs six views:
            intuition, mechanism, math, code, systems cost, and how it fails.
          </p>
          <a href="https://github.com/darshanlahamage/mechanist" target="_blank" rel="noreferrer"
             className="mt-4 inline-block rounded-md bg-[var(--accent)] px-3 py-1.5 text-[13px] font-semibold text-[#06201c]">
            Write it on GitHub
          </a>
        </div>
      ) : (
        <>
          <div className="sticky top-0 z-10 mt-7 -mx-2 flex gap-1 overflow-x-auto border-b border-[var(--line)] bg-[var(--bg)]/95 px-2 pb-0 pt-2 backdrop-blur">
            {LENSES.map(l => {
              const has = !!c.lenses?.[l.id]
              return (
                <button key={l.id} disabled={!has} onClick={() => setLens(l.id)}
                  title={l.question}
                  className={`shrink-0 border-b-2 px-3 py-2 text-[13px] transition ${
                    active === l.id ? 'border-[var(--accent)] font-semibold text-[var(--accent)]'
                    : has ? 'border-transparent text-[#9aa4b2] hover:text-white'
                    : 'border-transparent text-[#454d5a]'}`}>
                  {l.label}
                </button>
              )
            })}
          </div>
          <div className="mt-2 text-[12px] italic text-[var(--dim)]">
            {LENSES.find(l => l.id === active)?.question}
          </div>
          <div className="prose-m mt-3">{Body && <Body />}</div>
        </>
      )}

      {c.papers && c.papers.length > 0 && (
        <div className="mt-10 border-t border-[var(--line)] pt-6">
          <h3 className="mb-3 text-[13px] font-bold uppercase tracking-wider text-[var(--dim)]">Where this came from</h3>
          <div className="space-y-2">
            {c.papers.map(p => (
              <a key={p.url} href={p.url} target="_blank" rel="noreferrer"
                 className="block rounded-lg border border-[var(--line)] bg-[var(--panel)] p-3 transition hover:border-[var(--accent)]">
                <div className="flex items-baseline gap-2">
                  <span className="text-[13px] font-semibold text-[var(--text)]">{p.title}</span>
                  <span className="font-mono text-[11px] text-[var(--dim)]">{p.year}</span>
                </div>
                <div className="mt-1 text-[12.5px] leading-relaxed text-[var(--dim)]">{p.idea}</div>
              </a>
            ))}
          </div>
        </div>
      )}
    </article>
  )
}
