import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { CONCEPTS, TRACKS, trackOf } from '../content/registry'
import { LENSES } from '../content/types'

export default function ConceptPage() {
  const { id } = useParams()
  const c = CONCEPTS.find(x => x.id === id)
  const [active, setActive] = useState<string>('')
  const bodyRef = useRef<HTMLDivElement>(null)

  const nav = useMemo(() => {
    const flat = TRACKS.flatMap(t => t.concepts)
    const i = flat.findIndex(x => x.id === id)
    return { prev: flat[i - 1], next: flat[i + 1], index: i }
  }, [id])

  const sections = useMemo(() => LENSES.filter(l => c?.lenses?.[l.id]), [c])

  // highlight the heading you are currently reading
  useEffect(() => {
    if (!bodyRef.current) return
    const obs = new IntersectionObserver(
      entries => {
        const vis = entries.filter(e => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)
        if (vis[0]) setActive(vis[0].target.id)
      },
      { rootMargin: '-80px 0px -65% 0px' })
    bodyRef.current.querySelectorAll('section[id]').forEach(s => obs.observe(s))
    return () => obs.disconnect()
  }, [c?.id, sections.length])

  useEffect(() => { window.scrollTo(0, 0) }, [id])

  if (!c) return <div className="text-[var(--dim)]">Not found. <Link className="text-[var(--accent)]" to="/">Go home</Link></div>

  const track = trackOf(c.id)
  const trackIdx = TRACKS.findIndex(t => t.id === track?.id)
  const lessonIdx = track?.concepts.findIndex(x => x.id === c.id) ?? 0
  const Hero = c.hero

  return (
    <div className="flex gap-10">
      <article className="min-w-0 flex-1">
        <div className="flex items-center gap-2 text-[12px]">
          <Link to={`/track/${track?.id}`} className="text-[var(--accent)] hover:underline">{track?.title}</Link>
          <span className="text-[#39414e]">·</span>
          <span className="font-mono text-[var(--dim)]">lesson {trackIdx + 1}.{lessonIdx + 1}</span>
        </div>

        <h1 className="mt-2 text-[33px] font-bold leading-[1.12] tracking-tight">{c.title}</h1>
        <p className="mt-3 max-w-[62ch] text-[17px] leading-relaxed text-[#aab3c0]">{c.tagline}</p>

        {c.prereqs && c.prereqs.length > 0 && (
          <div className="mt-5 flex flex-wrap items-center gap-2 text-[12px]">
            <span className="text-[var(--dim)]">Read first:</span>
            {c.prereqs.map(p => {
              const pc = CONCEPTS.find(x => x.id === p)
              return pc ? (
                <Link key={p} to={`/c/${p}`}
                  className="rounded-full border border-[var(--line)] px-2.5 py-0.5 hover:border-[var(--accent)] hover:text-[var(--accent)]">
                  {pc.title}
                </Link>) : null
            })}
          </div>
        )}

        {Hero && <div className="mt-7"><Hero /></div>}

        {sections.length === 0 ? (
          <div className="mt-9 rounded-xl border border-dashed border-[var(--line)] p-9 text-center">
            <div className="text-[15px] font-semibold">Not written yet</div>
            <p className="mx-auto mt-2 max-w-[440px] text-[13px] leading-relaxed text-[var(--dim)]">
              This lesson is on the map but has no pages yet. It needs the idea, an animation of
              what actually happens, the maths, the code, what it costs and how it breaks.
            </p>
            <a href="https://github.com/darshanlahamage/mechanist" target="_blank" rel="noreferrer"
              className="mt-5 inline-block rounded-md bg-[var(--accent)] px-3.5 py-2 text-[13px] font-semibold text-[#06201c]">
              Write it on GitHub
            </a>
          </div>
        ) : (
          <div ref={bodyRef} className="mt-2">
            {sections.map((l, i) => {
              const Body = c.lenses![l.id]!
              return (
                <section key={l.id} id={l.id} className="scroll-mt-20 pt-9">
                  <div className="mb-4 flex items-center gap-3">
                    <span className="font-mono text-[11px] text-[#4c5563]">{String(i + 1).padStart(2, '0')}</span>
                    <h2 className="text-[20px] font-bold tracking-tight text-white">{l.label}</h2>
                    <span className="h-px flex-1 bg-[var(--line)]" />
                  </div>
                  <div className="prose-m max-w-[68ch]"><Body /></div>
                </section>
              )
            })}
          </div>
        )}

        {c.papers && c.papers.length > 0 && (
          <section className="mt-14 border-t border-[var(--line)] pt-7">
            <h2 className="mb-4 text-[14px] font-bold uppercase tracking-wider text-[var(--dim)]">Where this came from</h2>
            <div className="space-y-2">
              {c.papers.map(p => (
                <a key={p.url} href={p.url} target="_blank" rel="noreferrer"
                  className="block rounded-lg border border-[var(--line)] bg-[var(--panel)] p-3.5 transition hover:border-[var(--accent)]">
                  <div className="flex items-baseline gap-2">
                    <span className="text-[13.5px] font-semibold">{p.title}</span>
                    <span className="font-mono text-[11px] text-[var(--dim)]">{p.year}</span>
                  </div>
                  <div className="mt-1 text-[12.5px] leading-relaxed text-[var(--dim)]">{p.idea}</div>
                </a>
              ))}
            </div>
          </section>
        )}

        <nav className="mt-12 flex gap-3 border-t border-[var(--line)] pt-6">
          {nav.prev && (
            <Link to={`/c/${nav.prev.id}`}
              className="group flex-1 rounded-lg border border-[var(--line)] p-3.5 transition hover:border-[var(--accent)]">
              <div className="text-[11px] text-[var(--dim)]">← previous</div>
              <div className="mt-0.5 truncate text-[13.5px] font-medium group-hover:text-[var(--accent)]">{nav.prev.title}</div>
            </Link>
          )}
          {nav.next && (
            <Link to={`/c/${nav.next.id}`}
              className="group flex-1 rounded-lg border border-[var(--line)] p-3.5 text-right transition hover:border-[var(--accent)]">
              <div className="text-[11px] text-[var(--dim)]">next →</div>
              <div className="mt-0.5 truncate text-[13.5px] font-medium group-hover:text-[var(--accent)]">{nav.next.title}</div>
            </Link>
          )}
        </nav>
      </article>

      {sections.length > 1 && (
        <aside className="sticky top-8 hidden h-fit w-[168px] shrink-0 xl:block">
          <div className="mb-2 text-[10.5px] font-bold uppercase tracking-wider text-[#4c5563]">On this page</div>
          {sections.map(l => (
            <a key={l.id} href={`#${l.id}`}
              className={`block border-l py-[5px] pl-2.5 text-[12px] transition ${
                active === l.id ? 'border-[var(--accent)] text-[var(--accent)]'
                : 'border-[#232a34] text-[#7a8494] hover:text-white'}`}>
              {l.label}
            </a>
          ))}
        </aside>
      )}
    </div>
  )
}
