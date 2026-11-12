import { Link } from 'react-router-dom'
import { BUILT, CONCEPTS, TRACKS } from '../content/registry'

export default function Home() {
  const labs = CONCEPTS.filter(c => c.hasLab)
  return (
    <div>
      <h1 className="text-[34px] font-bold leading-[1.15] tracking-tight">
        See how AI models<br />actually work.
      </h1>
      <p className="mt-4 max-w-[640px] text-[16px] leading-relaxed text-[var(--dim)]">
        Most material tells you what a concept is called. This one shows you what happens to the
        numbers, why the formula has that shape, what it costs to run, and how it breaks. Every
        visual here is real math running in your browser — no drawn animations, no made-up values.
      </p>

      <div className="mt-6 flex flex-wrap gap-3">
        <Link to="/c/attention" className="rounded-md bg-[var(--accent)] px-4 py-2 text-[14px] font-semibold text-[#06201c]">
          Start with attention
        </Link>
        <a href="https://github.com/darshanlahamage/mechanist" target="_blank" rel="noreferrer"
           className="rounded-md border border-[var(--line)] px-4 py-2 text-[14px] font-semibold hover:border-[var(--accent)]">
          Contribute on GitHub
        </a>
      </div>

      <div className="mt-8 grid grid-cols-3 gap-3">
        {[['Concepts mapped', String(CONCEPTS.length)],
          ['Written so far', String(BUILT.length)],
          ['Interactive labs', String(labs.length)]].map(([l, v]) => (
          <div key={l} className="rounded-lg border border-[var(--line)] bg-[var(--panel)] px-3 py-2.5">
            <div className="font-mono text-[20px] font-semibold text-[var(--accent)]">{v}</div>
            <div className="text-[11px] text-[var(--dim)]">{l}</div>
          </div>
        ))}
      </div>

      <h2 className="mt-12 text-[13px] font-bold uppercase tracking-wider text-[var(--dim)]">Try these first</h2>
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        {labs.map(c => (
          <Link key={c.id} to={`/c/${c.id}`}
                className="group rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4 transition hover:border-[var(--accent)]">
            <div className="flex items-center gap-2">
              <span className="rounded bg-[var(--accent-2)]/15 px-1.5 py-0.5 text-[9px] font-bold tracking-wider text-[var(--accent-2)]">LAB</span>
              <span className="text-[14px] font-semibold group-hover:text-[var(--accent)]">{c.title}</span>
            </div>
            <p className="mt-1.5 text-[13px] leading-relaxed text-[var(--dim)]">{c.tagline}</p>
          </Link>
        ))}
      </div>

      <h2 className="mt-12 text-[13px] font-bold uppercase tracking-wider text-[var(--dim)]">Every track</h2>
      <div className="mt-3 space-y-2">
        {TRACKS.map(t => {
          const done = t.concepts.filter(c => c.status !== 'planned').length
          return (
            <Link key={t.id} to={`/track/${t.id}`}
                  className="block rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4 transition hover:border-[var(--accent)]">
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-[14px] font-semibold">{t.title}</span>
                <span className="shrink-0 font-mono text-[11px] text-[var(--dim)]">{done}/{t.concepts.length} written</span>
              </div>
              <p className="mt-1 text-[13px] leading-relaxed text-[var(--dim)]">{t.blurb}</p>
            </Link>
          )
        })}
      </div>

      <div className="mt-12 rounded-xl border border-dashed border-[var(--line)] p-5">
        <h3 className="text-[14px] font-semibold">This is early, and open</h3>
        <p className="mt-2 text-[13px] leading-relaxed text-[var(--dim)]">
          The map is large on purpose — it is the plan, in public. Grey entries are not written yet.
          Every concept needs six views: the idea, what happens to the numbers, the math, the code,
          what it costs to run, and how it fails. Pick one and write it.
        </p>
      </div>
    </div>
  )
}
