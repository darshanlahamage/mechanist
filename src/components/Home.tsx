import { Link } from 'react-router-dom'
import { BUILT, CONCEPTS, TRACKS } from '../content/registry'

const ANIMATED = ['attention', 'rope', 'diffusion-forward', 'moe']

export default function Home() {
  const animated = CONCEPTS.filter(c => ANIMATED.includes(c.id))
  const rest = BUILT.filter(c => !ANIMATED.includes(c.id))

  return (
    <div className="max-w-[820px]">
      <h1 className="text-[36px] font-bold leading-[1.1] tracking-tight">
        Watch what actually<br />happens inside the model.
      </h1>
      <p className="mt-5 max-w-[60ch] text-[16.5px] leading-relaxed text-[#aab3c0]">
        Most material tells you what a thing is called. Here you press play and watch the numbers
        move — a query meeting every key, a vector rotating with its position, an image emerging
        from noise, tokens being routed to experts. Every frame is computed from the real maths as
        it runs. Nothing is drawn by hand.
      </p>
      <p className="mt-3 max-w-[60ch] text-[16.5px] leading-relaxed text-[#aab3c0]">
        Each lesson then reads straight through: the idea, the animation, the maths, the code, what
        it costs to run, and how it breaks. One page, no tabs, in order.
      </p>

      <div className="mt-7 flex flex-wrap gap-3">
        <Link to="/c/attention"
          className="rounded-md bg-[var(--accent)] px-4 py-2.5 text-[14px] font-semibold text-[#06201c]">
          Start at attention →
        </Link>
        <a href="https://github.com/darshanlahamage/mechanist" target="_blank" rel="noreferrer"
          className="rounded-md border border-[var(--line)] px-4 py-2.5 text-[14px] font-semibold hover:border-[var(--accent)]">
          Contribute on GitHub
        </a>
      </div>

      <h2 className="mt-14 text-[12px] font-bold uppercase tracking-wider text-[var(--dim)]">
        Lessons that play
      </h2>
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        {animated.map(c => (
          <Link key={c.id} to={`/c/${c.id}`}
            className="group rounded-xl border border-[#2b3340] bg-[var(--panel)] p-4 transition hover:border-[var(--accent)]">
            <div className="flex items-center gap-2">
              <span className="grid h-5 w-5 place-items-center rounded-full bg-[var(--accent)]/15 text-[9px] text-[var(--accent)]">▶</span>
              <span className="text-[14.5px] font-semibold group-hover:text-[var(--accent)]">{c.title}</span>
            </div>
            <p className="mt-1.5 text-[13px] leading-relaxed text-[var(--dim)]">{c.tagline}</p>
          </Link>
        ))}
      </div>

      <h2 className="mt-12 text-[12px] font-bold uppercase tracking-wider text-[var(--dim)]">
        Also written
      </h2>
      <div className="mt-3 flex flex-wrap gap-2">
        {rest.map(c => (
          <Link key={c.id} to={`/c/${c.id}`}
            className="rounded-full border border-[var(--line)] px-3 py-1.5 text-[12.5px] text-[#b6bec9] transition hover:border-[var(--accent)] hover:text-[var(--accent)]">
            {c.title}
          </Link>
        ))}
      </div>

      <h2 className="mt-12 text-[12px] font-bold uppercase tracking-wider text-[var(--dim)]">
        The path, end to end
      </h2>
      <p className="mt-2 max-w-[62ch] text-[13.5px] leading-relaxed text-[var(--dim)]">
        Fourteen parts, in the order they build on each other. The sidebar follows the same order —
        open a part to see its lessons. Greyed lessons are mapped but not written yet, and that list
        doubles as the to-do list for anyone who wants to help.
      </p>
      <div className="mt-4 space-y-1.5">
        {TRACKS.map((t, i) => {
          const done = t.concepts.filter(c => c.status !== 'planned').length
          return (
            <Link key={t.id} to={`/track/${t.id}`}
              className="flex items-center gap-3 rounded-lg border border-[var(--line)] bg-[var(--panel)] px-4 py-3 transition hover:border-[var(--accent)]">
              <span className="font-mono text-[11px] text-[#4c5563]">{String(i + 1).padStart(2, '0')}</span>
              <span className="flex-1 text-[13.5px] font-medium">{t.title}</span>
              <span className="h-1 w-20 overflow-hidden rounded-full bg-[#1f252e]">
                <span className="block h-full rounded-full bg-[var(--accent)]"
                  style={{ width: `${(done / t.concepts.length) * 100}%` }} />
              </span>
              <span className="w-10 shrink-0 text-right font-mono text-[10.5px] text-[var(--dim)]">
                {done}/{t.concepts.length}
              </span>
            </Link>
          )
        })}
      </div>
    </div>
  )
}
