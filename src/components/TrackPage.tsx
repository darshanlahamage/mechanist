import { Link, useParams } from 'react-router-dom'
import { TRACKS } from '../content/registry'

export default function TrackPage() {
  const { id } = useParams()
  const t = TRACKS.find(x => x.id === id)
  if (!t) return <div className="text-[var(--dim)]">No such track.</div>

  return (
    <div>
      <h1 className="text-[28px] font-bold tracking-tight">{t.title}</h1>
      <p className="mt-2 max-w-[640px] text-[15px] leading-relaxed text-[var(--dim)]">{t.blurb}</p>
      <div className="mt-6 space-y-2">
        {t.concepts.map(c => (
          <Link key={c.id} to={`/c/${c.id}`}
                className="block rounded-lg border border-[var(--line)] bg-[var(--panel)] p-3.5 transition hover:border-[var(--accent)]">
            <div className="flex items-center gap-2">
              <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                c.status === 'stable' ? 'bg-[var(--accent)]' : c.status === 'draft' ? 'bg-[var(--warn)]' : 'bg-[#cec4ae]'}`} />
              <span className={`text-[14px] font-semibold ${c.status === 'planned' ? 'text-[#7b7364]' : ''}`}>{c.title}</span>
              {c.hasLab && <span className="rounded bg-[var(--accent-2)]/15 px-1.5 py-0.5 text-[9px] font-bold tracking-wider text-[var(--accent-2)]">LAB</span>}
              <span className="ml-auto shrink-0 font-mono text-[10px] text-[#8f8775]">
                {c.status === 'planned' ? 'not written' : 'difficulty ' + c.difficulty}
              </span>
            </div>
            <p className="mt-1 pl-3.5 text-[12.5px] leading-relaxed text-[var(--dim)]">{c.tagline}</p>
          </Link>
        ))}
      </div>
    </div>
  )
}
