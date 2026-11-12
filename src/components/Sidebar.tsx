import { useEffect, useRef, useState } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { TRACKS } from '../content/registry'

export default function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const loc = useLocation()
  const currentId = loc.pathname.startsWith('/c/') ? loc.pathname.slice(3) : null
  const currentTrack = TRACKS.find(t => t.concepts.some(c => c.id === currentId))

  const [open, setOpen] = useState<Set<string>>(
    () => new Set(currentTrack ? [currentTrack.id] : [TRACKS[0].id]))
  const activeRef = useRef<HTMLAnchorElement>(null)

  // keep the part containing the current lesson open, and scroll to it
  useEffect(() => {
    if (currentTrack) setOpen(s => new Set(s).add(currentTrack.id))
  }, [currentTrack?.id])
  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: 'nearest' })
  }, [currentId])

  const toggle = (id: string) =>
    setOpen(s => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n })

  const totalDone = TRACKS.reduce((a, t) => a + t.concepts.filter(c => c.status !== 'planned').length, 0)
  const total = TRACKS.reduce((a, t) => a + t.concepts.length, 0)

  return (
    <div className="flex h-full flex-col">
      <Link to="/" onClick={onNavigate}
        className="flex shrink-0 items-center gap-2.5 border-b border-[var(--line)] px-4 py-3.5">
        <div className="grid h-7 w-7 place-items-center rounded-md bg-[var(--accent)] text-[13px] font-bold text-[#dbece8]">M</div>
        <div>
          <div className="text-[14px] font-bold leading-none">Mechanist</div>
          <div className="mt-1 text-[10px] text-[var(--dim)]">{totalDone} of {total} lessons written</div>
        </div>
      </Link>

      <nav className="min-h-0 flex-1 overflow-y-auto px-2 py-2.5">
        {TRACKS.map((t, ti) => {
          const isOpen = open.has(t.id)
          const done = t.concepts.filter(c => c.status !== 'planned').length
          const isCurrent = currentTrack?.id === t.id
          return (
            <div key={t.id} className="mb-0.5">
              <button onClick={() => toggle(t.id)}
                className={`flex w-full items-center gap-2 rounded-md px-2 py-[7px] text-left transition hover:bg-black/[0.035] ${
                  isCurrent ? 'text-[var(--text)]' : 'text-[#6d6659]'}`}>
                <span className={`shrink-0 font-mono text-[10px] ${isCurrent ? 'text-[var(--accent)]' : 'text-[#9c9382]'}`}>
                  {String(ti + 1).padStart(2, '0')}
                </span>
                <span className="flex-1 truncate text-[12.5px] font-semibold">{t.title}</span>
                <span className="shrink-0 font-mono text-[9.5px] text-[#9c9382]">{done}/{t.concepts.length}</span>
                <span className={`shrink-0 text-[9px] text-[#8f8775] transition-transform ${isOpen ? 'rotate-90' : ''}`}>▶</span>
              </button>

              {isOpen && (
                <div className="mb-1.5 ml-[14px] border-l border-[#dcd3c0] pl-1.5">
                  {t.concepts.map((c, ci) => {
                    const active = c.id === currentId
                    return (
                      <NavLink key={c.id} to={`/c/${c.id}`} onClick={onNavigate}
                        ref={active ? activeRef : undefined}
                        className={`flex items-center gap-2 rounded-md px-2 py-[5px] text-[12.5px] transition ${
                          active ? 'bg-[var(--accent)]/12 font-medium text-[var(--accent)]'
                          : c.status === 'planned' ? 'text-[#8f8775] hover:bg-black/[0.03]'
                          : 'text-[#555045] hover:bg-black/[0.035] hover:text-[var(--text)]'}`}>
                        <span className="shrink-0 font-mono text-[9.5px] text-[#9c9382]">{ti + 1}.{ci + 1}</span>
                        <span className="flex-1 truncate">{c.title}</span>
                        {c.hasLab && <span className="shrink-0 text-[9px] text-[var(--accent-2)]">▸</span>}
                      </NavLink>
                    )
                  })}
                </div>
              )}
            </div>
          )
        })}
      </nav>
    </div>
  )
}
