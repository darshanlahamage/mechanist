import { Link, NavLink, useLocation } from 'react-router-dom'
import { useState, type ReactNode } from 'react'
import { TRACKS } from '../content/registry'

export default function Layout({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false)
  const loc = useLocation()

  return (
    <div className="flex min-h-full">
      {/* sidebar */}
      <aside className={`fixed inset-y-0 left-0 z-40 w-[272px] shrink-0 overflow-y-auto border-r border-[var(--line)] bg-[var(--panel)] transition-transform lg:static lg:translate-x-0 ${open ? 'translate-x-0' : '-translate-x-full'}`}>
        <Link to="/" className="flex items-center gap-2 border-b border-[var(--line)] px-4 py-3.5">
          <div className="grid h-7 w-7 place-items-center rounded-md bg-[var(--accent)] font-bold text-[#06201c]">M</div>
          <div>
            <div className="text-[14px] font-bold leading-none">Mechanist</div>
            <div className="mt-0.5 text-[10px] text-[var(--dim)]">see how AI actually works</div>
          </div>
        </Link>

        <nav className="px-2 py-3">
          {TRACKS.map(t => {
            const built = t.concepts.filter(c => c.status !== 'planned')
            return (
              <div key={t.id} className="mb-4">
                <div className="flex items-baseline justify-between px-2 py-1">
                  <Link to={`/track/${t.id}`} className="text-[11px] font-bold uppercase tracking-wider text-[var(--dim)] hover:text-[var(--accent)]">
                    {t.title}
                  </Link>
                  <span className="font-mono text-[10px] text-[#4c5563]">{built.length}/{t.concepts.length}</span>
                </div>
                {t.concepts.map(c => (
                  <NavLink key={c.id} to={`/c/${c.id}`} onClick={() => setOpen(false)}
                    className={({ isActive }) =>
                      `flex items-center gap-2 rounded-md px-2 py-[5px] text-[13px] transition ${
                        isActive ? 'bg-[var(--accent)]/12 text-[var(--accent)]'
                        : c.status === 'planned' ? 'text-[#5b6472] hover:bg-white/[0.03]'
                        : 'text-[#b6bec9] hover:bg-white/[0.04] hover:text-white'}`}>
                    <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                      c.status === 'stable' ? 'bg-[var(--accent)]'
                      : c.status === 'draft' ? 'bg-[var(--warn)]' : 'bg-[#39414e]'}`} />
                    <span className="truncate">{c.title}</span>
                    {c.hasLab && <span className="ml-auto shrink-0 text-[9px] font-bold tracking-wide text-[var(--accent-2)]">LAB</span>}
                  </NavLink>
                ))}
              </div>
            )
          })}
        </nav>
      </aside>

      {open && <div className="fixed inset-0 z-30 bg-black/50 lg:hidden" onClick={() => setOpen(false)} />}

      <div className="min-w-0 flex-1">
        <header className="sticky top-0 z-20 flex items-center gap-3 border-b border-[var(--line)] bg-[var(--bg)]/90 px-4 py-2.5 backdrop-blur lg:hidden">
          <button onClick={() => setOpen(true)} className="rounded-md border border-[var(--line)] px-2 py-1 text-[13px]">☰</button>
          <span className="text-[13px] font-semibold">Mechanist</span>
        </header>
        <main key={loc.pathname} className="mx-auto w-full max-w-[940px] px-5 py-8 lg:px-10 lg:py-10">
          {children}
        </main>
      </div>
    </div>
  )
}
