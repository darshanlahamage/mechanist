import { useState, type ReactNode } from 'react'
import { useLocation } from 'react-router-dom'
import Sidebar from './Sidebar'

export default function Layout({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false)
  const loc = useLocation()

  return (
    <div className="flex min-h-full">
      <aside className={`fixed inset-y-0 left-0 z-40 w-[278px] shrink-0 border-r border-[var(--line)] bg-[var(--panel)] transition-transform lg:translate-x-0 ${open ? 'translate-x-0' : '-translate-x-full'}`}>
        <Sidebar onNavigate={() => setOpen(false)} />
      </aside>

      {open && <div className="fixed inset-0 z-30 bg-[#22201b]/40 lg:hidden" onClick={() => setOpen(false)} />}

      <div className="min-w-0 flex-1 lg:ml-[278px]">
        <header className="sticky top-0 z-20 flex items-center gap-3 border-b border-[var(--line)] bg-[var(--bg)]/92 px-4 py-2.5 backdrop-blur lg:hidden">
          <button onClick={() => setOpen(true)}
            className="rounded-md border border-[var(--line)] px-2 py-1 text-[13px]">☰</button>
          <span className="text-[13px] font-semibold">Mechanist</span>
        </header>
        <main key={loc.pathname} className="mx-auto w-full max-w-[1180px] px-5 py-9 lg:px-12 lg:py-12">
          {children}
        </main>
      </div>
    </div>
  )
}
