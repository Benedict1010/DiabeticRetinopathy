import { Activity, Clock3, Home, Info, Menu, PlusCircle, ScanEye, X } from 'lucide-react'
import { NavLink } from 'react-router-dom'
import { useState } from 'react'

const links = [
  ['/', 'Dashboard', Home], ['/clinical-data', 'New analysis', PlusCircle],
  ['/history', 'Demo history', Clock3], ['/about', 'About prototype', Info],
]
export default function Sidebar() {
  const [open, setOpen] = useState(false)
  return <><button className="mobile-menu" onClick={() => setOpen(!open)} aria-label="Toggle navigation">{open ? <X /> : <Menu />}</button>
    <aside className={`sidebar ${open ? 'sidebar-open' : ''}`}>
      <NavLink to="/" className="brand" onClick={() => setOpen(false)}><span className="brand-mark"><ScanEye /></span><span>Retina<span>View</span><small>SCREENING PROTOTYPE</small></span></NavLink>
      <nav aria-label="Primary navigation">{links.map(([to, label, Icon]) => <NavLink end={to === '/'} key={to} to={to} onClick={() => setOpen(false)}><Icon size={18} />{label}</NavLink>)}</nav>
      <div className="sidebar-bottom"><div className="demo-badge"><Activity size={16} /> Demo mode</div><p>For education and interface demonstration only.</p></div>
    </aside></>
}
