import { Check } from 'lucide-react'
import { Link } from 'react-router-dom'
const steps = [['Clinical data','/clinical-data'], ['Fundus image','/image-upload'], ['Review','/review'], ['Analysis','/analysis'], ['Results','/results']]
export default function Stepper({ current = 1 }) { return <ol className="stepper">{steps.map(([label, to], index) => <li className={index + 1 <= current ? 'active' : ''} key={to}><Link to={index + 1 < current ? to : '#'} aria-current={index + 1 === current ? 'step' : undefined}><span>{index + 1 < current ? <Check size={14}/> : index + 1}</span><small>{label}</small></Link></li>)}</ol> }
