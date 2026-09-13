import { CircleCheckBig, Sparkles } from 'lucide-react'
import Card from '../common/Card'
import ConfidenceIndicator from './ConfidenceIndicator'
export default function PredictionCard({ result }) { return <Card className="prediction-card"><div className="prediction-copy"><span className="eyebrow"><Sparkles size={15}/> MOCK SCREENING OUTPUT</span><h2>{result.label}</h2><p>This is a simulated result for interface testing. It is not generated from the uploaded image or clinical information.</p><div className="prediction-meta"><CircleCheckBig size={18}/><span>Demo workflow completed</span></div></div><ConfidenceIndicator value={result.confidence}/></Card> }
