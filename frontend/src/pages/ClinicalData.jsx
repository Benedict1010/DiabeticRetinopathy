import Stepper from '../components/analysis/Stepper'
import Card from '../components/common/Card'
import ClinicalForm from '../components/analysis/ClinicalForm'
export default function ClinicalData() { return <div className="workflow-page"><div className="page-intro"><span className="eyebrow">NEW ANALYSIS</span><h1>Clinical data</h1><p>Add available patient and clinical context for this prototype case.</p></div><Stepper current={1}/><Card className="form-card"><ClinicalForm/></Card></div> }
