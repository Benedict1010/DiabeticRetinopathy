import Stepper from '../components/analysis/Stepper'
import Card from '../components/common/Card'
import ImageUploadComponent from '../components/analysis/ImageUpload'
export default function ImageUpload() { return <div className="workflow-page"><div className="page-intro"><span className="eyebrow">NEW ANALYSIS</span><h1>Fundus image upload</h1><p>Select a retinal fundus image to include in the mock analysis workflow.</p></div><Stepper current={2}/><Card className="form-card upload-card"><ImageUploadComponent/></Card></div> }
