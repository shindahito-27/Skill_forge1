import { useRef, useState } from 'react'

const ACCEPTED_TYPES = '.pdf,.docx'

function UploadCard({ onAnalyze, isLoading }) {
  const resumeInputRef = useRef(null)
  const jdInputRef = useRef(null)
  const [selectedResume, setSelectedResume] = useState(null)
  const [selectedJd, setSelectedJd] = useState(null)
  const [dragActive, setDragActive] = useState(false)

  const onFilePicked = (file, type) => {
    if (!file) return
    const fileName = file.name.toLowerCase()
    if (!fileName.endsWith('.pdf') && !fileName.endsWith('.docx')) {
      return
    }
    if (type === 'resume') {
      setSelectedResume(file)
      return
    }
    setSelectedJd(file)
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setDragActive(false)
    const [file] = event.dataTransfer.files || []
    onFilePicked(file, 'resume')
  }

  const handleAnalyze = () => {
    if (!selectedResume || isLoading) return
    onAnalyze({ resumeFile: selectedResume, jdFile: selectedJd })
  }

  return (
    <div className="upload-card">
      <h2>Upload Your Resume</h2>
      <p>Drop a resume and optionally a JD to generate role fit, gap analysis, and roadmap visuals.</p>
      <button className="secondary-btn" onClick={() => resumeInputRef.current?.click()}>
        Choose Resume
      </button>
      <div
        className={`dropzone ${dragActive ? 'drag-active' : ''}`}
        onDragOver={(event) => {
          event.preventDefault()
          setDragActive(true)
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
      >
        <span>{selectedResume ? selectedResume.name : 'Drag and drop resume here'}</span>
      </div>
      <input
        ref={resumeInputRef}
        hidden
        type="file"
        accept={ACCEPTED_TYPES}
        onChange={(event) => onFilePicked(event.target.files?.[0], 'resume')}
      />
      <div className="jd-upload-row">
        <button className="secondary-btn" onClick={() => jdInputRef.current?.click()}>
          Upload Job Description
        </button>
        <span className="jd-file-name">{selectedJd ? selectedJd.name : 'Default JD will be used if left empty'}</span>
      </div>
      <input
        ref={jdInputRef}
        hidden
        type="file"
        accept={ACCEPTED_TYPES}
        onChange={(event) => onFilePicked(event.target.files?.[0], 'jd')}
      />
      <button className="primary-btn" onClick={handleAnalyze} disabled={!selectedResume || isLoading}>
        {isLoading ? 'Analyzing...' : 'Generate Insight Dashboard'}
      </button>
    </div>
  )
}

export default UploadCard
