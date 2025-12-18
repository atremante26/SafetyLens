import { useState, useEffect } from 'react'
import toast, { Toaster } from 'react-hot-toast'
import './App.css'
import InputStage from './components/InputStage'
import ModelStage from './components/ModelStage'
import PredictionStage from './components/PredictionStage'
import ExplainStage from './components/ExplainStage'

function App() {
  const [text, setText] = useState('')
  const [selectedModel, setSelectedModel] = useState('logreg')
  const [selectedTask, setSelectedTask] = useState('Q_overall')
  const [prediction, setPrediction] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  
  const [selectedMethod, setSelectedMethod] = useState('lime')
  const [explanation, setExplanation] = useState(null)
  const [isExplaining, setIsExplaining] = useState(false)

  // Keyboard shortcut: Ctrl+Enter to predict
  useEffect(() => {
    const handleKeyPress = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && text && !isLoading) {
        handlePredict()
      }
    }
    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [text, isLoading])

  const handlePredict = async () => {
    setIsLoading(true)
    setPrediction(null)
    setExplanation(null)
    
    const loadingToast = toast.loading('Analyzing safety...')
    
    try {
      const response = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          model: selectedModel,
          task: selectedTask
        })
      })
      
      if (!response.ok) {
        throw new Error(`Prediction failed: ${response.statusText}`)
      }
      
      const data = await response.json()
      setPrediction(data)
      
      toast.success(`Prediction complete: ${data.label}`, {
        id: loadingToast,
        icon: data.label === 'Safe' ? '◉' : '◈'
      })
    } catch (error) {
      console.error('Prediction failed:', error)
      toast.error('Prediction failed. Check if backend is running.', {
        id: loadingToast,
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleExplain = async () => {
    if (!prediction) {
      toast.error('Please run a prediction first!')
      return
    }

    setIsExplaining(true)
    
    const loadingToast = toast.loading('Computing attributions...')
    
    try {
      const response = await fetch('http://localhost:8000/api/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          model: selectedModel,
          method: selectedMethod,
          task: selectedTask,
          num_features: 10
        })
      })
      
      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Explanation failed')
      }
      
      const data = await response.json()
      setExplanation(data)
      
      toast.success(`${data.method.toUpperCase()} explanation generated!`, {
        id: loadingToast,
        icon: '⚙'
      })
    } catch (error) {
      console.error('Explanation failed:', error)
      toast.error(error.message || 'Explanation failed', {
        id: loadingToast,
      })
    } finally {
      setIsExplaining(false)
    }
  }

  const copyResults = () => {
    if (!prediction) return

    const resultText = `SafetyLens Analysis
Model: ${prediction.model}
Prediction: ${prediction.label}
Confidence: ${(prediction.probability * 100).toFixed(1)}%

Input Text:
${text}
${explanation ? `

Explanation (${explanation.method.toUpperCase()}):
${explanation.tokens.map((t, i) => `${i + 1}. ${t.token}: ${t.attribution.toFixed(3)}`).join('\n')}` : ''}`

    navigator.clipboard.writeText(resultText)
    toast.success('Results copied to clipboard!', {
      icon: '⎘'
    })
  }

  return (
    <div className="app">
      <Toaster 
        position="top-right"
        toastOptions={{
          duration: 3000,
          style: {
            background: '#1a1a1a',
            color: '#e0e0e0',
            border: '1px solid #333',
          },
          success: {
            iconTheme: {
              primary: '#10b981',
              secondary: '#1a1a1a',
            },
          },
          error: {
            iconTheme: {
              primary: '#ef4444',
              secondary: '#1a1a1a',
            },
          },
        }}
      />

      <header>
      <div className="header-content">
        <h1>
          <span className="logo-icon">◆</span>
          SafetyLens
        </h1>
        <p>Multi-Model Content Safety Detection with Explainable AI</p>
      </div>
      
      <div className="header-actions">
        {(prediction || explanation) && (
          <button className="copy-button" onClick={copyResults}>
            <span className="button-icon">⎘</span>
            Copy Results
          </button>
        )}
        <span className="keyboard-hint">
          <span className="hint-icon">⌘</span>
          Press Ctrl+Enter to predict
        </span>
      </div>
    </header>

      <div className="pipeline">
        <InputStage 
          text={text}
          setText={setText}
        />
        
        <ModelStage
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
          selectedTask={selectedTask}
          setSelectedTask={setSelectedTask}
          text={text}
          isLoading={isLoading}
          onPredict={handlePredict}
        />
        
        <PredictionStage
          prediction={prediction}
          isLoading={isLoading}
        />
        
        <ExplainStage
          prediction={prediction}
          selectedModel={selectedModel}
          selectedMethod={selectedMethod}
          setSelectedMethod={setSelectedMethod}
          explanation={explanation}
          isExplaining={isExplaining}
          onExplain={handleExplain}
        />
      </div>
    </div>
  )
}

export default App