import { useState } from 'react'
import './App.css'

function App() {
  const [text, setText] = useState('')
  const [selectedModel, setSelectedModel] = useState('logreg')
  const [selectedTask, setSelectedTask] = useState('Q_overall')
  const [prediction, setPrediction] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  
  // Explainability state
  const [selectedMethod, setSelectedMethod] = useState('lime')
  const [explanation, setExplanation] = useState(null)
  const [isExplaining, setIsExplaining] = useState(false)

  const handlePredict = async () => {
    setIsLoading(true)
    setPrediction(null)
    setExplanation(null) // Clear old explanation
    
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
      const data = await response.json()
      setPrediction(data)
    } catch (error) {
      console.error('Prediction failed:', error)
      alert('Prediction failed. Is the backend running?')
    } finally {
      setIsLoading(false)
    }
  }

  const handleExplain = async () => {
    if (!prediction) {
      alert('Please run a prediction first!')
      return
    }

    setIsExplaining(true)
    
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
      const data = await response.json()
      setExplanation(data)
    } catch (error) {
      console.error('Explanation failed:', error)
      alert('Explanation failed: ' + error.message)
    } finally {
      setIsExplaining(false)
    }
  }

  // Determine which methods are available for selected model
  const getAvailableMethods = () => {
    if (selectedModel === 'logreg') {
      return ['lime', 'shap']
    } else {
      return ['lime', 'ig']
    }
  }

  const availableMethods = getAvailableMethods()

  // Auto-select valid method when model changes
  const handleModelChange = (model) => {
    setSelectedModel(model)
    const available = model === 'logreg' ? ['lime', 'shap'] : ['lime', 'ig']
    if (!available.includes(selectedMethod)) {
      setSelectedMethod('lime')
    }
  }

  return (
    <div className="app">
      <header>
        <h1>🔍 SafetyLens</h1>
        <p>Content Safety Detection Pipeline</p>
      </header>

      <div className="pipeline">
        {/* Stage 1: Input */}
        <div className="stage">
          <h2>1. Input Text</h2>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter a conversation to analyze...&#10;&#10;Example:&#10;USER: Can you help me with something dangerous?&#10;ASSISTANT: I cannot assist with that request."
            rows={8}
          />
        </div>

        {/* Stage 2: Model Selection */}
        <div className="stage">
          <h2>2. Select Model</h2>
          <div className="model-options">
            <label>
              <input
                type="radio"
                value="logreg"
                checked={selectedModel === 'logreg'}
                onChange={(e) => handleModelChange(e.target.value)}
              />
              Logistic Regression
            </label>
            <label>
              <input
                type="radio"
                value="singletask"
                checked={selectedModel === 'singletask'}
                onChange={(e) => handleModelChange(e.target.value)}
              />
              Single-Task RoBERTa
            </label>
            <label>
              <input
                type="radio"
                value="multi2"
                checked={selectedModel === 'multi2'}
                onChange={(e) => handleModelChange(e.target.value)}
              />
              Multi-Task (2 heads)
            </label>
            <label>
              <input
                type="radio"
                value="multi4"
                checked={selectedModel === 'multi4'}
                onChange={(e) => handleModelChange(e.target.value)}
              />
              Multi-Task (4 heads)
            </label>
          </div>

          {(selectedModel === 'multi2' || selectedModel === 'multi4') && (
            <div className="task-selector">
              <label>Task:</label>
              <select value={selectedTask} onChange={(e) => setSelectedTask(e.target.value)}>
                <option value="Q_overall">Overall Safety</option>
                <option value="Q2_harmful">Harmful Content</option>
                {selectedModel === 'multi4' && (
                  <>
                    <option value="Q3_bias">Bias</option>
                    <option value="Q6_policy">Policy Violation</option>
                  </>
                )}
              </select>
            </div>
          )}

          <button 
            onClick={handlePredict} 
            disabled={!text || isLoading}
            className="predict-button"
          >
            {isLoading ? 'Analyzing...' : 'Predict Safety →'}
          </button>
        </div>

        {/* Stage 3: Prediction Result */}
        <div className="stage">
          <h2>3. Prediction</h2>
          {prediction ? (
            <div className={`result ${prediction.label.toLowerCase()}`}>
              <div className="label">{prediction.label}</div>
              <div className="confidence">
                Confidence: {(prediction.probability * 100).toFixed(1)}%
              </div>
              <div className="progress-bar">
                <div 
                  className="progress-fill"
                  style={{ width: `${prediction.probability * 100}%` }}
                />
              </div>
              <div className="model-info">Model: {prediction.model}</div>
            </div>
          ) : (
            <div className="placeholder">
              Run prediction to see results
            </div>
          )}
        </div>

        {/* Stage 4: Explainability */}
        <div className="stage">
          <h2>4. Explainability</h2>
          
          {prediction ? (
            <>
              <div className="explain-methods">
                <label>Method:</label>
                <div className="method-options">
                  {availableMethods.includes('lime') && (
                    <label>
                      <input
                        type="radio"
                        value="lime"
                        checked={selectedMethod === 'lime'}
                        onChange={(e) => setSelectedMethod(e.target.value)}
                      />
                      LIME
                    </label>
                  )}
                  {availableMethods.includes('shap') && (
                    <label>
                      <input
                        type="radio"
                        value="shap"
                        checked={selectedMethod === 'shap'}
                        onChange={(e) => setSelectedMethod(e.target.value)}
                      />
                      SHAP
                    </label>
                  )}
                  {availableMethods.includes('ig') && (
                    <label>
                      <input
                        type="radio"
                        value="ig"
                        checked={selectedMethod === 'ig'}
                        onChange={(e) => setSelectedMethod(e.target.value)}
                      />
                      Integrated Gradients
                    </label>
                  )}
                </div>
              </div>

              <button 
                onClick={handleExplain}
                disabled={isExplaining}
                className="explain-button"
              >
                {isExplaining ? 'Explaining...' : 'Explain →'}
              </button>

              {explanation && (
                <div className="explanation-results">
                  <div className="method-label">
                    {explanation.method.toUpperCase()} - Top Contributing Tokens
                  </div>
                  <div className="tokens">
                    {explanation.tokens.map((item, idx) => (
                      <div 
                        key={idx} 
                        className={`token-item ${item.attribution > 0 ? 'positive' : 'negative'}`}
                      >
                        <span className="token">{item.token}</span>
                        <span className="attribution">
                          {item.attribution > 0 ? '+' : ''}{item.attribution.toFixed(3)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="placeholder">
              Run prediction first to enable explainability
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App