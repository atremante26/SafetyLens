function ModelStage({ 
  selectedModel, 
  setSelectedModel,
  selectedTask,
  setSelectedTask,
  text,
  isLoading,
  onPredict
}) {
  const handleModelChange = (model) => {
    setSelectedModel(model)
    if (model === 'multi2' && !['Q_overall', 'Q2_harmful'].includes(selectedTask)) {
      setSelectedTask('Q_overall')
    }
  }

  const modelInfo = {
    logreg: {
      name: 'Logistic Regression',
      desc: 'TF-IDF + Logistic Regression',
      details: 'Fast, interpretable baseline model',
      f1: '0.422'
    },
    singletask: {
      name: 'Single-Task RoBERTa',
      desc: 'Binary classifier',
      details: 'Best performing model',
      f1: '0.543'
    },
    multi2: {
      name: 'Multi-Task (2 heads)',
      desc: 'Overall + Harmful',
      details: 'Shared representation learning',
      f1: '0.469'
    },
    multi4: {
      name: 'Multi-Task (4 heads)',
      desc: 'Overall + Harmful + Bias + Policy',
      details: 'Multi-dimensional analysis',
      f1: '0.461'
    }
  }

  return (
    <div className="stage">
      <h2>2. Select Model</h2>
      
      <div className="model-options">
        {Object.entries(modelInfo).map(([key, info]) => (
          <label key={key} className="model-option" title={info.details}>
            <input
              type="radio"
              value={key}
              checked={selectedModel === key}
              onChange={(e) => handleModelChange(e.target.value)}
            />
            <div className="model-info">
              <span className="model-name">{info.name}</span>
              <span className="model-desc">{info.desc}</span>
              <span className="model-f1">F1: {info.f1}</span>
            </div>
          </label>
        ))}
      </div>

      {(selectedModel === 'multi2' || selectedModel === 'multi4') && (
        <div className="task-selector">
          <label>Task:</label>
          <select value={selectedTask} onChange={(e) => setSelectedTask(e.target.value)}>
            <option value="Q_overall">Overall Safety</option>
            <option value="Q2_harmful">Harmful Content</option>
            {selectedModel === 'multi4' && (
              <>
                <option value="Q3_bias">Bias Detection</option>
                <option value="Q6_policy">Policy Violation</option>
              </>
            )}
          </select>
        </div>
      )}

      <button 
        onClick={onPredict} 
        disabled={!text || isLoading}
        className="predict-button"
      >
        {isLoading ? (
          <>
            <span className="spinner"></span>
            Analyzing...
          </>
        ) : (
          'Predict Safety →'
        )}
      </button>
    </div>
  )
}

export default ModelStage