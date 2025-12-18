function ExplainStage({
  prediction,
  selectedModel,
  selectedMethod,
  setSelectedMethod,
  explanation,
  isExplaining,
  onExplain
}) {
  if (!prediction) {
    return (
      <div className="stage">
        <h2>4. Explainability</h2>
        <div className="placeholder">
          <div className="placeholder-icon">⚙</div>
          <p>Run prediction first to enable explainability</p>
        </div>
      </div>
    )
  }

  const availableMethods = selectedModel === 'logreg' 
    ? ['lime', 'shap'] 
    : ['lime', 'ig']

  const methodNames = {
    'lime': 'LIME',
    'shap': 'SHAP',
    'ig': 'Integrated Gradients'
  }

  const methodDescriptions = {
    'lime': 'Local Interpretable Model-agnostic Explanations',
    'shap': 'SHapley Additive exPlanations',
    'ig': 'Gradient-based attribution'
  }

  const getSummaryStats = () => {
    if (!explanation) return null
    
    const positive = explanation.tokens.filter(t => t.attribution > 0)
    const negative = explanation.tokens.filter(t => t.attribution < 0)
    
    return {
      positive: positive.length,
      negative: negative.length,
      topPositive: positive[0],
      topNegative: negative[0]
    }
  }

  const stats = getSummaryStats()

  return (
    <div className="stage">
      <h2>4. Explainability</h2>
      
      <div className="explain-methods">
        <label>Method:</label>
        <div className="method-options">
          {availableMethods.map(method => (
            <label key={method} className="method-option">
              <input
                type="radio"
                value={method}
                checked={selectedMethod === method}
                onChange={(e) => setSelectedMethod(e.target.value)}
              />
              <div className="method-info">
                <span className="method-name">{methodNames[method]}</span>
                <span className="method-desc">{methodDescriptions[method]}</span>
              </div>
            </label>
          ))}
        </div>
      </div>

      <button 
        onClick={onExplain}
        disabled={isExplaining}
        className="explain-button"
      >
        {isExplaining ? (
          <>
            <span className="spinner"></span>
            Explaining...
          </>
        ) : (
          <>Generate Explanation →</>
        )}
      </button>

      {isExplaining && (
        <div className="loading-state">
          <div className="spinner-large"></div>
          <p>Computing attributions...</p>
        </div>
      )}

      {explanation && !isExplaining && (
        <div className="explanation-results">
          <div className="method-label">
            {methodNames[explanation.method]} Results
          </div>

          {stats && (
            <div className="explanation-summary">
              <div className="summary-item">
                <span className="summary-label">▲ Unsafe</span>
                <span className="summary-value positive">{stats.positive}</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">▼ Safe</span>
                <span className="summary-value negative">{stats.negative}</span>
              </div>
            </div>
          )}

          <div className="tokens">
            {explanation.tokens.map((item, idx) => (
              <div 
                key={idx} 
                className={`token-item ${item.attribution > 0 ? 'positive' : 'negative'}`}
              >
                <span className="token-rank">#{idx + 1}</span>
                <span className="token">{item.token}</span>
                <span className="attribution">
                  {item.attribution > 0 ? '+' : ''}{item.attribution.toFixed(3)}
                </span>
                <div 
                  className="attribution-bar"
                  style={{ 
                    width: `${Math.abs(item.attribution) * 100}%`,
                    opacity: Math.min(Math.abs(item.attribution) * 2, 1)
                  }}
                />
              </div>
            ))}
          </div>

          <div className="explanation-note">
            <strong>◆ Note:</strong> 
            {explanation.method === 'lime' && ' Positive values push toward "Unsafe" prediction.'}
            {explanation.method === 'shap' && ' SHAP values show feature importance for the prediction.'}
            {explanation.method === 'ig' && ' Gradients show how tokens influence the model output.'}
          </div>
        </div>
      )}
    </div>
  )
}

export default ExplainStage