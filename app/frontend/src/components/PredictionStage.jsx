function PredictionStage({ prediction, isLoading }) {
  if (isLoading) {
    return (
      <div className="stage">
        <h2>3. Prediction</h2>
        <div className="loading-state">
          <div className="spinner-large"></div>
          <p>Analyzing safety...</p>
        </div>
      </div>
    )
  }

  if (!prediction) {
    return (
      <div className="stage">
        <h2>3. Prediction</h2>
        <div className="placeholder">
          <div className="placeholder-icon">⊙</div>
          <p>Run prediction to see results</p>
        </div>
      </div>
    )
  }

  const confidencePercent = (prediction.probability * 100).toFixed(1)
  const isHighConfidence = prediction.probability > 0.7 || prediction.probability < 0.3

  return (
    <div className="stage">
      <h2>3. Prediction</h2>
      <div className={`result ${prediction.label.toLowerCase()}`}>
        <div className="result-header">
          <div className="label-badge">
            <span className="label-icon">
              {prediction.label === 'Safe' ? '◉' : '◈'}
            </span>
            {prediction.label}
          </div>
          <div className="confidence-badge">
            <span className="badge-icon">{isHighConfidence ? '●' : '○'}</span>
            {confidencePercent}%
          </div>
        </div>

        <div className="progress-bar">
          <div 
            className="progress-fill"
            style={{ width: `${confidencePercent}%` }}
          />
        </div>

        <div className="result-details">
          <div className="detail-item">
            <span className="detail-label">Model:</span>
            <span className="detail-value">{prediction.model}</span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Confidence:</span>
            <span className="detail-value">
              {isHighConfidence ? 'High' : 'Low'}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default PredictionStage