import { useState, useEffect } from 'react'

function About() {
  return (
    <div className="about-page">
      <div className="about-content">
        {/* Project Overview */}
        <section className="about-section">
          <h2>Project Overview</h2>
          <p>
            SafetyLens is a research project exploring explainable AI for content safety detection. We compare three model architectures—logistic regression, single-task transformers, and multi-task transformers—to examine tradeoffs between model complexity, predictive performance, and interpretability.
          </p>
          <p>
            This project was completed as part of COSC-243: Natural Language Processing at Amherst College (Fall 2025).
          </p>
        </section>

        {/* Research Questions */}
        <section className="about-section">
          <h2>Research Questions</h2>
          <div className="research-questions">
            <div className="question-card">
              <span className="question-number">1</span>
              <p>Can transformers outperform traditional baselines for multidimensional safety detection?</p>
            </div>
            <div className="question-card">
              <span className="question-number">2</span>
              <p>Does multi-task learning improve performance across safety dimensions?</p>
            </div>
            <div className="question-card">
              <span className="question-number">3</span>
              <p>How do models reason differently about high-confidence vs. uncertain predictions?</p>
            </div>
          </div>
        </section>

        {/* Dataset */}
        <section className="about-section">
          <h2>
            Dataset: 
            <a 
              href="https://github.com/google-research-datasets/dices-dataset" 
              target="_blank" 
              rel="noopener noreferrer"
              className="dataset-link"
            >
              DICES-350<span className="external-link-icon">↗</span>
            </a>
          </h2>
          <p>
            We used the DICES-350 dataset, which contains 350 conversations annotated 
            across multiple safety dimensions including overall safety, harmful content, 
            bias, and policy violations.
          </p>
          <div className="dataset-stats">
            <div className="stat-card">
              <div className="stat-value">350</div>
              <div className="stat-label">Unique Conversations</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">4</div>
              <div className="stat-label">Safety Dimensions</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">43,050</div>
              <div className="stat-label">Individual Ratings</div>
            </div>
          </div>
        </section>

        {/* Model Performance */}
        <section className="about-section">
          <h2>Model Performance</h2>

          <div className="performance-table">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Task</th>
                  <th>F1</th>
                  <th>PR-AUC</th>
                  <th>Pos %</th>
                </tr>
              </thead>

              <tbody>
                {/* Q_overall */}
                <tr>
                  <td>Logistic Regression</td>
                  <td>Q_overall</td>
                  <td>0.422</td>
                  <td>0.414</td>
                  <td>35.1%</td>
                </tr>
                <tr>
                  <td>Single-Task</td>
                  <td>Q_overall</td>
                  <td>0.543</td>
                  <td>0.660</td>
                  <td>34.1%</td>
                </tr>
                <tr>
                  <td>Multi-Task (2 Heads)</td>
                  <td>Q_overall</td>
                  <td>0.469</td>
                  <td>0.366</td>
                  <td>32.4%</td>
                </tr>
                <tr>
                  <td>Multi-Task (4 Heads)</td>
                  <td>Q_overall</td>
                  <td>0.461</td>
                  <td>0.370</td>
                  <td>32.4%</td>
                </tr>

                {/* Divider */}
                <tr className="group-divider" aria-hidden="true"><td colSpan={5} /></tr>

                {/* Q2_harmful */}
                <tr>
                  <td>Multi-Task (2 Heads)</td>
                  <td>Q2_harmful</td>
                  <td>0.426</td>
                  <td>0.345</td>
                  <td>17.7%</td>
                </tr>
                <tr>
                  <td>Multi-Task (4 Heads)</td>
                  <td>Q2_harmful</td>
                  <td>0.414</td>
                  <td>0.422</td>
                  <td>17.7%</td>
                </tr>

                {/* Divider */}
                <tr className="group-divider" aria-hidden="true"><td colSpan={5} /></tr>

                {/* Rare tasks */}
                <tr>
                  <td>Multi-Task (4 Heads)</td>
                  <td>Q3_bias</td>
                  <td>0.322</td>
                  <td>0.201</td>
                  <td>9.8%</td>
                </tr>
                <tr>
                  <td>Multi-Task (4 Heads)</td>
                  <td>Q6_policy</td>
                  <td>0.282</td>
                  <td>0.163</td>
                  <td>10.0%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Key Findings */}
        <section className="about-section">
          <h2>Key Findings</h2>
          <div className="findings-grid">
            <div className="finding-card">
              <div className="finding-icon">◈</div>
              <h3>Model Complexity ≠ Performance</h3>
              <p>
                Single-task RoBERTa outperformed multi-task variants on overall safety detection, indicating that added architectural complexity did not improve performance under severe class imbalance.
              </p>
            </div>
            <div className="finding-card">
              <div className="finding-icon">⊙</div>
              <h3>Class Imbalance Matters</h3>
              <p>
                Severe class imbalance disproportionately degraded multi-task performance, especially on rare safety dimensions such as bias and policy violations.
              </p>
            </div>
            <div className="finding-card">
              <div className="finding-icon">⚙</div>
              <h3>Explainability Insights</h3>
              <p>
                Integrated Gradients provided the most reliable token-level explanations for transformer models, while LIME offered qualitative contrast and SHAP proved effective primarily for linear baselines.
              </p>
            </div>
          </div>
        </section>

        {/* Downloads */}
        <section className="about-section">
          <h2>Downloads</h2>
          <div className="downloads-grid">
            <a href="/paper.pdf" className="download-card" download>
              <div className="download-icon">▣</div>
              <div className="download-info">
                <h3>Paper</h3>
                <p>Full technical report (PDF)</p>
              </div>
              <span className="download-arrow">→</span>
            </a>
            <a href="/poster.pdf" className="download-card" download>
              <div className="download-icon">▦</div>
              <div className="download-info">
                <h3>Poster</h3>
                <p>Conference-style poster (PDF)</p>
              </div>
              <span className="download-arrow">→</span>
            </a>
          </div>
        </section>

        {/* Team */}
        <section className="about-section">
          <h2>Credits</h2>
          <div className="team-info">
            <p>
              <strong>Institution:</strong> Amherst College<br/>
              <strong>Course:</strong> COSC-243: Natural Language Processing<br/>
              <strong>Semester:</strong> Fall 2025<br/>
            </p>
            <div className="links">
              <a href="https://github.com/atremante26/SafetyLens" className="link-button">
                <span>◆</span> View on GitHub
              </a>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}

export default About