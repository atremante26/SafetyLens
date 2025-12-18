function About() {
  return (
    <div className="about-page">
      <div className="about-header">
        <h1>
          <span className="logo-icon">◆</span>
          About SafetyLens
        </h1>
        <p className="subtitle">Multi-Model Content Safety Detection Research</p>
      </div>

      <div className="about-content">
        {/* Project Overview */}
        <section className="about-section">
          <h2>Project Overview</h2>
          <p>
            SafetyLens is a comprehensive research project exploring explainable AI 
            for content safety detection. We compare three model architectures—logistic 
            regression, single-task transformers, and multi-task transformers—to understand 
            the tradeoffs between model complexity, performance, and interpretability.
          </p>
          <p>
            This project was completed as part of [Course Name] at Amherst College in Fall 2024.
          </p>
        </section>

        {/* Research Questions */}
        <section className="about-section">
          <h2>Research Questions</h2>
          <div className="research-questions">
            <div className="question-card">
              <span className="question-number">H1</span>
              <p>Does model architecture complexity correlate with performance?</p>
            </div>
            <div className="question-card">
              <span className="question-number">H2</span>
              <p>Do multi-task models outperform single-task variants?</p>
            </div>
            <div className="question-card">
              <span className="question-number">H3</span>
              <p>How do different explainability methods compare?</p>
            </div>
          </div>
        </section>

        {/* Dataset */}
        <section className="about-section">
          <h2>Dataset: DICES-350</h2>
          <p>
            We used the DICES-350 dataset, which contains 350 conversations annotated 
            across multiple safety dimensions including overall safety, harmful content, 
            bias, and policy violations.
          </p>
          <div className="dataset-stats">
            <div className="stat-card">
              <div className="stat-value">350</div>
              <div className="stat-label">Conversations</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">4</div>
              <div className="stat-label">Safety Dimensions</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">~10%</div>
              <div className="stat-label">Unsafe Examples</div>
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
                  <th>Architecture</th>
                  <th>F1 Score</th>
                  <th>Key Finding</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Logistic Regression</td>
                  <td>TF-IDF + LogReg</td>
                  <td>0.422</td>
                  <td>Fast baseline</td>
                </tr>
                <tr className="best-model">
                  <td>Single-Task RoBERTa</td>
                  <td>Binary Classifier</td>
                  <td>0.543</td>
                  <td>Best performance</td>
                </tr>
                <tr>
                  <td>Multi-Task (2 heads)</td>
                  <td>Shared Encoder</td>
                  <td>0.469</td>
                  <td>Shared learning</td>
                </tr>
                <tr>
                  <td>Multi-Task (4 heads)</td>
                  <td>Shared Encoder</td>
                  <td>0.461</td>
                  <td>Multi-dimensional</td>
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
              <div className="finding-icon">📊</div>
              <h3>Model Complexity ≠ Performance</h3>
              <p>
                Single-task RoBERTa outperformed multi-task variants, suggesting 
                that architectural complexity doesn't always improve results.
              </p>
            </div>
            <div className="finding-card">
              <div className="finding-icon">⚖️</div>
              <h3>Class Imbalance Matters</h3>
              <p>
                Severe class imbalance (9% unsafe) hurt multi-task model 
                convergence and generalization.
              </p>
            </div>
            <div className="finding-card">
              <div className="finding-icon">🔍</div>
              <h3>Explainability Insights</h3>
              <p>
                LIME, SHAP, and Integrated Gradients provide complementary 
                views of model decision-making.
              </p>
            </div>
          </div>
        </section>

        {/* Downloads */}
        <section className="about-section">
          <h2>Downloads</h2>
          <div className="downloads-grid">
            <a href="/paper.pdf" className="download-card" download>
              <div className="download-icon">📄</div>
              <div className="download-info">
                <h3>Research Paper</h3>
                <p>Full technical report (PDF)</p>
              </div>
              <span className="download-arrow">→</span>
            </a>
            <a href="/poster.pdf" className="download-card" download>
              <div className="download-icon">📊</div>
              <div className="download-info">
                <h3>Research Poster</h3>
                <p>Conference-style poster (PDF)</p>
              </div>
              <span className="download-arrow">→</span>
            </a>
          </div>
        </section>

        {/* Tech Stack */}
        <section className="about-section">
          <h2>Technology Stack</h2>
          <div className="tech-stack">
            <div className="tech-category">
              <h3>Machine Learning</h3>
              <div className="tech-tags">
                <span className="tech-tag">PyTorch</span>
                <span className="tech-tag">Transformers</span>
                <span className="tech-tag">scikit-learn</span>
                <span className="tech-tag">RoBERTa</span>
              </div>
            </div>
            <div className="tech-category">
              <h3>Explainability</h3>
              <div className="tech-tags">
                <span className="tech-tag">LIME</span>
                <span className="tech-tag">SHAP</span>
                <span className="tech-tag">Captum (IG)</span>
              </div>
            </div>
            <div className="tech-category">
              <h3>Web Development</h3>
              <div className="tech-tags">
                <span className="tech-tag">React</span>
                <span className="tech-tag">FastAPI</span>
                <span className="tech-tag">Vite</span>
              </div>
            </div>
          </div>
        </section>

        {/* Team */}
        <section className="about-section">
          <h2>Team & Credits</h2>
          <div className="team-info">
            <p>
              <strong>Project Lead:</strong> Andrew [Last Name]<br/>
              <strong>Institution:</strong> Amherst College<br/>
              <strong>Course:</strong> [Course Name & Number]<br/>
              <strong>Semester:</strong> Fall 2024<br/>
              <strong>Teammates:</strong> Michael [Last Name], Tyler [Last Name]
            </p>
            <div className="links">
              <a href="https://github.com/yourusername/safetylens" className="link-button">
                <span>⚙</span> View on GitHub
              </a>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}

export default About