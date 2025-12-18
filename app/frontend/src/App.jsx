import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import './App.css'
import Home from './pages/Home'
import About from './pages/About'

function App() {
  return (
    <Router>
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
          
          <div className="header-bottom">
            <nav className="nav-tabs">
              <NavLink 
                to="/" 
                className={({ isActive }) => `nav-tab ${isActive ? 'active' : ''}`}
              >
                Home
              </NavLink>
              <NavLink 
                to="/about" 
                className={({ isActive }) => `nav-tab ${isActive ? 'active' : ''}`}
              >
                About
              </NavLink>
            </nav>
            
            <span className="keyboard-hint-header">
              <span className="hint-icon">⌘</span>
              Press Ctrl+Enter to predict
            </span>
          </div>
        </header>

        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
        </Routes>

        <footer>
          <p>© 2025 SafetyLens</p>
        </footer>
      </div>
    </Router>
  )
}

export default App