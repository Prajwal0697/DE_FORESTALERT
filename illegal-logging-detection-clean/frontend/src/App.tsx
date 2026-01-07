import React, { useState } from 'react';
import './App.css';
import RealTimeAudioClassifier from './components/RealTimeAudioClassifier';
import LiveRecorder from './components/LiveRecorder';
import AudioRecorder from './components/testcomp';

function App() {
  const [activeTab, setActiveTab] = useState<'upload' | 'record'>('upload');

  return (
    <div 
      className="app-container forest-theme-bg"
      style={{
        backgroundImage: `linear-gradient(120deg,#e9f5ee 0%,#e6efdb 60%,#d4ebfd 100%), url('/forest-bg.jpg')`,
        backgroundRepeat: 'no-repeat',
        backgroundSize: 'cover',
        backgroundPosition: 'center top',
        minHeight: '100vh'
      }}
    >
      <header className="main-header improved-header">
        <div className="branding-block">
          {/* Placeholder for logo. Replace with <img src={logo} alt="ForestGuard Logo" /> if you have a logo. */}
          <span className="forest-logo-emoji">🌲🎤</span>
          <h1>ForestGuard: Illegal Sound Detection</h1>
          <p className="subtitle">Protecting Nature with AI</p>
        </div>
      </header>
      <div className="main-content">
        <nav className="side-navbar improved-navbar">
          <div 
            className={`nav-item ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            <i className="icon-upload"></i>
            <span>Upload Audio</span>
          </div>
          <div 
            className={`nav-item ${activeTab === 'record' ? 'active' : ''}`}
            onClick={() => setActiveTab('record')}
          >
            <i className="icon-mic"></i>
            <span>Live Recording</span>
            {activeTab === 'record' && <span className="live-dot">● LIVE</span>}
          </div>
        </nav>
        <main className="content-area">
          {activeTab === 'upload' ? (
            <RealTimeAudioClassifier />
          ) : (
            <AudioRecorder />
          )}
        </main>
      </div>
      <footer className="main-footer">
        <p>&copy; {new Date().getFullYear()} ForestGuard – Making forests safer with AI 🤖🌳</p>
      </footer>
    </div>
  );
}

export default App;
