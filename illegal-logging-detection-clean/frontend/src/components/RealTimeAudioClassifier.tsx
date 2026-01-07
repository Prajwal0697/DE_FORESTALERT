import React, { useState, useRef } from 'react';
import axios, { AxiosError } from 'axios';

interface ClassificationResult {
  classification: string;
  confidence: number;
  confidence_scores: Record<string, number>;
}

const RealTimeAudioClassifier: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [classification, setClassification] = useState<ClassificationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      
      // Validate file type
      const allowedTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/x-wav'];
      if (!allowedTypes.includes(file.type)) {
        setError('Please select a valid audio file (WAV or MP3)');
        return;
      }

      // Validate file size (e.g., max 10MB)
      const maxSize = 10 * 1024 * 1024; // 10MB
      if (file.size > maxSize) {
        setError('File is too large. Maximum size is 10MB');
        return;
      }

      setSelectedFile(file);
      setClassification(null);
      setError(null);
    }
  };

  const classifyAudio = async () => {
    if (!selectedFile) {
      setError('Please select an audio file first');
      return;
    }

    setIsLoading(true);
    setError(null);
    setClassification(null);

    try {
      // Create FormData to send the audio file
      const formData = new FormData();
      formData.append('audio', selectedFile, selectedFile.name);

      // Send to backend for classification
      const response = await axios.post<ClassificationResult>('http://localhost:8000/api/classify-audio', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Update classification result
      console.log(response.data);
      setClassification(response.data);
    } catch (err) {
      // More detailed error handling
      if (axios.isAxiosError(err)) {
        const axiosError = err as AxiosError;
        if (axiosError.response) {
          const errorMessage = axiosError.response.data instanceof Object 
            ? JSON.stringify(axiosError.response.data) 
            : axiosError.response.data;
          
          console.error('Classification error response:', errorMessage);
          setError(`Classification failed: ${errorMessage}`);
        } else if (axiosError.request) {
          console.error('No response received:', axiosError.request);
          setError('No response from server. Please check your connection.');
        } else {
          console.error('Error setting up request:', axiosError.message);
          setError(`Request error: ${axiosError.message}`);
        }
      } else {
        console.error('Unexpected classification error:', err);
        setError('An unexpected error occurred during classification');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    setClassification(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="sound-classifier-container">
      <div className="sound-classifier-card">
        <h2><span className="forest-logo-emoji">🌲</span> Forest Sound Classifier</h2>
        <p className="desc-line">Upload audio from the forest to check for illegal sound activity</p>
        <div className="file-upload-section">
          <input 
            type="file" 
            id="audio-file-input"
            accept="audio/wav,audio/mpeg,audio/mp3,audio/x-wav" 
            onChange={handleFileChange}
            ref={fileInputRef}
            style={{ display: 'none' }}
          />
          <label htmlFor="audio-file-input" className="file-upload-label nature-upload-label">
            {selectedFile ? 
              <div className="selected-file nature-selected-file">
                <span><i className="icon-upload"></i> {selectedFile.name}</span>
                <button type="button" onClick={clearFile} className="clear-file-btn">
                  ✕
                </button>
              </div> : 
              <div className="upload-placeholder">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21.2 15c.7-1.2 1-2.5.7-3.9-.6-2-2.4-3.6-4.4-3.9C15.1 5.6 13.2 4 11 4c-2.5 0-4.6 1.7-5.1 4-1.7.3-3 1.7-3.3 3.4C2.4 13 4 15 6 15h1.5v-2H6c-.8 0-1.3-.7-1.2-1.5.1-.5.5-1 1-1.2.4-.2.7-.6.8-1.1.2-1.1 1.1-2 2.2-2.2.7-.1 1.4.1 1.9.5.4.4.6.9.6 1.5v3h2v-3c0-.6.2-1.1.6-1.5.5-.4 1.2-.6 1.9-.5 1.1.2 2 1.1 2.2 2.2.1.5.4.9.8 1.1.5.3.9.7 1 1.2.1.8-.4 1.5-1.2 1.5h-1.5v2H18c2 0 3.7-2 3.2-4z"/>
                  <path d="M12 16v-7"/>
                  <path d="m15 12-3-3-3 3"/>
                </svg>
                <p>Drag & drop or click to upload audio</p>
                <small>WAV or MP3 (max 10MB)</small>
              </div>
            }
          </label>
          {selectedFile && (
            <button 
              onClick={classifyAudio} 
              disabled={isLoading}
              className="classify-btn nature-classify-btn"
            >
              <i className="icon-mic"></i> {isLoading ? 'Classifying...' : 'Classify Audio'}
            </button>
          )}
        </div>
        {error && (
          <div className="error-message nature-error-message">
            <p>{error}</p>
          </div>
        )}
        {classification && (
          <div className="classification-result">
            <h3>Classification Result <span role="img" aria-label="result">🔎</span></h3>
            <div className="result-details">
              <p>
                <strong>Classification:</strong>
                <span 
                  className={`classification-label ${classification.classification.toLowerCase()}`}
                  style={{marginLeft:'12px'}}
                >{classification.classification==='illegal'? '🪓 ILLEGAL':'🌳 NATURAL'}</span>
              </p>
              <p><strong>Confidence:</strong> {(classification.confidence * 100).toFixed(2)}%</p>
              <div className="confidence-breakdown">
                <h4>Confidence Scores:</h4>
                {Object.entries(classification.confidence_scores).map(([className, score]) => (
                  <div key={className} className="confidence-bar">
                    <span>{className==='illegal'?'🪓':'🌳'} {className}</span>
                    <div className="bar">
                      <div 
                        className="bar-fill" 
                        style={{ 
                          width: `${score * 100}%`, 
                          backgroundColor: score > 0.5 ? '#4CAF50' : '#FF5722' 
                        }}
                      ></div>
                    </div>
                    <span>{(score * 100).toFixed(2)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default RealTimeAudioClassifier;

