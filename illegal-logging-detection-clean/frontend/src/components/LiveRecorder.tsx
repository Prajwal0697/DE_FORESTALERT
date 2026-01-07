import React, { useState, useRef, useEffect } from 'react';
import MicRecorder from 'mic-recorder-to-mp3';
import axios, { AxiosError } from 'axios';

interface ClassificationResult {
  classification: string;
  confidence: number;
  confidence_scores: Record<string, number>;
}

const LiveRecorder: React.FC = () => {
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [classification, setClassification] = useState<ClassificationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [recordingTime, setRecordingTime] = useState<number>(0);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);

  const recorderRef = useRef<{
    start: () => Promise<void>;
    stop: () => any;
  } | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const recordingDuration = 6; // Changed to 6 seconds

  // Initialize the recorder
  useEffect(() => {
    recorderRef.current = new MicRecorder({ bitRate: 128 });

    // Cleanup function
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  const startRecording = () => {
    if (recorderRef.current) {
      recorderRef.current
        .start()
        .then(() => {
          setIsRecording(true);
          setClassification(null);
          setError(null);
          setRecordingTime(0);
          setIsProcessing(false);

          // Start timer
          timerRef.current = setInterval(() => {
            setRecordingTime(prev => {
              // Automatically stop recording at 6 seconds
              if (prev + 1 >= recordingDuration) {
                stopRecording();
                return recordingDuration;
              }
              return prev + 1;
            });
          }, 1000);
        })
        .catch((e: Error) => {
          console.error('Error starting recording:', e);
          setError('Could not start recording');
        });
    } else {
      setError('Recorder not initialized');
    }
  };

  const stopRecording = () => {
    if (recorderRef.current) {
      console.log("Stopping recording...");
      try {
        // Stop timer
        if (timerRef.current) {
          clearInterval(timerRef.current);
        }

        const stopResult = recorderRef.current.stop();
        console.log("Recorder stop method called:", stopResult.getmp3());

        // Check if stopResult is a promise or has a then method
        if (stopResult && (typeof stopResult.then === 'function')) {
          stopResult
            .then(([buffer, blob]: [ArrayBuffer, Blob]) => {
              setIsRecording(false);
              setIsProcessing(true);
              classifyAudio(blob);
            })
            .catch((e: Error) => {
              console.error('Error stopping recording:', e);
              setError('Could not stop recording');
              setIsRecording(false);
              setIsProcessing(false);
            });
        } else if (Array.isArray(stopResult)) {
          // Fallback for direct array return
          const [buffer, blob] = stopResult;
          setIsRecording(false);
          setIsProcessing(true);
          classifyAudio(blob);
        } else {
          console.error('Unexpected stop method result', stopResult);
          setError('Could not stop recording');
          setIsRecording(false);
          setIsProcessing(false);
        }
      } catch (e) {
        console.error('Exception in stop recording:', e);
        setError('Could not stop recording');
        setIsRecording(false);
        setIsProcessing(false);
      }
    } else {
      setError('Recorder not initialized');
    }
  };

  const classifyAudio = async (audioFile: Blob) => {
    try {
      // Create FormData to send the audio file
      const formData = new FormData();
      formData.append('audio', audioFile, 'recording.mp3');

      // Send to backend for classification
      const response = await axios.post<ClassificationResult>('http://localhost:8000/api/classify-audio', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Update classification result
      setClassification(response.data);
      setIsProcessing(false);
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
      setIsProcessing(false);
    }
  };

  // Format time as MM:SS
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="live-recorder-container">
      <div className="recording-controls">
        <div className="recording-indicator">
          {isRecording && (
            <div className="recording-pulse">
              <div className="pulse-dot"></div>
              <span className="recording-emoji">🎤</span>
            </div>
          )}
          {!isRecording && (
            <span className="recording-emoji paused">🎙️</span>
          )}
          <span className="recording-time">
            {formatTime(recordingTime)}
          </span>
        </div>
        <div className="record-buttons nature-recorder-btns">
          {!isRecording && !isProcessing ? (
            <button 
              onClick={startRecording} 
              className="start-recording-btn"
              disabled={isRecording || isProcessing}
            >
              <i className="icon-mic"></i>
              Start Recording
            </button>
          ) : isRecording ? (
            <button 
              onClick={stopRecording} 
              className="stop-recording-btn"
              disabled={!isRecording}
            >
              <i className="icon-stop"></i>
              Stop Recording
            </button>
          ) : isProcessing ? (
            <button 
              className="processing-btn"
              disabled
            >
              Processing...
            </button>
          ) : null}
        </div>
        {error && (
          <div className="error-message nature-error-message">
            <p>{error}</p>
          </div>
        )}
        {classification && (
          <div className="classification-result">
            <h3>Classification Result <span role="img" aria-label="search">🔎</span></h3>
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

export default LiveRecorder;
