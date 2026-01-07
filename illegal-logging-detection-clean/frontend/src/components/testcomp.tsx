import React, { useState, useEffect } from "react";
import MicRecorder from "mic-recorder-to-mp3";

const Mp3Recorder = new MicRecorder({ bitRate: 128 });

interface ClassificationResult {
  classification: string;
  confidence: number;
  confidence_scores: Record<string, number>;
}

const AudioRecorder: React.FC = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [classification, setClassification] = useState<ClassificationResult | null>(null);
  const [blobURL, setBlobURL] = useState<string | null>(null);
  const [myBlob, setMyBlob] = useState<Blob | null>(null);
  const [isBlocked, setIsBlocked] = useState(false);

  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then(() => setIsBlocked(false))
      .catch(() => {
        setIsBlocked(true);
        alert("Microphone access denied. Please allow it in your browser.");
      });
  }, []);

  const startRecording = () => {
    if (isBlocked) {
      alert("Microphone permission is blocked.");
    } else {
      Mp3Recorder.start()
        .then(() => setIsRecording(true))
        .catch((e:unknown) => console.error("Error starting recording:", e));
    }
  };

  const stopRecording = () => {
    Mp3Recorder.stop()
      .getMp3()
      .then(([buffer, blob]: [ArrayBuffer[], Blob]) => {
        setMyBlob(blob);
        const blobUrl = URL.createObjectURL(blob);
        setBlobURL(blobUrl);
        setIsRecording(false);
      })
      .catch((e:unknown) => console.error("Error stopping recording:", e));
  };

  const submitToBK = async () => {
    if (!myBlob) {
      alert("No recording found.");
      return;
    }

    const formData = new FormData();
    formData.append("audio", myBlob, "recording.mp3");

    try {
      const res = await fetch("http://localhost:8000/api/classify-audio", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Failed to classify audio");

      const data: ClassificationResult = await res.json();
      setClassification(data);
    } catch (error) {
      console.error(error);
      alert("Error contacting backend");
    }
  };

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <h1 style={styles.title}>🎤 Audio Classifier</h1>

        {isRecording && (
          <div style={styles.recordingIndicator}>
            <div style={styles.recordDot}></div>
            <span style={styles.recordText}>Recording...</span>
          </div>
        )}

        <div style={styles.buttonGroup}>
          <button
            onClick={startRecording}
            disabled={isRecording}
            style={{
              ...styles.button,
              background: isRecording ? "#666" : "#34a853",
              cursor: isRecording ? "not-allowed" : "pointer",
            }}
          >
            🎙 Start
          </button>
          <button
            onClick={stopRecording}
            disabled={!isRecording}
            style={{
              ...styles.button,
              background: !isRecording ? "#666" : "#ea4335",
              cursor: !isRecording ? "not-allowed" : "pointer",
            }}
          >
            ⏹ Stop
          </button>
        </div>

        {blobURL && (
          <div style={styles.audioSection}>
            <audio controls src={blobURL} style={styles.audio}></audio>
            <button onClick={submitToBK} style={styles.analyzeBtn}>
              🔍 Analyze Audio
            </button>
          </div>
        )}

        {classification && (
          <div style={styles.resultBox}>
            <h3 style={styles.resultTitle}>
              Result <span role="img" aria-label="search">🔎</span>
            </h3>

            <p style={styles.resultText}>
              <strong>Classification:</strong>{" "}
              <span
                style={{
                  ...styles.label,
                  background:
                    classification.classification === "illegal"
                      ? "#d93025"
                      : "#34a853",
                }}
              >
                {classification.classification === "illegal"
                  ? "🪓 ILLEGAL"
                  : "🌳 NATURAL"}
              </span>
            </p>

            <p style={styles.resultText}>
              <strong>Confidence:</strong>{" "}
              {(classification.confidence * 100).toFixed(2)}%
            </p>

            <div style={{ marginTop: "12px" }}>
              <h4 style={{ marginBottom: "8px", color: "#ddd" }}>Confidence Breakdown:</h4>
              {Object.entries(classification.confidence_scores).map(
                ([className, score]) => (
                  <div key={className} style={{ marginBottom: "10px" }}>
                    <div style={styles.barLabel}>
                      <span>
                        {className === "illegal" ? "🪓" : "🌳"} {className}
                      </span>
                      <span>{(score * 100).toFixed(2)}%</span>
                    </div>
                    <div style={styles.barContainer}>
                      <div
                        style={{
                          ...styles.barFill,
                          width: `${score * 100}%`,
                          backgroundColor: score > 0.5 ? "#34a853" : "#d93025",
                        }}
                      ></div>
                    </div>
                  </div>
                )
              )}
            </div>
          </div>
        )}

        <p style={styles.footer}>Built with ❤️ for forest sound detection</p>
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  page: {
    minHeight: "100vh",
    background:
      "linear-gradient(135deg, #0f2027, #203a43, #2c5364)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontFamily: "'Poppins', sans-serif",
  },
  card: {
    background: "rgba(255, 255, 255, 0.1)",
    borderRadius: "20px",
    padding: "30px",
    width: "380px",
    color: "white",
    boxShadow: "0 8px 25px rgba(0, 0, 0, 0.3)",
    backdropFilter: "blur(15px)",
    textAlign: "center",
  },
  title: {
    fontSize: "26px",
    fontWeight: 600,
    marginBottom: "20px",
    letterSpacing: "0.5px",
  },
  recordingIndicator: {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    gap: "8px",
    marginBottom: "15px",
  },
  recordDot: {
    width: "10px",
    height: "10px",
    backgroundColor: "#ff4b4b",
    borderRadius: "50%",
    animation: "pulse 1s infinite",
  },
  recordText: {
    color: "#ff4b4b",
    fontWeight: 500,
  },
  buttonGroup: {
    display: "flex",
    justifyContent: "center",
    gap: "10px",
    marginBottom: "20px",
  },
  button: {
    border: "none",
    padding: "10px 18px",
    color: "white",
    fontWeight: 500,
    borderRadius: "25px",
    transition: "all 0.3s ease",
  },
  audioSection: {
    marginTop: "10px",
  },
  audio: {
    width: "100%",
    borderRadius: "10px",
    marginBottom: "10px",
  },
  analyzeBtn: {
    width: "100%",
    background: "linear-gradient(90deg, #4285f4, #6c63ff)",
    border: "none",
    color: "white",
    fontWeight: 500,
    padding: "10px",
    borderRadius: "25px",
    cursor: "pointer",
    transition: "transform 0.2s ease",
  },
  resultBox: {
    background: "rgba(255, 255, 255, 0.1)",
    padding: "15px",
    borderRadius: "15px",
    marginTop: "20px",
    textAlign: "left",
  },
  resultTitle: {
    fontSize: "18px",
    fontWeight: 600,
    marginBottom: "10px",
  },
  resultText: {
    fontSize: "14px",
    marginBottom: "6px",
  },
  label: {
    padding: "3px 8px",
    borderRadius: "8px",
    fontWeight: 600,
    color: "white",
    fontSize: "13px",
  },
  barLabel: {
    display: "flex",
    justifyContent: "space-between",
    fontSize: "13px",
    marginBottom: "4px",
  },
  barContainer: {
    width: "100%",
    height: "6px",
    backgroundColor: "rgba(255,255,255,0.2)",
    borderRadius: "10px",
    overflow: "hidden",
  },
  barFill: {
    height: "100%",
    borderRadius: "10px",
    transition: "width 0.5s ease",
  },
  footer: {
    fontSize: "11px",
    color: "rgba(255,255,255,0.6)",
    marginTop: "20px",
  },
};

export default AudioRecorder;
