// src/components/Predictor.js
import React, { useState, useEffect, useRef, useCallback } from 'react';

const BASE_API_URL = process.env.REACT_APP_API_BASE_URL || 'https://avainapp.onrender.com';

// Render free tier cold starts can take up to 90s.
// We retry every 6s for up to 15 attempts = 90s coverage.
const MAX_RETRIES = 15;
const RETRY_DELAY_MS = 6000;
const FETCH_TIMEOUT_MS = 8000; // abort hung requests after 8s so retries aren't blocked

function Predictor() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [wakeUpAttempt, setWakeUpAttempt] = useState(0);
  const [countdown, setCountdown] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const retryTimerRef = useRef(null);
  const countdownTimerRef = useRef(null);

  const clearTimers = () => {
    if (retryTimerRef.current) clearTimeout(retryTimerRef.current);
    if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
  };

  const startCountdown = (seconds) => {
    setCountdown(seconds);
    if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
    countdownTimerRef.current = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) { clearInterval(countdownTimerRef.current); return 0; }
        return prev - 1;
      });
    }, 1000);
  };

  const fetchModels = useCallback(async (attempt = 1) => {
    setWakeUpAttempt(attempt);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

    try {
      console.log(`[Attempt ${attempt}/${MAX_RETRIES}] Fetching ${BASE_API_URL}/models`);
      const response = await fetch(`${BASE_API_URL}/models`, { signal: controller.signal });
      clearTimeout(timeoutId);

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`HTTP ${response.status}: ${text}`);
      }

      const data = await response.json();
      console.log('Models loaded:', data);
      clearTimers();
      setModels(data.models || []);
      setModelsLoading(false);
      setWakeUpAttempt(0);
      setCountdown(0);

    } catch (err) {
      clearTimeout(timeoutId);
      const reason = err.name === 'AbortError' ? 'Request timed out' : err.message;
      console.warn(`Attempt ${attempt} failed: ${reason}`);

      if (attempt < MAX_RETRIES) {
        // Show countdown before next attempt
        const delaySec = Math.round(RETRY_DELAY_MS / 1000);
        startCountdown(delaySec);
        retryTimerRef.current = setTimeout(() => fetchModels(attempt + 1), RETRY_DELAY_MS);
      } else {
        clearTimers();
        setError(
          `Could not reach the backend after ${MAX_RETRIES} attempts (~90 seconds). ` +
          `The server at ${BASE_API_URL} may be down. Last error: ${reason}`
        );
        setModelsLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    fetchModels(1);
    return () => clearTimers(); // cleanup on unmount
  }, [fetchModels]);

  const handleRetry = () => {
    clearTimers();
    setError(null);
    setModelsLoading(true);
    setModels([]);
    fetchModels(1);
  };

  const getWeatherDisplay = (prediction) => {
    switch (prediction) {
      case "Clear Day":
        return { icon: 'fas fa-sun', text: 'Sunny' };
      case "Impending Rain (Low Pressure)":
        return { icon: 'fas fa-cloud-showers-heavy', text: 'Rainy' };
      case "Cloudy/Overcast":
        return { icon: 'fas fa-cloud', text: 'Cloudy' };
      case "High Wind/Storm Warning":
        return { icon: 'fas fa-wind', text: 'Windy' };
      case "Unknown/Ambiguous":
        return { icon: 'fas fa-smog', text: 'Foggy' };
      default:
        return { icon: 'fas fa-question-circle', text: prediction };
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) { setError('Please select an audio file.'); return; }
    if (!selectedModel) { setError('Please select a model.'); return; }

    setIsLoading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('model_name', selectedModel);

    // 60-second timeout — audio processing on Render's free CPU can be slow
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000);

    try {
      const response = await fetch(`${BASE_API_URL}/predict/single`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `Server error (${response.status})`);

      const display = getWeatherDisplay(data.weather_prediction);
      setResult({ ...display, modelUsed: data.model_used });

    } catch (err) {
      clearTimeout(timeoutId);
      if (err.name === 'AbortError') {
        setError('The prediction timed out after 60 seconds. The server may be overloaded — please try again.');
      } else if (err.message === 'Failed to fetch') {
        setError('Cannot reach the server. Please check your internet connection or try again in a moment.');
      } else {
        setError(err.message || 'An unknown error occurred.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Build the wake-up status label
  const wakeUpLabel = () => {
    if (wakeUpAttempt <= 1) return 'Connecting to backend...';
    const secondsSpent = Math.round(((wakeUpAttempt - 1) * RETRY_DELAY_MS) / 1000);
    if (countdown > 0) {
      return `Server is starting up... retrying in ${countdown}s (${secondsSpent}s elapsed)`;
    }
    return `Waking up server... attempt ${wakeUpAttempt} of ${MAX_RETRIES}`;
  };

  // Progress bar: how far through the 90s window we are
  const progressPercent = wakeUpAttempt > 0
    ? Math.min(100, Math.round(((wakeUpAttempt - 1) / MAX_RETRIES) * 100))
    : 0;

  return (
    <section id="predictor" className="section">
      <form className="upload-form" onSubmit={handleSubmit}>

        {/* MODEL SELECTION */}
        <h2>Select The model</h2>
        <div className="model-selection-grid" role="radiogroup">
          {modelsLoading ? (
            <div className="model-wakeup-notice">
              <div className="spinner-small"></div>
              <div className="wakeup-text-block">
                <span className="wakeup-label">{wakeUpLabel()}</span>
                <div className="wakeup-progress-bar">
                  <div className="wakeup-progress-fill" style={{ width: `${progressPercent}%` }}></div>
                </div>
                <span className="wakeup-hint">
                  ☕ Render free tier sleeps after inactivity. This only happens once.
                </span>
              </div>
            </div>
          ) : error ? (
            <div className="backend-error-block">
              <p className="no-models-msg">⚠️ {error}</p>
              <button type="button" className="retry-btn" onClick={handleRetry}>
                🔄 Retry Connection
              </button>
            </div>
          ) : models.length === 0 ? (
            <p className="no-models-msg">⚠️ Backend responded but returned no models.</p>
          ) : (
            models.map(modelName => (
              <div key={modelName}>
                <input
                  type="radio" id={modelName} name="model-selection"
                  value={modelName} checked={selectedModel === modelName}
                  onChange={(e) => setSelectedModel(e.target.value)}
                />
                <label htmlFor={modelName} className="model-card">
                  <i className="fas fa-microchip model-icon"></i>
                  <span>{modelName}</span>
                </label>
              </div>
            ))
          )}
        </div>

        <h3 className="file-select-title">Select The File</h3>
        <input
          type="file"
          accept="audio/*"
          onChange={(e) => setSelectedFile(e.target.files[0])}
          required
        />

        <button type="submit" disabled={isLoading || modelsLoading || !!error}>
          {isLoading ? 'Analyzing...' : 'Predict Weather'}
        </button>

      </form>

      {isLoading && (
        <div className="result-container visible loading">
          <div className="spinner"></div>
          <p>Uploading and analyzing... Please wait.</p>
        </div>
      )}
      {!modelsLoading && error && !isLoading && (
        <div className="result-container visible error">
          <i className="result-icon fas fa-exclamation-triangle"></i>
          <p>{error}</p>
        </div>
      )}
      {result && (
        <div className="result-container visible">
          <i className={`result-icon ${result.icon}`}></i>
          <div className="result-text">
            <strong>Predicted Weather: {result.text}</strong>
            <small><em>(Prediction from: {result.modelUsed})</em></small>
          </div>
        </div>
      )}

    </section>
  );
}

export default Predictor;