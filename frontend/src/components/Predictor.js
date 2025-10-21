// src/components/Predictor.js
import React, { useState, useEffect } from 'react';

const BASE_API_URL = process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:8000';

function Predictor() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(''); // No default model
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchModels() {
      try {
        const response = await fetch(`${BASE_API_URL}/models`);
        if (!response.ok) throw new Error('Failed to fetch models list.');
        
        const data = await response.json();
        setModels(data.models || []);
      } catch (err) {
        setError('Could not load models. Is the backend server running?');
        console.error("Error fetching models:", err);
      }
    }
    fetchModels();
  }, []);

  const getWeatherDisplay = (prediction) => {
    // (This function is unchanged)
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
    if (!selectedFile) {
      setError('Please select an audio file.'); return;
    }
    // Check if a model is selected
    if (!selectedModel) {
      setError('Please select a model.'); return;
    }
    
    setIsLoading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('model_name', selectedModel); // Always send model_name
    
    // We only use the 'single' endpoint now, as per your sketch
    const apiEndpoint = `${BASE_API_URL}/predict/single`; 

    try {
      const response = await fetch(apiEndpoint, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Prediction failed.');
      
      const display = getWeatherDisplay(data.weather_prediction);
      setResult({ ...display, modelUsed: data.model_used });

    } catch (err) {
      setError(err.message || 'An unknown error occurred.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <section id="predictor" className="section">
      <form className="upload-form" onSubmit={handleSubmit}>
        
        {/* --- NEW LAYOUT BASED ON SKETCH --- */}
        <h2>Select The model</h2>
        <div className="model-selection-grid" role="radiogroup">
          {models.map(modelName => (
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
          ))}
        </div>

        <h3 className="file-select-title">Select The File</h3>
        <input 
          type="file" 
          accept="audio/*"
          onChange={(e) => setSelectedFile(e.target.files[0])}
          required 
        />

        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Analyzing...' : 'Predict Weather'}
        </button>
        {/* --- END OF NEW LAYOUT --- */}

      </form>

      {/* --- Result Display Area (Unchanged) --- */}
      {isLoading && (
        <div className="result-container visible loading">
          <div className="spinner"></div>
          <p>Uploading and analyzing... Please wait.</p>
        </div>
      )}
      {error && (
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