import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setResult(null);
  };

  const handleSubmit = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);

    try {
      setLoading(true);
      const res = await axios.post('https://diplom-2-tfho.onrender.com/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(res.data);
    } catch (err) {
      console.error(err);
      setResult({ error: 'Prediction failed' });
    } finally {
      setLoading(false);
    }
  };

return (
  <div className="App">
    <h2>Upload an Image</h2>

    {previewUrl && (
      <div className="image-preview-container">
        <img src={previewUrl} alt="Preview" />
      </div>
    )}

    <div className="upload-section">
      <input type="file" onChange={handleFileChange} />
    </div>

    <div>
      <button onClick={handleSubmit} disabled={!file || loading}>
        {loading ? 'Predicting...' : 'Predict'}
      </button>
    </div>

    {result && result.label && (
      <div className="result-box">
        <h3>Prediction Result</h3>
        <p><strong>Label:</strong> {result.label}</p>
        <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%</p>
      </div>
    )}

  </div>
);

}

export default App;
