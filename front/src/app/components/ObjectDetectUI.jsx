"use client";
import React, { useState, useRef, useEffect } from 'react';

export default function ObjectDetectUI() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState([]);
  const [loading, setLoading] = useState(false);
  const canvasRef = useRef();
  const videoRef = useRef();

  const captureAndSend = async () => {
    if (!videoRef.current) return;

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    ctx.drawImage(videoRef.current, 0, 0);

    canvas.toBlob(async (blob) => {
      if (!blob) return;

      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      setLoading(true);
      try {
        const res = await fetch("http://127.0.0.1:5000/detect", {
          method: "POST",
          body: formData,
        });
        const data = await res.json();
        setResult(data.detections);
      } catch (err) {
        console.error(err);
      }
      setLoading(false);
    }, "image/jpeg");
  };

  useEffect(() => {
    if (!videoRef.current) return;
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
      videoRef.current.srcObject = stream;
    });
  }, []);

  useEffect(() => {
    if (!videoRef.current || result.length === 0) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const video = videoRef.current;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    result.forEach((r) => {
      const [x1, y1, x2, y2] = r.box;
      const color = "limegreen";

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      ctx.fillStyle = color;
      ctx.font = "16px Arial";
      ctx.fillText(`${r.label} (${Math.round(r.confidence * 100)}%)`, x1 + 4, y1 + 18);
    });
  }, [result]);

  return (
    <div style={{ maxWidth: '700px', margin: '0 auto', padding: '2rem' }}>
      <h2 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '1rem' }}>
        📷 Live Camera Waste Detection
      </h2>

      <video ref={videoRef} autoPlay style={{ width: '100%', borderRadius: '8px' }} />
      <canvas ref={canvasRef} style={{ width: '100%', marginTop: '1rem', borderRadius: '8px' }} />

      <button
        onClick={captureAndSend}
        disabled={loading}
        style={{
          backgroundColor: '#4CAF50',
          color: 'white',
          padding: '10px 20px',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer',
          marginTop: '1rem',
        }}
      >
        {loading ? 'Detecting...' : 'Capture & Detect'}
      </button>
    </div>
  );
}