import React, { useState, useRef } from 'react';
import axios from 'axios';
import './index.css';

function App() {
  const [video, setVideo] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const videoRef = useRef(null); // To access the video element
  const canvasRef = useRef(null); // To access the canvas element

  const handleVideoChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setVideo(file);
      setVideoUrl(URL.createObjectURL(file)); // Preview the selected video
    }
  };

  const handleVideoUpload = async () => {
    if (!video) {
      alert("Please upload a video.");
      return;
    }

    const formData = new FormData();
    formData.append("file", video);

    try {
      setLoading(true);  // Start loading
      const response = await axios.post('http://127.0.0.1:8000/detection', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      // Log the response data to check its structure
      console.log("Backend Response:", response.data);

      setResult(response.data); // Set the result from the backend
    } catch (error) {
      console.error("Error uploading video:", error);
      alert("An error occurred while processing the video.");
    } finally {
      setLoading(false);  // End loading
    }
  };

  const drawBoundingBoxes = (ctx, boxes, confidences, classes) => {
    // Ensure boxes, confidences, and classes are valid arrays before drawing
    if (!Array.isArray(boxes) || !Array.isArray(confidences) || !Array.isArray(classes)) return;

    // Log values to debug drawing logic
    console.log("Boxes:", boxes);
    console.log("Confidences:", confidences);
    console.log("Classes:", classes);

    boxes.forEach((box, index) => {
      const [x, y, width, height] = box;
      const confidence = confidences[index];
      const className = classes[index];

      // Draw the bounding box
      ctx.beginPath();
      ctx.rect(x, y, width, height);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "red";
      ctx.fillStyle = "red";
      ctx.stroke();

      // Draw the label and confidence
      ctx.font = "16px Arial";
      ctx.fillStyle = "red";
      ctx.fillText(`${className} (${Math.round(confidence * 100)}%)`, x, y > 10 ? y - 5 : 10);
    });
  };

  const processVideoFrame = () => {
    if (!videoRef.current || !canvasRef.current || !result) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Set the canvas dimensions to match the video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw the video frame onto the canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Log the result before drawing bounding boxes
    console.log("Detection Result:", result);

    const { boxes = [], confidences = [], classes = [] } = result.detection_results || {};

    // Log detection arrays to debug
    console.log("Boxes:", boxes);
    console.log("Confidences:", confidences);
    console.log("Classes:", classes);

    // Draw bounding boxes if arrays are not empty
    if (boxes.length && confidences.length && classes.length) {
      drawBoundingBoxes(ctx, boxes, confidences, classes);
    } else {
      console.log("No bounding boxes detected.");
    }
  };

  const handlePlay = () => {
    if (videoRef.current) {
      videoRef.current.play();
      setInterval(processVideoFrame, 100); // Process every 100ms (10 FPS), adjust as needed
    }
  };

  return (
    <div className="App">
      <h1>Object Detection using FastAPI and React - Video Upload</h1>

      {/* Video Upload */}
      <input type="file" accept="video/*" onChange={handleVideoChange} />
      <button onClick={handleVideoUpload} disabled={loading}>
        {loading ? "Uploading..." : "Upload Video for Detection"}
      </button>

      {videoUrl && (
        <div>
          <h2>Uploaded Video</h2>
          <video
            ref={videoRef}
            src={videoUrl}
            controls
            width="100%"
            style={{ marginBottom: '20px' }}
          ></video>
          <button onClick={handlePlay} disabled={loading}>
            Play Video with Detection
          </button>
        </div>
      )}

      {result && (
        <div>
          <h2>Detection Results</h2>
          <canvas
            ref={canvasRef}
            style={{ border: '1px solid #000', maxWidth: '100%' }}
          ></canvas>
        </div>
      )}
    </div>
  );
}

export default App;
