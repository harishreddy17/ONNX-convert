import React, { useState } from 'react';
import axios from 'axios';
import './index.css';

function App() {
  const [image, setImage] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [result, setResult] = useState(null);

  const handleImageChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setImage(file);
      setImageUrl(URL.createObjectURL(file)); // Preview the selected image
    }
  };

  const handleImageUpload = async () => {
    if (!image) {
      alert("Please upload an image.");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await axios.post('http://127.0.0.1:8000/detection', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setResult(response.data); // Set the result from the backend
    } catch (error) {
      console.error("Error uploading image:", error);
      alert("An error occurred while processing the image.");
    }
  };

  const drawBoundingBoxes = (ctx, boxes, confidences, classes) => {
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
      ctx.fillText(`${className} (${Math.round(confidence)}%)`, x, y > 10 ? y - 5 : 10);
    });
  };

  const renderDetectionResults = () => {
    if (!result || !imageUrl) return;

    const { boxes, confidences, classes } = result;

    // Create an image to overlay the bounding boxes
    const img = new Image();
    img.src = imageUrl;
    img.onload = () => {
      const canvas = document.getElementById('detectionCanvas');
      const ctx = canvas.getContext('2d');
      
      // Set canvas dimensions to match the image
      canvas.width = img.width;
      canvas.height = img.height;

      // Draw the image on the canvas
      ctx.drawImage(img, 0, 0);

      // Draw bounding boxes
      drawBoundingBoxes(ctx, boxes, confidences, classes);
    };
  };

  return (
    <div className="App">
      <h1>Object Detection using FastAPI and React</h1>

      {/* Image Upload */}
      <input type="file" accept="image/*" onChange={handleImageChange} />
      <button onClick={handleImageUpload}>Upload Image for Detection</button>

      {imageUrl && (
        <div>
          <h2>Uploaded Image</h2>
          <img src={imageUrl} alt="Uploaded" style={{ maxWidth: '100%', height: 'auto' }} />
        </div>
      )}

      {result && (
        <div>
          <h2>Detection Results</h2>
          <canvas id="detectionCanvas" style={{ border: '1px solid #000' }}></canvas>
        </div>
      )}

      {/* Trigger rendering when results are available */}
      {result && renderDetectionResults()}
    </div>
  );
}

export default App;
