import React, { useState } from 'react';
import axios from 'axios';
import './index.css';

function App() {
  const [file, setFile] = useState(null);
  const [fileUrl, setFileUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [isVideo, setIsVideo] = useState(false);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setFileUrl(URL.createObjectURL(selectedFile)); // Preview the selected file
      setIsVideo(selectedFile.type.startsWith("video")); // Check if the file is a video
    }
  };

  const handleFileUpload = async () => {
    if (!file) {
      alert("Please upload a file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post('http://127.0.0.1:8000/detection', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setResult(response.data); // Set the result from the backend
    } catch (error) {
      console.error("Error uploading file:", error);
      alert("An error occurred while processing the file.");
    }
  };

  const drawBoundingBoxes = (ctx, boxes, confidences, classes, originalWidth, originalHeight, canvasWidth, canvasHeight) => {
    boxes.forEach((box, index) => {
      const [x, y, width, height] = box;
      const confidence = confidences[index];
      const className = classes[index];
  
      // Scale the bounding boxes to fit the resized canvas
      const scaledX = x * (canvasWidth / originalWidth);
      const scaledY = y * (canvasHeight / originalHeight);
      const scaledWidth = width * (canvasWidth / originalWidth);
      const scaledHeight = height * (canvasHeight / originalHeight);
  
      // Draw the bounding box
      ctx.beginPath();
      ctx.rect(scaledX, scaledY, scaledWidth, scaledHeight);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "red";
      ctx.fillStyle = "red";
      ctx.stroke();
  
      // Draw the label and confidence
      ctx.font = "16px Arial";
      ctx.fillStyle = "red";
      ctx.fillText(`${className} (${Math.round(confidence)}%)`, scaledX, scaledY > 10 ? scaledY - 5 : 10);
    });
  };

  const renderDetectionResults = () => {
    if (!result || !fileUrl) return;
  
    const desiredWidth = 640; // Example of resizing the video to a smaller width
    const isVideoElement = isVideo; // Assuming `isVideo` is a flag to check if it's a video
  
    if (isVideoElement) {
      // Check if the video element already exists
      let videoElement = document.getElementById('detectionVideo');
      
      if (!videoElement) {
        // Create the video element only if it doesn't already exist
        videoElement = document.createElement('video');
        videoElement.id = 'detectionVideo';  // Ensure a unique ID
        videoElement.src = fileUrl;
        videoElement.controls = true;
        videoElement.autoplay = true;
        videoElement.loop = false;  // Set loop to false if you want the video to play only once
        videoElement.style.width = `${desiredWidth}px`;  // Set desired width for the video
        videoElement.style.height = 'auto';  // Maintain aspect ratio
  
        // Append the video to the container (only once)
        document.getElementById('videoContainer').appendChild(videoElement);
      }
  
      // When the video is loaded, start drawing
      videoElement.onloadeddata = () => {
        const canvas = document.getElementById('detectionCanvas');
        const ctx = canvas.getContext('2d');
  
        // Set canvas dimensions to match the resized video
        const aspectRatio = videoElement.videoWidth / videoElement.videoHeight;
        const videoHeight = desiredWidth / aspectRatio;  // Calculate height based on width and aspect ratio
  
        canvas.width = desiredWidth;
        canvas.height = videoHeight;
  
        // Set up video play behavior
        videoElement.play();
  
        // Process each frame and update bounding boxes
        videoElement.ontimeupdate = () => {
          // Clear previous frame (if any)
          ctx.clearRect(0, 0, canvas.width, canvas.height); 
          ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height); // Draw the resized video
  
          // Calculate the index for the current frame
          const totalFrames = result.frames_results.length;
          const currentTime = videoElement.currentTime;
          const currentFrameIndex = Math.floor((currentTime / videoElement.duration) * totalFrames);
  
          // Check if the current frame has detection results
          const frameResults = result.frames_results[currentFrameIndex];
  
          if (frameResults) {
            const { boxes, confidences, classes } = frameResults;
  
            // Draw bounding boxes and scale them to the resized video
            drawBoundingBoxes(ctx, boxes, confidences, classes, videoElement.videoWidth, videoElement.videoHeight, canvas.width, canvas.height);
          }
        };
      };
    } else {
      // Handle image detection results (same as before)
      const { boxes, confidences, classes } = result.detection_results;
  
      const img = new Image();
      img.src = fileUrl;
      img.onload = () => {
        const canvas = document.getElementById('detectionCanvas');
        const ctx = canvas.getContext('2d');
  
        // Resize the canvas if necessary (for the image size)
        const aspectRatio = img.width / img.height;
        const imgHeight = desiredWidth / aspectRatio;  // Calculate height based on width and aspect ratio
  
        canvas.width = desiredWidth;
        canvas.height = imgHeight;
  
        // Draw the image on the canvas
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  
        // Draw bounding boxes
        drawBoundingBoxes(ctx, boxes, confidences, classes, img.width, img.height, canvas.width, canvas.height);
      };
    }
  };
  
  // Modify the drawBoundingBoxes function to scale bounding boxes correctly
  
  
  
  

  return (
    <div className="App">
      <h1>Object Detection using FastAPI and React</h1>

      {/* File Upload */}
      <input type="file" accept="image/*,video/*" onChange={handleFileChange} />
      <button onClick={handleFileUpload}>Upload File for Detection</button>

      {fileUrl && (
        <div>
          <h2>Uploaded File</h2>
          {isVideo ? (
            <div id="videoContainer">
              {/* Video element will be dynamically inserted here */}
            </div>
          ) : (
            <img src={fileUrl} alt="Uploaded" style={{ maxWidth: '100%', height: 'auto' }} />
          )}
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
