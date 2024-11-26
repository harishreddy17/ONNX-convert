const MODEL_PATH = "best1.onnx?v=" + new Date().getTime(); // // Replace with the actual path to your ONNX model
const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 640;

const videoUpload = document.getElementById("video-upload");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// Set canvas dimensions
canvas.width = VIDEO_WIDTH;
canvas.height = VIDEO_HEIGHT;

const YOLO_CLASSES = [];

// Load the ONNX model
let session;
(async function loadModel() {
  session = await ort.InferenceSession.create(MODEL_PATH);
})();

// Handle video upload
videoUpload.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (file) {
    video.src = URL.createObjectURL(file);
    video.onloadedmetadata = () => {
      video.play();
      processVideoFrame();
    };
  }
});

// Process each frame
async function processVideoFrame() {
  if (video.paused || video.ended) return;

  ctx.drawImage(video, 0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);
  const imageData = ctx.getImageData(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);
  const inputTensor = preprocessImage(imageData);

  // Run model inference
  const results = await session.run({ images: inputTensor });
  const detections = processResults(results);

  drawDetections(detections);

  requestAnimationFrame(processVideoFrame);
}

// Preprocess the image for YOLOv8 input
function preprocessImage(imageData) {
  const { data, width, height } = imageData;
  const input = new Float32Array(VIDEO_WIDTH * VIDEO_HEIGHT * 3);

  const scaleX = VIDEO_WIDTH / width;
  const scaleY = VIDEO_HEIGHT / height;

  // Resize to VIDEO_WIDTH x VIDEO_HEIGHT
  let index = 0;
  for (let i = 0; i < VIDEO_HEIGHT; i++) {
    for (let j = 0; j < VIDEO_WIDTH; j++) { 
      const x = Math.min(width - 1, Math.floor(j / scaleX));
      const y = Math.min(height - 1, Math.floor(i / scaleY));
      const pixelIndex = (i * width + j) * 4;
      input[index++] = data[pixelIndex] / 255.0;
      input[index++] = data[pixelIndex + 1] / 255.0;
      input[index++] = data[pixelIndex + 2] / 255.0;
    }
  }

  return new ort.Tensor("float32", input, [1, 3, VIDEO_HEIGHT, VIDEO_WIDTH]);
}

// Adjusting processResults function
function processResults(results, originalWidth, originalHeight) {
  const outputData = results.output0.data;
  const detections = [];
  const NUM_CLASSES = 8;

  const numPredictions = outputData.length / (NUM_CLASSES + 5);

  for (let i = 0; i < numPredictions; i++) {
    const startIdx = i * (NUM_CLASSES + 5);

    const x = outputData[startIdx];
    const y = outputData[startIdx + 1];
    const w = outputData[startIdx + 2];
    const h = outputData[startIdx + 3];
    const confidence = outputData[startIdx + 4];

    if (confidence > 0.3) {
      // Adjust confidence threshold
      const classProbabilities = outputData.slice(
        startIdx + 5,
        startIdx + 5 + NUM_CLASSES
      );
      const classID = classProbabilities.indexOf(
      Math.max(...classProbabilities)
      );

      if (classID >= 0 && classID < NUM_CLASSES) {
        const xMin = Math.max(0, (x - w / 2) * VIDEO_WIDTH);
        const yMin = Math.max(0, (y - h / 2) * VIDEO_HEIGHT);
        const boxWidth = Math.min(w * VIDEO_WIDTH, VIDEO_WIDTH - xMin);
        const boxHeight = Math.min(h * VIDEO_HEIGHT, VIDEO_HEIGHT - yMin);

        const finalConfidence = Math.min(
          1,
          confidence * classProbabilities[classID]
        );
        detections.push({
          xMin,
          yMin,
          boxWidth,
          boxHeight,
          confidence: finalConfidence,
          classID,
        });
      }
    }
  }

  return detections;
}

// Draw bounding boxes
function drawDetections(detections) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const scaleX = VIDEO_WIDTH / originalWidth;
  const scaleY = VIDEO_HEIGHT / originalHeight;

  detections.forEach((detection) => {
    const { xMin, yMin, boxWidth, boxHeight, confidence, classID } = detection;

    const classNames = ['Bonnet', 'Bumper', 'Dickey', 'Door', 'Fender', 'Light', 'Windshield'];
    const classLabel = classNames[classID] || "Unknown";

    ctx.strokeStyle = "red";
    ctx.lineWidth = 1;
    ctx.strokeRect(xMin * scaleX, yMin * scaleY, boxWidth * scaleX, boxHeight * scaleY);
    ctx.fillStyle = "red";
    ctx.font = "12px Arial";
    ctx.fillText(
      `${classLabel}: ${(confidence * 100).toFixed(2)}%`,
      xMin * scaleX,
      yMin * scaleY- 10
    );
  });
}
