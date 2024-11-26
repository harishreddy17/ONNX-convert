import io
import tempfile
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import os
from typing import List, Tuple
from numpy import ndarray
from PIL import Image

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Detection:
    def __init__(self, 
                 model_path: str, 
                 classes: List[str]):
        self.model_path = model_path
        self.classes = classes
        self.model = self.__load_model()

    def __load_model(self):
        ort_session = ort.InferenceSession(self.model_path)
        return ort_session

    def __extract_output(self, 
                         preds: ndarray, 
                         image_shape: Tuple[int, int], 
                         input_shape: Tuple[int, int],
                         score: float = 0.01,
                         nms: float = 0.3, 
                         confidence: float = 0.0) -> dict:
        class_ids, confs, boxes = [], [], []

        image_height, image_width = image_shape
        input_height, input_width = input_shape
        x_factor = image_width / input_width
        y_factor = image_height / input_height

        rows = preds[0].shape[0]
        for i in range(rows):
            row = preds[0][i]
            conf = row[4]
            classes_score = row[4:]
            _, _, _, max_idx = cv2.minMaxLoc(classes_score)
            class_id = max_idx[1]

            if classes_score[class_id] > score:
                confs.append(conf)
                label = self.classes[int(class_id)]
                class_ids.append(label)

                # Extract boxes
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = [left, top, width, height]
                boxes.append(box)

        # Perform Non-Maximum Suppression (NMS)
        indexes = cv2.dnn.NMSBoxes(boxes, confs, confidence, nms)

        # Debugging: log the shape and content of indexes
        print(f"Indexes after NMS: {indexes}")

        # Check if NMS returned valid indexes
        if indexes is None or len(indexes) == 0:
            print("No valid bounding boxes after NMS.")
            return {
                'boxes': [],
                'confidences': [],
                'classes': []
            }
        
        # If indexes is a tuple, extract the first element (it contains the indices)
        if isinstance(indexes, tuple):
            indexes = indexes[0]  # Get the first element which contains the indices

        r_class_ids, r_confs, r_boxes = [], [], []
        for i in indexes.flatten():  # Flatten and iterate over the indices
            r_class_ids.append(class_ids[i])
            r_confs.append(float(confs[i]) * 100)  # Multiply by 100 to convert to percentage
            r_boxes.append(list(boxes[i]))  # Convert to list for serialization

        # Convert to Python-native types for FastAPI serialization
        return {
            'boxes': r_boxes, 
            'confidences': r_confs, 
            'classes': r_class_ids
        }

    def __call__(self, image: ndarray, width: int = 640, height: int = 640, score: float = 0.01, nms: float = 0.3, confidence: float = 0.0) -> dict:
        image_resized = cv2.resize(image, (width, height))  # Resize to the model's input size
        image_normalized = image_resized.astype(np.float32) / 255.0  # Normalize the image to [0, 1]
    
        # Ensure the channels are in RGB (if the model expects RGB, or BGR depending on training)
        image_input = image_normalized[..., ::-1]  # Convert to RGB if needed (BGR to RGB)
    
        # Add batch dimension: (1, 3, height, width)
        image_input = np.transpose(image_input, (2, 0, 1))  # Change from HWC to CHW format
        image_input = np.expand_dims(image_input, axis=0)  # Add batch dimension (1, 3, height, width)
    
        # Run inference using ONNX Runtime
        inputs = {self.model.get_inputs()[0].name: image_input}
        preds = self.model.run(None, inputs)
    
        # Fix the shape issue: Ensure predictions are in the right format
        preds = np.array(preds[0])  # Convert to numpy array for easier manipulation
        if preds.ndim == 3:
            preds = preds.transpose(0, 2, 1)  # Transpose if necessary
        
        # Extract output
        results = self.__extract_output(
            preds=preds,
            image_shape=image.shape[:2],
            input_shape=(height, width),
            score=score,
            nms=nms,
            confidence=confidence
        )
        return results


# Initialize detection model
detection = Detection(
    model_path='best1.onnx', 
    classes=['Bonnet', 'Bumper', 'Dickey', 'Door', 'Fender', 'Light', 'Windshield']
)

@app.post('/detection')
async def post_detection(file: UploadFile = File(...)):
    try:
        # Read the file as bytes
        content = await file.read()
        
        # Check if the uploaded file is a video or image based on file extension
        if file.filename.endswith(('mp4', 'avi', 'mov', 'mkv')):
            # Process as video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
                # Save video content to a temporary file
                temp_video_file.write(content)
                temp_video_path = temp_video_file.name
            
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Failed to read video file")
            
            results = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform detection on each frame
                detection_results = detection(frame)
                
                # Optionally, draw bounding boxes on the frame (for visualization)
                for box, conf, cls in zip(detection_results['boxes'], detection_results['confidences'], detection_results['classes']):
                    left, top, width, height = box
                    cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
                    cv2.putText(frame, f"{cls} ({conf:.2f}%)", (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Append the results of the current frame
                results.append(detection_results)
            
            cap.release()

            # Optionally, delete the temporary video file
            os.remove(temp_video_path)

            return {"status": "success", "frames_results": results}
        
        elif file.filename.endswith(('jpg', 'jpeg', 'png')):
            # Process as image
            image = Image.open(io.BytesIO(content)).convert("RGB")
            image = np.array(image)
            image = image[:, :, ::-1].copy()  # Convert RGB to BGR
            results = detection(image)

            # Convert the results to Python native types for JSON serialization
            results_serializable = {
                "boxes": [list(box) for box in results["boxes"]],
                "confidences": [float(conf) for conf in results["confidences"]],
                "classes": results["classes"]
            }

            return {"status": "success", "detection_results": results_serializable}

        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the file: {str(e)}")




if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)