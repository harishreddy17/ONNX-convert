import io
import uvicorn
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
from typing import List
from numpy import ndarray
from typing import Tuple
from PIL import Image

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to specific origins for better security
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
        # Load the ONNX model using onnxruntime
        ort_session = ort.InferenceSession(self.model_path)
        return ort_session

    def __extract_output(self, 
                         preds: ndarray, 
                         image_shape: Tuple[int, int], 
                         input_shape: Tuple[int, int],
                         score: float = 0.1,
                         nms: float = 0.0, 
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

        r_class_ids, r_confs, r_boxes = [], [], []
        indexes = cv2.dnn.NMSBoxes(boxes, confs, confidence, nms) 
        for i in indexes.flatten():
            r_class_ids.append(class_ids[i])
            r_confs.append(confs[i]*100)
            r_boxes.append(boxes[i])

        return {
            'boxes': r_boxes, 
            'confidences': r_confs, 
            'classes': r_class_ids
        }

    def __call__(self, image: ndarray, width: int = 640, height: int = 640, score: float = 0.1, nms: float = 0.0, confidence: float = 0.0) -> dict:
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
    
        print(f"Predictions shape: {preds[0].shape}")  # Debugging line to check the shape of the model output
    
        # Fix the shape issue: Ensure predictions are in the right format
        preds = np.array(preds[0])  # Convert to numpy array for easier manipulation
        if preds.ndim == 3:
            preds = preds.transpose(0, 2, 1)  # Transpose if necessary
        
        print(f"Predictions after transpose: {preds.shape}")  # Debugging line
    
        # Extract output
        results = self.__extract_output(
            preds=preds,
            image_shape=image.shape[:2],
            input_shape=(height, width),
            score=score,
            nms=nms,
            confidence=confidence
        )
        print(f"Extracted results: {results}")  # Debugging line
        return results


# Initialize detection model
detection = Detection(
    model_path='ctruonxx.onnx', 
    classes=['Bonnet', 'Bumper', 'Dickey', 'Door', 'Fender', 'Light', 'Windshield']
)

@app.post('/detection')
async def post_detection(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        print(f"Image shape (before conversion): {np.array(image).shape}")  # Debugging line
        image = np.array(image)
        image = image[:, :, ::-1].copy()  # Convert RGB to BGR
        print(f"Image shape (after conversion): {image.shape}")  # Debugging line

        results = detection(image)
        print(f"Detection results: {results}")  # Debugging line

        results['boxes'] = [box for box in results['boxes']]
        results['confidences'] = [float(conf) for conf in results['confidences']]
        results['classes'] = [str(cls) for cls in results['classes']]
        
        return results
    except Exception as e:
        print(f"Error: {str(e)}")  # Debugging line
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8080)
