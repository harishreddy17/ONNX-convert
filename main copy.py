import io
import uvicorn
import numpy as np
import nest_asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import cv2
from typing import List
from numpy import ndarray
from typing import Tuple
from PIL import Image
import base64
from fastapi import Response

class Detection:
    def __init__(self, 
                 model_path: str, 
                 classes: List[str]):
        self.model_path = model_path
        self.classes = classes
        self.model = self.__load_model()

    def __load_model(self) -> cv2.dnn_Net:
        net = cv2.dnn.readNet(self.model_path)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use CPU if CUDA is unavailable
        return net

    def __extract_output(self, 
                         preds: ndarray, 
                         image_shape: Tuple[int, int], 
                         input_shape: Tuple[int, int],
                         score: float=0.1,
                         nms: float=0.0, 
                         confidence: float=0.0) -> dict:
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
        for i in indexes:
            r_class_ids.append(class_ids[i])
            r_confs.append(confs[i]*100)
            r_boxes.append(boxes[i])

        return {
            'boxes': r_boxes, 
            'confidences': r_confs, 
            'classes': r_class_ids
        }

    def __call__(self,
                 image: ndarray, 
                 width: int=640, 
                 height: int=640, 
                 score: float=0.1,
                 nms: float=0.0, 
                 confidence: float=0.0) -> dict:
        blob = cv2.dnn.blobFromImage(
            image, 1/255.0, (width, height), 
            swapRB=True, crop=False
        )
        self.model.setInput(blob)
        preds = self.model.forward()
        preds = preds.transpose((0, 2, 1))

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
    model_path='best.onnx', 
    classes=['damaged door', 'damaged window', 'damaged headlight', 'damaged mirror', 'dent', 'damaged hood', 'damaged bumper', 'damaged wind shield']
)

app = FastAPI()

@app.post('/detection')
def post_detection(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
        image = np.array(image)
        image = image[:,:,::-1].copy()  # Convert RGB to BGR
        results = detection(image)
        
        # Convert numpy arrays to lists for JSON serialization
        results['boxes'] = [box for box in results['boxes']]
        results['confidences'] = [float(conf) for conf in results['confidences']]
        results['classes'] = [str(cls) for cls in results['classes']]
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8080)
