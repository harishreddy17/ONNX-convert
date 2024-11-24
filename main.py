import base64
import os
import tempfile

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("last.pt")
UPLOAD_DIR = "processed_images"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


@app.post("/process-video-frames/")
async def process_video_frames(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(await file.read())
            input_path = tmp_file.name

        cap = cv2.VideoCapture(input_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        processed_frames = []
        damage_details = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            height, width = frame.shape[:2]
            new_width = width
            new_height = height
            scale = 1
            # Check if height or width is greater than 640
             
            if (width > 640):
                 # Scale the width to 640 and adjust height to maintain aspect ratio
                new_width = 640  

            if (height > 420):
                # Scale the height to 640 and adjust width to maintain aspect ratio
                new_height = 420  

            """  # New dimensions
            new_width =  int(width * scale)
            new_height = int(height * scale) """
            frame =  cv2.resize(frame, (new_width, new_height))
            results = model(frame)
            frame_damages = []

            if len(results[0].boxes) > 0:
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])

                        damage_info = {
                            "type": model.names[class_id],
                            "confidence": f"{confidence:.2f}",
                            "location": f"x:{int(x1)},y:{int(y1)}",
                        }
                        frame_damages.append(damage_info)

                        cv2.rectangle(
                            frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 0),
                            2,
                        )
                        label = f"{model.names[class_id]} {confidence:.2f}"
                        cv2.putText(
                            frame,
                            label,
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )

            _, buffer = cv2.imencode(".jpg", frame)
            base64_frame = base64.b64encode(buffer).decode("utf-8")
            processed_frames.append({"frame": base64_frame, "damages": frame_damages})
            damage_details.extend(frame_damages)

        cap.release()
        os.remove(input_path)

        return JSONResponse(
            {
                "frames": processed_frames,
                "total_frames": frame_count,
                "fps": fps,
                "damage_summary": damage_details,
            }
        )

    except Exception as e:
        if "input_path" in locals() and os.path.exists(input_path):
            os.remove(input_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    output_filename = f"processed_{os.urandom(8).hex()}.jpg"
    output_path = os.path.join(UPLOAD_DIR, output_filename)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(await file.read())
            input_path = tmp_file.name

        frame = cv2.imread(input_path)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not read the image")

        damage_details = []
        height, width = frame.shape[:2]
        new_width = width
        new_height = height
        scale = 1
        # Check if height or width is greater than 640
             
        if (width > 640):
            # Scale the width to 640 and adjust height to maintain aspect ratio
            new_width = 640  

        if (height > 420):
            # Scale the height to 640 and adjust width to maintain aspect ratio
            new_height = 420  

            """  # New dimensions
        new_width =  int(width * scale)
        new_height = int(height * scale) """
        frame =  cv2.resize(frame, (new_width, new_height))
        results = model(frame)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                damage_info = {
                    "type": model.names[class_id],
                    "confidence": f"{confidence:.2f}",
                    "location": f"x:{int(x1)},y:{int(y1)}",
                }
                damage_details.append(damage_info)

                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                )
                label = f"{model.names[class_id]} {confidence:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        cv2.imwrite(output_path, frame)

        if os.path.exists(input_path):
            os.remove(input_path)

        if not os.path.exists(output_path):
            raise RuntimeError("Failed to create processed image file")

        _, buffer = cv2.imencode(".jpg", frame)
        base64_image = base64.b64encode(buffer).decode("utf-8")

        return JSONResponse({"image": base64_image, "damage_details": damage_details})

    except Exception as e:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        import traceback

        print(traceback.format_exc())

        raise HTTPException(status_code=500, detail=str(e))
