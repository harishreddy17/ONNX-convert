import onnxruntime as ort
import cv2
import numpy as np

# Initialize the ONNX Runtime session
model_path = "best1.onnx"
session = ort.InferenceSession(model_path)

# Define the video path
video_path = "test2.mp4"  # Replace with the path to your test video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError("Video could not be loaded. Check the path.")

# Function to extract predictions and filter based on confidence and NMS
def extract_output(preds, image_shape, input_shape, classes, score=0.1, nms=0.4, confidence=0.0):
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

        # Only keep boxes where confidence score exceeds the threshold
        if classes_score[class_id] > score:
            confs.append(conf)
            label = classes[int(class_id)]
            class_ids.append(label)

            # Extract boxes
            x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
            left = int((x - 0.5 * w) * x_factor)
            top = int((y - 0.5 * h) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            box = [left, top, width, height]
            boxes.append(box)

    # Apply Non-Maximum Suppression (NMS)
    r_class_ids, r_confs, r_boxes = [], [], []
    indexes = cv2.dnn.NMSBoxes(boxes, confs, confidence, nms)

    if len(indexes) > 0:
        indexes = indexes.flatten()  # Flatten the indexes array to get a list of indices
        for i in indexes:
            r_class_ids.append(class_ids[i])
            r_confs.append(confs[i] * 100)
            r_boxes.append(boxes[i])

    return {
        'boxes': r_boxes,
        'confidences': r_confs,
        'classes': r_class_ids
    }

# Define classes (ensure these match your model's labels)
classes = ['Bonnet', 'Bumper', 'Dickey', 'Door', 'Fender', 'Light', 'Windshield']

# Process each frame of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if the video is finished

    # Prepare the input blob
    input_shape = session.get_inputs()[0].shape[2:4]  # Expected input shape, e.g., (640, 640)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, input_shape, swapRB=True, crop=False)

    # Perform inference
    preds = session.run(None, {session.get_inputs()[0].name: blob})[0]  # Accessing the output directly
    preds = preds.transpose((0, 2, 1))  # Adjust dimensions if needed

    # Extract predictions with confidence threshold and NMS
    results = extract_output(
        preds=preds,
        image_shape=frame.shape[:2],
        input_shape=input_shape,
        classes=classes,
        score=0.01,     # Minimum confidence score to filter predictions (0.5 means 50% confidence)
        nms=0.0,      # NMS threshold to remove overlapping boxes (adjust as necessary)
        confidence=0.0
    )

    # Visualize predictions on the frame
    for box, cls, conf in zip(results['boxes'], results['classes'], results['confidences']):
        left, top, width, height = box
        label = f"{cls}: {conf:.2f}"
        cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow("Detections", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()