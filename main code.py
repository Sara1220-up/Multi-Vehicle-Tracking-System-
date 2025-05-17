import cv2
from ultralytics import YOLO
from norfair import Detection, Tracker, draw_tracked_objects
import os

# Load YOLOv8 model
model = YOLO("models/yolov8.pt")  #can have any yolo model
CLASS_NAMES = model.names  # Class labels (e.g., car, truck, bus)

# Norfair tracker setup
tracker = Tracker(distance_function="euclidean", distance_threshold=30)

# Create output folder if it doesn't exist
os.makedirs("output", exist_ok=True)

# Open video input
cap = cv2.VideoCapture("videos/input.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video writer
out = cv2.VideoWriter("videos/output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# Open text file to save comparison results
output_log = open("output/comp_file.txt", "w")
frame_count = 0

# Function to convert YOLO detections to Norfair format
def yolo_to_norfair_detections(results, confidence_threshold=0.4):
    detections = []
    for result in results:
        for box in result.boxes:
            if box.conf < confidence_threshold:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            detections.append(Detection(points=[[cx, cy]], scores=[box.conf.cpu().item()]))
    return detections

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model.predict(frame, verbose=False)
    detections = yolo_to_norfair_detections(results)
    tracked_objects = tracker.update(detections=detections)

    # Draw YOLO boxes and Norfair tracking
    output_log.write(f"Frame: {frame_count}\n")

    for result in results:
        for box in result.boxes:
            if box.conf < 0.4:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = CLASS_NAMES[cls_id]
            conf = float(box.conf[0])
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Try to match with tracked object
            matched_id = None
            for obj in tracked_objects:
                tx, ty = obj.estimate[0]
                if abs(tx - cx) < 30 and abs(ty - cy) < 30:
                    matched_id = obj.id
                    break

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Log results
            output_log.write(
                f"ID: {matched_id if matched_id else 'NA'}, "
                f"Class: {label}, Confidence: {conf:.2f}, "
                f"Box: [{x1}, {y1}, {x2}, {y2}]\n"
            )

    #  tracked object IDs (center point)
    draw_tracked_objects(frame, tracked_objects, draw_ids=True)

    output_log.write("-" * 60 + "\n")

    # Save and show
    out.write(frame)
    cv2.imshow("Multi-Vehicle Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
out.release()
output_log.close()
cv2.destroyAllWindows()
