# Step 1: Install Dependencies
!pip install ultralytics opencv-python numpy scipy norfair

# Step 2: Import Libraries
from ultralytics import YOLO
import cv2
import numpy as np
from norfair import Detection, Tracker

# Step 3: Load YOLOv8 Model
model = YOLO('yolov8n.pt')  # Pre-trained YOLOv8 model

# Step 4: Initialize Tracker
tracker = Tracker(distance_function="euclidean", distance_threshold=30)

# Step 5: Process Video for Object Tracking with Obstacle Avoidance
def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracked_objects = {}
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = []

        # Initialize counters for obstacles in different regions
        count_center = 0
        count_left = 0
        count_right = 0
        traffic_light_detected = False  # Flag for traffic light detection

        # Define region thresholds (customize these as needed)
        left_limit = width * 0.3
        right_limit = width * 0.7
        vertical_threshold = height * 0.5  # Only consider obstacles in the lower half

        # Loop over detections for vehicles, obstacles, and traffic lights
        for result in results:
            boxes = result.boxes.xyxy
            classes = result.boxes.cls
            confidences = result.boxes.conf

            for box, cls_id, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(cls_id)]
                confidence = conf.item()

                # Detect traffic lights
                if label == 'traffic light':
                    traffic_light_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    continue  # Skip further processing for traffic lights

                # Process vehicles (non-ego vehicles)
                if label in ['car', 'bus', 'truck', 'motorcycle']:
                    detections.append(Detection(points=np.array([[x1, y1], [x2, y2]])))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    # Evaluate position for obstacle warnings
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    if center_y > vertical_threshold:
                        if left_limit <= center_x <= right_limit:
                            count_center += 1
                        elif center_x < left_limit:
                            count_left += 1
                        elif center_x > right_limit:
                            count_right += 1

                # Process obstacles like persons
                elif label == 'person':
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    # Evaluate position for obstacle warnings
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    if center_y > vertical_threshold:
                        if left_limit <= center_x <= right_limit:
                            count_center += 1
                        elif center_x < left_limit:
                            count_left += 1
                        elif center_x > right_limit:
                            count_right += 1

                # Additional categories (e.g., on-road irregularities) can be added here

        # Decide on instruction based on obstacle distribution,
        # but only if a traffic light is NOT detected.
        instruction = None
        if not traffic_light_detected:
            if count_center > 0:
                instruction = "Obstacle Ahead! Slow Down!"
            elif count_left > count_right and count_left > 0:
                instruction = "Obstacles on Left! Change Direction Right!"
            elif count_right > count_left and count_right > 0:
                instruction = "Obstacles on Right! Change Direction Left!"
        else:
            # If a traffic light is detected, assume the vehicle is stopped;
            # so do not overlay any additional instructions.
            instruction = None

        # Update tracker for vehicles
        tracked_objects_list = tracker.update(detections=detections)
        for obj in tracked_objects_list:
            obj_id = obj.id
            center = np.mean(obj.estimate, axis=0)
            center_x, center_y = center
            if obj_id in tracked_objects:
                prev_x, prev_y = tracked_objects[obj_id]
                speed = np.linalg.norm([center_x - prev_x, center_y - prev_y]) * fps  # Pixels per second
                cv2.putText(frame, f'ID {obj_id} Speed: {speed:.2f}', (int(center_x), int(center_y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            tracked_objects[obj_id] = (center_x, center_y)

        # Display the obstacle instruction if one is determined
        if instruction is not None:
            cv2.putText(frame, instruction, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print("Processing complete. Video saved at:", output_path)

# Step 6: Run the Full Pipeline with Video
process_video('/content/drive/MyDrive/4644521-uhd_2562_1440_30fps.mp4', 'output_video1.mp4')
