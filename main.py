import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort
from util import get_car, write_csv, read_license_plate ,preprocess_image

# Load the pretrained YOLOv8 models for vehicle and license plate detection
vehicle_model = YOLO('yolov8n.pt')  # Pretrained YOLOv8 for vehicle detection
license_plate_model = YOLO('best.pt')  # Your custom trained license plate model

# Initialize SORT tracker for vehicle tracking
mot_tracker = Sort()

# Load the video
cap = cv2.VideoCapture('video2.mp4')
frame_nmr = -1
results = {}

while True:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        break

    results[frame_nmr] = {}

    # Vehicle Detection using YOLOv8
    vehicle_detections = vehicle_model(frame)[0]
    vehicles = []
    for detection in vehicle_detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        vehicles.append([x1, y1, x2, y2, score])

    # Track vehicles using SORT
    tracked_vehicles = mot_tracker.update(np.array(vehicles))

    # License Plate Detection using custom trained YOLO model
    license_plate_detections = license_plate_model(frame)[0]
    for license_plate in license_plate_detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Assign the license plate to the nearest vehicle
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, tracked_vehicles)

        if car_id != -1:
            # Crop the license plate from the frame
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            
            #img2=preprocess_image(license_plate_crop)

            license_plate_text, license_plate_score = read_license_plate(license_plate_crop)
            
            # Store the results in the dictionary
            if car_id not in results[frame_nmr]:
                results[frame_nmr][car_id] = {}

            results[frame_nmr][car_id] = {
                'car_bbox': [xcar1, ycar1, xcar2, ycar2],
                'plate_bbox': [x1, y1, x2, y2],
                'plate_score': score,
                'license_plate_text': license_plate_text if license_plate_text else 'N/A',
                'license_plate_score': license_plate_score if license_plate_score else 0.0
            }

cap.release()

# Write the results to a CSV file
write_csv(results, 'test.csv')

