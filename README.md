# license_plate_recognition_using_yolov8_and_easyocr
![Screenshot 2024-12-22 211149](https://github.com/user-attachments/assets/fc4adf71-289a-4f2f-96f3-21f2cebc0784)

## google doc for the project overview
![CLICK HERE](https://drive.google.com/file/d/1i0qCDdtO4tOhHNmj0HC8J-lL7rPlVAmT/view?usp=sharing)
## Workflow

### 1. Object Detection
- Used **YOLOv8** (`yolov8n.pt`) for detecting number plates.
- Integrated a custom-trained model tailored for Indian number plates.

### 2. Text Recognition
- Incorporated **EasyOCR** for extracting text from detected plates.
- Modified the logic in EasyOCR to improve recognition accuracy for region-specific number plates.

### 3. Bounding Box Stability
- Interpolated results from CSV outputs to ensure stable bounding boxes across frames.
- Selected number plate predictions based on the highest confidence score.

### 4. Future Improvements
- Plan to train EasyOCR on a customized dataset to enhance recognition accuracy for Indian number plates.

---

## Tools and Technologies

- **YOLOv8:** For object detection.
- **EasyOCR:** For text recognition.
- **Python:** For implementation.
- **CSV Interpolation:** To stabilize bounding boxes.

