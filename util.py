import csv
import string
import easyocr
import numpy as np
import re

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {
    'O': '0',  # Zero often misread as 'O'
    'I': '1',  # 'I' misread as '1'
    'S': '5',  # 'S' misread as '5'
    'B': '8',  # 'B' misread as '8'
    'G': '6',  # 'G' misread as '6'
    'Z': '2',  # 'Z' misread as '2'
    'L': '1'   # 'L' misread as '1'
}

# Digit to character mapping (reversing the above mistakes)
dict_int_to_char = {
    '0': 'O',  # '0' misread as 'O'
    '1': 'I',  # '1' misread as 'I'
    '5': 'S',  # '5' misread as 'S'
    '8': 'B',  # '8' misread as 'B'
    '6': 'G',  # '6' misread as 'G'
    '2': 'Z'   # '2' misread as 'Z'
}

def write_csv(results, output_path):
    """
    Write the results to a CSV file.
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox',
                         'license_plate_bbox_score', 'license_number', 'license_number_score'])

        for frame_nmr in results.keys():
            for car_id, data in results[frame_nmr].items():
                if 'license_plate_text' in data:
                    writer.writerow([
                        frame_nmr, car_id,
                        '[{} {} {} {}]'.format(*data['car_bbox']),
                        '[{} {} {} {}]'.format(*data['plate_bbox']),
                        data['plate_score'],
                        data['license_plate_text'],
                        data['license_plate_score']
                    ])

import re

def license_complies_format(text):
    """
    Check if the license plate text complies with general patterns for Indian license plates.
    This version is simplified and allows flexibility for detecting potential formats.
    """
    # Remove spaces, dashes, or special characters
    text = re.sub(r'[^A-Z0-9]', '', text.upper())  # Allow only alphanumeric characters

    # Flexible pattern for Indian license plates
    pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,2}[0-9]{1,4}$'  # Covers common formats (e.g., DL12AB1234, TN23A1234)
    
    return bool(re.match(pattern, text))

def format_license(text):
    """
    Format the license plate text by correcting common misread characters (e.g., '0' to 'O').
    This helps normalize the text for easier validation.
    """
    # Mapping for commonly misread characters
    char_correction = {
    '0': 'O',  # Zero to 'O'
    '1': 'I',  # One to 'I'
    '5': 'S',  # Five to 'S'
    '8': 'B',  # Eight to 'B'
    'G': '6',  # 'G' to '6'
    'Z': '2',  # 'Z' to '2'
    'L': '1',  # 'L' to '1'
    'D': '0',  # 'D' to '0' (sometimes misread)
    'P': 'R',  # 'P' can be misread as 'R'
    'Q': 'O',  # 'Q' misread as 'O' in some cases
    'C': 'O',  # 'C' to 'O' (in fonts where these are similar)
    'J': '1',  # 'J' to '1' (misread in certain fonts)
    'A': '4',  # 'A' to '4' (common OCR misreads)
    'E': '3'   # 'E' to '3' (sometimes misread as 3)
}

    # Correct the characters based on the mapping
    formatted_text = ''.join(char_correction.get(char.upper(), char.upper()) for char in text)

    return formatted_text

def read_license_plate(license_plate_crop):
    """
    Use EasyOCR to read license plate text from the cropped image.
    """
    detections = reader.readtext(license_plate_crop)
    
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')  # Normalize text
        print(f"Detected Text: {text}, Confidence: {score}")  # Debugging line

        # Validate and format license plate text
        if license_complies_format(text):
            return format_license(text), score

    return None, None  # Return None if no valid license plate is found




def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.
    """
    x1, y1, x2, y2, _, _ = license_plate  # Unpack license plate coordinates

    for vehicle in vehicle_track_ids:
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle
        # Check if the license plate bounding box is fully within the car bounding box
        if x1 >= xcar1 and y1 >= ycar1 and x2 <= xcar2 and y2 <= ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id

    return -1, -1, -1, -1, -1  # Return default values if no car is found

import cv2
import numpy as np

def preprocess_image(image, denoise_strength=10, threshold_value=127, max_value=255):
    """
    Preprocess the input image by converting to grayscale, applying denoising, and binarization.

    Args:
        image (numpy.ndarray): Input image in BGR or BGRA format.
        denoise_strength (int): Strength of the denoising filter (default: 10).
        threshold_value (int): Threshold value for binarization (default: 127).
        max_value (int): Maximum value for binary thresholding (default: 255).

    Returns:
        numpy.ndarray: Preprocessed image ready for OCR or further processing.
    """
    if image is None:
        raise ValueError("Input image is None.")

    # Step 1: Convert to grayscale
    if len(image.shape) == 2:  # If already grayscale
        grayscale = image
    elif image.shape[2] == 4:  # BGRA image
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:  # BGR image
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    

    # Step 3: Apply binary thresholding
    _, binary = cv2.threshold(grayscale, threshold_value, max_value, cv2.THRESH_BINARY)

    return binary
