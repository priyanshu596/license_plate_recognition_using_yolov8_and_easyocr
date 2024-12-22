


##new code delete if doesn't works

import ast
import cv2
import numpy as np
import pandas as pd

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    """
    Draws extended borders around a bounding box.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Top-left
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    # Bottom-left
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    # Top-right
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    # Bottom-right
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

# Load results from CSV
results = pd.read_csv('test_interpolated.csv')

# Load video
video_path = 'video2.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Video properties
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# Prepare license plate data
license_plate = {}
for car_id in np.unique(results['car_id']):
    try:
        # Get the license plate with the highest score
        max_score = np.amax(results[results['car_id'] == car_id]['license_number_score'])
        license_plate[car_id] = {
            'license_crop': None,
            'license_plate_number': results[
                (results['car_id'] == car_id) &
                (results['license_number_score'] == max_score)
            ]['license_number'].iloc[0]
        }

        # Get the frame number where the license plate has the highest score
        frame_nmr = results[
            (results['car_id'] == car_id) &
            (results['license_number_score'] == max_score)
        ]['frame_nmr'].iloc[0]

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nmr)
        ret, frame = cap.read()

        if not ret or frame is None:
            print(f"Error: Unable to read frame {frame_nmr} for car ID {car_id}.")
            continue

        # Get license plate bounding box
        x1, y1, x2, y2 = ast.literal_eval(
            results[
                (results['car_id'] == car_id) &
                (results['license_number_score'] == max_score)
            ]['license_plate_bbox'].iloc[0]
            .replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
        )

        license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

        license_plate[car_id]['license_crop'] = license_crop

    except Exception as e:
        print(f"Error processing car ID {car_id}: {e}")

# Reset video reader
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Process video frame-by-frame
frame_nmr = -1
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1

    if not ret:
        break

    df_ = results[results['frame_nmr'] == frame_nmr]
    for row_indx in range(len(df_)):
        try:
            # Draw vehicle bounding box
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(
                df_.iloc[row_indx]['car_bbox']
                .replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
            )
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)),
                        color=(0, 255, 0), thickness=25, line_length_x=200, line_length_y=200)

            # Draw license plate bounding box
            x1, y1, x2, y2 = ast.literal_eval(
                df_.iloc[row_indx]['license_plate_bbox']
                .replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
            )
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

            # Overlay license plate image and text
            license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']

            if license_crop is not None:
                H, W, _ = license_crop.shape

                # Calculate overlay region and clip to frame dimensions
                y_start = max(0, int(car_y1) - H - 100)
                y_end = max(0, int(car_y1) - 100)
                x_start = max(0, int((car_x2 + car_x1 - W) / 2))
                x_end = min(frame.shape[1], x_start + W)

                if y_end - y_start == H and x_end - x_start == W:
                    frame[y_start:y_end, x_start:x_end] = license_crop

                # White background for text
                text_y_start = max(0, y_start - 300)
                text_y_end = max(0, y_start - 100)
                frame[text_y_start:text_y_end, x_start:x_end] = (255, 255, 255)

                # Add license plate number text
                license_number = license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number']
                if license_number:
                    license_number = str(license_number)
                    (text_width, text_height), _ = cv2.getTextSize(
                        license_number, cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17
                    )
                    text_x = max(0, int((car_x2 + car_x1 - text_width) / 2))
                    text_y = max(0, y_start - 200 + (text_height // 2))
                    cv2.putText(frame, license_number, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 4.3, (0, 0, 0), 17)

        except Exception as e:
            print(f"Error drawing on frame {frame_nmr}: {e}")

    out.write(frame)

out.release()
cap.release()
print("Video processing completed. Output saved to 'output.mp4'.")




