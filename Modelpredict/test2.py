import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

# Create a client for inference
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="07quyzZXOaGmcC69IFrz"
)

def process_frame(frame, model_id):
    # Convert the frame to an image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Perform inference on the frame
    result = CLIENT.infer(
        image,
        model_id=model_id
    )

    # Filter predictions to find circles
    circle_predictions = [pred for pred in result['predictions'] if pred['class'] == 'circle-va51']

    # Check if there are exactly 3 circles
    if len(circle_predictions) == 3:
        print("The image contains exactly 3 circles.")

        # Sort predictions by y-coordinate from top to bottom
        sorted_predictions = sorted(circle_predictions, key=lambda x: x['y'])

        draw = ImageDraw.Draw(image)

        # Draw bounding boxes and display the result
        for i, prediction in enumerate(sorted_predictions):
            x0 = prediction['x'] - prediction['width'] / 2
            y0 = prediction['y'] - prediction['height'] / 2
            x1 = prediction['x'] + prediction['width'] / 2
            y1 = prediction['y'] + prediction['height'] / 2
            draw.rectangle([x0, y0, x1, y1], outline="red", width=2)

        # Connect boxes and calculate angles
        if len(sorted_predictions) >= 2:
            first_box = sorted_predictions[0]
            second_box = sorted_predictions[1]
            x0_first = first_box['x']
            y0_first = first_box['y']
            x1_second = second_box['x']
            y1_second = second_box['y']
            draw.line([x0_first, y0_first, x1_second, y1_second], fill="blue", width=2)

            angle_rad1 = np.arctan2(y1_second - y0_first, x1_second - x0_first)
            angle_deg1 = np.degrees(angle_rad1)
            if angle_deg1 > 90:  # CVA
                angle_A = 180 - angle_deg1
            elif angle_deg1 < 0:
                angle_A = 180 + angle_deg1
            else:
                angle_A = angle_deg1
            angle_A = round(angle_A, 2)
            print(f"CVA : {angle_A} degrees")
            draw.text((x0_first + 30, y0_first + 30), f"CVA = {angle_A}°", fill="OrangeRed")

        if len(sorted_predictions) >= 2:
            second_box = sorted_predictions[1]
            y_position = second_box['y']
            draw.line((0, y_position, image.width, y_position), fill='lawngreen', width=2)

        if len(sorted_predictions) >= 3:
            second_box = sorted_predictions[2]
            third_box = sorted_predictions[1]
            x0_second = second_box['x']
            y0_second = second_box['y']
            x1_third = third_box['x']
            y1_third = third_box['y']
            draw.line([x0_second, y0_second, x1_third, y1_third], fill="blue", width=2)

            angle_rad2 = np.arctan2(y1_third - y0_second, x1_third - x0_second)
            angle_deg2 = np.degrees(angle_rad2)
            if angle_deg2 > 90:  # FSP
                angle_B = 180 - angle_deg2
            elif angle_deg2 < -90:
                angle_B = 180 + angle_deg2
            else:
                angle_B = -angle_deg2
            angle_B = round(angle_B, 2)
            print(f"FSP : {angle_B} degrees")
            draw.text((x0_second + 30, y0_second - 30), f"FSP = {angle_B}°", fill="Magenta")

        print(f"total angle : {round(angle_A + angle_B, 2)} degrees")

    return np.array(image)

def process_video(video_path, model_id):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    output_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Processing frame {frame_count}")
        processed_frame = process_frame(frame, model_id)
        output_frames.append(processed_frame)
        frame_count += 1

    cap.release()

    height, width, layers = output_frames[0].shape
    video_name = f"processed_{video_path}"
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

    for frame in output_frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"Video saved as {video_name}")

video_path = "video.MP4"
process_video(video_path, "black_marker/1")
