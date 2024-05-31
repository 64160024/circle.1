import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model
model = YOLO("circle_best.pt")

# Open the camera
cap = cv2.VideoCapture(0)

def draw_lines_between_circles(frame, circles):
    # Sort circles by their y-coordinate (the second element in each circle tuple)
    sorted_circles = sorted(circles, key=lambda c: c[1])
    
    # Draw lines between the circles
    if len(sorted_circles) >= 2:
        for i in range(len(sorted_circles) - 1):
            cv2.line(frame, sorted_circles[i], sorted_circles[i + 1], (0, 255, 0), 2)
    
    # Calculate the midpoint between the second circle and the frame center
    if len(sorted_circles) >= 3:
        center_x = frame.shape[1] // 2
        center_y = sorted_circles[1][1]  # Y-coordinate of the second circle
        # Draw lines from the midpoint to the left and right edges of the frame
        cv2.line(frame, (center_x, center_y), (0, center_y), (0, 255, 0), 2)  # Left line
        cv2.line(frame, (center_x, center_y), (frame.shape[1], center_y), (0, 255, 0), 2)  # Right line

        # Calculate angles between lines
        angle_1_2 = np.arctan2(sorted_circles[0][1] - sorted_circles[1][1], sorted_circles[0][0] - sorted_circles[1][0]) * 180 / np.pi
        angle_2_3 = np.arctan2(sorted_circles[1][1] - sorted_circles[2][1], sorted_circles[1][0] - sorted_circles[2][0]) * 180 / np.pi
        
        # Use provided formulas to calculate angles
        angle_A = angle_B = 0
        if angle_1_2 > 90:
            angle_A = 180 - angle_1_2
        elif angle_1_2 < -90:
            angle_A = 180 - (-angle_1_2)
        else:
            angle_A = angle_1_2
        
        if angle_2_3 > 90:
            angle_B = 180 - angle_2_3
        elif angle_2_3 < -90:
            angle_B = 180 - (-angle_2_3)
        else:
            angle_B = -angle_2_3
        
        # Convert negative angles to positive
        angle_A = abs(angle_A)
        angle_B = abs(angle_B)

        # Calculate midpoints between circles
        midpoint_1_2 = ((sorted_circles[0][0] + sorted_circles[1][0]) // 2, (sorted_circles[0][1] + sorted_circles[1][1]) // 2)
        midpoint_2_3 = ((sorted_circles[1][0] + sorted_circles[2][0]) // 2, (sorted_circles[1][1] + sorted_circles[2][1]) // 2)

        # Print and draw angles at midpoints
        angle_text_1_2 = f"CVA : {round(angle_A, 2)} degrees"
        angle_text_2_3 = f"FSP : {round(angle_B, 2)} degrees"
        
        # Draw text
        font_scale = 0.5
        cv2.putText(frame, angle_text_1_2, midpoint_1_2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (127, 0, 255), 1)
        cv2.putText(frame, angle_text_2_3, midpoint_2_3, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (127, 0, 255), 1)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the camera
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # Extract detections from results
        detections = results[0].boxes

        # List to hold circle centers
        circle_centers = []

        for detection in detections:
            if detection.cls == 0:  # Assuming '0' is the class ID for circles
                # Get the coordinates of the bounding box
                x1, y1, x2, y2 = map(int, detection.xyxy[0])
                # Calculate the center of the bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                circle_centers.append((center_x, center_y))

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # If exactly 3 circles are detected, draw lines between them
        if len(circle_centers) == 3:
            draw_lines_between_circles(annotated_frame, circle_centers)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if there is an error reading the frame
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
