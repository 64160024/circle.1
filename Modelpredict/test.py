# import os
# from datetime import datetime
# from ultralytics import YOLO

# # Create a directory to store results if it doesn't exist
# output_dir = "results"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Load a model
# model = YOLO("best1.pt")  # pretrained YOLOv8n model

# # Run batched inference on a list of images
# results = model(["vdo2.MP4"])  # return a list of Results objects

# # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
    
#     # Generate filename using current date and time
#     current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     filename = os.path.join(output_dir, f"result_{current_time}.jpg")
    
#     result.show()  # display to screen
#     result.save(filename=filename)  # save to disk with the generated filename


# import os
# from datetime import datetime
# from ultralytics import YOLO

# # Create a directory to store results if it doesn't exist
# output_dir = "results"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Load a model
# model = YOLO("best1.pt")  # pretrained YOLOv8n model

# # Path to input video
# input_video_path = "vdo2.MP4"

# # Process video frames
# results = model(input_video_path)  # Perform inference on the video

# # Process results
# for i, result in enumerate(results):
#     # Generate filename using current date and time
#     current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     filename = os.path.join(output_dir, f"result_{current_time}_{i}.jpg")
    
#     # Save frame with detections
#     result.save(filename=filename)  # Save frame with detections
    
#     # Optionally, you can also display the frame
#     # result.show()  # Display frame with detections


##########โค้ดเอา vdo มา predict
# import cv2
# from ultralytics import YOLO


# # Load the YOLOv8 model
# model = YOLO("best1.pt")

# # Open the video file
# video_path = "vdo2.MP4"
# cap = cv2.VideoCapture(video_path)

# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     if success:
#         # Run YOLOv8 inference on the frame
#         results = model(frame)

#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()

#         # Display the annotated frame
#         cv2.imshow("YOLOv8 Inference", annotated_frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()



##############โค้ดเปิดกล้องเฉยๆ
# import cv2

# # เริ่มต้นการจับภาพจากกล้อง (โดยทั่วไปใช้ index 0)
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("ไม่สามารถเปิดกล้องได้")
#     exit()

# while True:
#     # อ่านภาพจากกล้อง
#     ret, frame = cap.read()

#     if not ret:
#         print("ไม่สามารถรับภาพจากกล้องได้")
#         break

#     # แสดงภาพ
#     cv2.imshow('frame', frame)

#     # รอการกดปุ่ม 'q' เพื่อปิดการแสดงผล
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # ปล่อยกล้องและปิดหน้าต่างแสดงผล
# cap.release()
# cv2.destroyAllWindows()


#######อันนี้ได้กรอบของวงกลมกับเส้นเชื่อม
# import cv2
# from ultralytics import YOLO

# # Load the YOLOv8 model
# model = YOLO("best1.pt")

# # Open the video file
# video_path = "vdo2.MP4"
# cap = cv2.VideoCapture(video_path)

# def draw_lines_between_circles(frame, circles):
#     # Sort circles by their y-coordinate (the second element in each circle tuple)
#     sorted_circles = sorted(circles, key=lambda c: c[1])
    
#     # Draw lines between the circles
#     if len(sorted_circles) >= 2:
#         for i in range(len(sorted_circles) - 1):
#             cv2.line(frame, sorted_circles[i], sorted_circles[i + 1], (0, 255, 0), 2)

# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     if success:
#         # Run YOLOv8 inference on the frame
#         results = model(frame)
        
#         # Extract detections from results
#         detections = results[0].boxes

#         # List to hold circle centers
#         circle_centers = []

#         for detection in detections:
#             if detection.cls == 0:  # Assuming '0' is the class ID for circles
#                 # Get the coordinates of the bounding box
#                 x1, y1, x2, y2 = map(int, detection.xyxy[0])
#                 # Calculate the center of the bounding box
#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
#                 circle_centers.append((center_x, center_y))

#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()

#         # If exactly 3 circles are detected, draw lines between them
#         if len(circle_centers) == 3:
#             draw_lines_between_circles(annotated_frame, circle_centers)

#         # Display the annotated frame
#         cv2.imshow("YOLOv8 Inference", annotated_frame)

#         # Break the loop if 'q' is pressed or 3 circles are detected in the same frame
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()


###outputเป็นรูปเก็บใส่โฟลเดอร์
# import cv2
# from ultralytics import YOLO
# import numpy as np
# import os
# from datetime import datetime

# # Load the YOLOv8 model
# model = YOLO("best70.pt")

# def draw_lines_between_circles(frame, circles, output_dir):
#     sorted_circles = sorted(circles, key=lambda c: c[1])
#     if len(sorted_circles) >= 2:
#         for i in range(len(sorted_circles) - 1):
#             cv2.line(frame, sorted_circles[i], sorted_circles[i + 1], (0, 255, 0), 2)
#     if len(sorted_circles) >= 3:
#         center_x = frame.shape[1] // 2
#         center_y = sorted_circles[1][1]
#         cv2.line(frame, (center_x, center_y), (0, center_y), (0, 255, 0), 2)
#         cv2.line(frame, (center_x, center_y), (frame.shape[1], center_y), (0, 255, 0), 2)
        
#         angle_1_2 = np.arctan2(sorted_circles[0][1] - sorted_circles[1][1], sorted_circles[0][0] - sorted_circles[1][0]) * 180 / np.pi
#         angle_2_3 = np.arctan2(sorted_circles[1][1] - sorted_circles[2][1], sorted_circles[1][0] - sorted_circles[2][0]) * 180 / np.pi
        
#         angle_A = 180 - abs(angle_1_2) if abs(angle_1_2) > 90 else abs(angle_1_2)
#         angle_B = 180 - abs(angle_2_3) if abs(angle_2_3) > 90 else abs(angle_2_3)
        
#         midpoint_1_2 = ((sorted_circles[0][0] + sorted_circles[1][0]) // 2, (sorted_circles[0][1] + sorted_circles[1][1]) // 2)
#         midpoint_2_3 = ((sorted_circles[1][0] + sorted_circles[2][0]) // 2, (sorted_circles[1][1] + sorted_circles[2][1]) // 2)
        
#         angle_text_1_2 = f"CVA : {round(angle_A, 2)} degrees"
#         angle_text_2_3 = f"FSP : {round(angle_B, 2)} degrees"
        
#         font_scale = 0.5
#         cv2.putText(frame, angle_text_1_2, midpoint_1_2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (127,0,255), 1)
#         cv2.putText(frame, angle_text_2_3, midpoint_2_3, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (127,0,255), 1)
        
#         os.makedirs(output_dir, exist_ok=True)
#         output_path = os.path.join(output_dir, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
#         cv2.imwrite(output_path, frame)
#         print(f"Saved frame with angles: CVA = {round(angle_A, 2)}, FSP = {round(angle_B, 2)} at {output_path}")

# def process_image(image_path, output_dir):
#     frame = cv2.imread(image_path)
#     results = model(frame)
#     detections = results[0].boxes
#     circle_centers = []
#     for detection in detections:
#         if detection.cls == 0:
#             x1, y1, x2, y2 = map(int, detection.xyxy[0])
#             center_x = (x1 + x2) // 2
#             center_y = (y1 + y2) // 2
#             circle_centers.append((center_x, center_y))
#     annotated_frame = results[0].plot()
#     if len(circle_centers) == 3:
#         draw_lines_between_circles(annotated_frame, circle_centers, output_dir)

# def process_video(video_path, output_dir):
#     cap = cv2.VideoCapture(video_path)
#     best_frame = None
#     best_circle_centers = []
#     while cap.isOpened():
#         success, frame = cap.read()
#         if success:
#             results = model(frame)
#             detections = results[0].boxes
#             circle_centers = []
#             for detection in detections:
#                 if detection.cls == 0:
#                     x1, y1, x2, y2 = map(int, detection.xyxy[0])
#                     center_x = (x1 + x2) // 2
#                     center_y = (y1 + y2) // 2
#                     circle_centers.append((center_x, center_y))
#             if len(circle_centers) == 3:
#                 best_frame = frame.copy()
#                 best_circle_centers = circle_centers
#         else:
#             break
#     if best_frame is not None and len(best_circle_centers) == 3:
#         draw_lines_between_circles(best_frame, best_circle_centers, output_dir)
#     cap.release()
#     cv2.destroyAllWindows()

# def main(file_path, output_dir):
#     if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
#         process_image(file_path, output_dir)
#     elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):
#         process_video(file_path, output_dir)
#     else:
#         print("Unsupported file format")

# # Replace with your image or video file path and desired output directory
# file_path = "vdo1.MP4"
# output_dir = "output"
# main(file_path, output_dir)

