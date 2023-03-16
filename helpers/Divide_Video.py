import cv2
import os

# Input Path
video_path = '/home/jawabreh/Desktop/HumaneX Project/Products/FR/FR_Dataset/Team_Videos_Scan/Masked/Milena/Milena_Masked.MOV'

# Output Path
output_dir = '/home/jawabreh/Desktop/HumaneX Project/Products/FR/FR_Dataset/Team_Scan_Dataset/Masked/Milena'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create a VideoCapture object to read the input video
cap = cv2.VideoCapture(video_path)

# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Calculate the frame interval to capture for 150 images
frame_interval = total_frames // 250 # change this number according to your needs 

# Set the initial frame counter to 0
frame_counter = 0

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Check if this is the frame to capture
    if frame_counter % frame_interval == 0 and frame_counter // frame_interval < 250:
        # Save the frame as a JPEG image
        output_path = os.path.join(output_dir, f'{frame_counter//frame_interval + 1:03}.jpg')
        cv2.imwrite(output_path, frame)
    
    # Increment the frame counter
    frame_counter += 1
    
    if frame_counter >= total_frames:
        break

# Release the video capture object
cap.release()
