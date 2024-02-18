import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read video feed
video_capture = cv.VideoCapture("assets/inputvideo.mp4")

# Get the first frame in the entire video sequence
ret, first_frame = video_capture.read()

# Converts frame to grayscale for detecting edges
prev_gray_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

# Create image filled with zero intensities with the same dimensions as frame
mask = np.zeros_like(first_frame)

# Set image saturate to maximum
mask[..., 1] = 255

avg_motion_array = []
avg_angle_array = []

while video_capture.isOpened():

    # Get the current frame being projected in the video
    ret, frame = video_capture.read()

    if ret == 0:
        break

    # Open new window and display input frame
    cv.imshow("input", frame)

    # Convert the current frame to grayscale
    curr_gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Get dense optical flow by Farneback method
    flow = cv.calcOpticalFlowFarneback(prev_gray_frame, curr_gray_frame,
                                       None,
                                       0.5, 3, 15, 3, 15, 1.2, 0)
    
    # Compute the magnitude and angle (direction)
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

    # Get the average motion between the previous and current frame
    avg_motion_vector = np.mean(magnitude)
    avg_angle_vector = np.mean(angle)

    print("Magnitude:", avg_motion_vector)
    print("Angle:", avg_angle_vector)

    avg_motion_array.append(avg_motion_vector)
    avg_angle_array.append(avg_angle_vector)

    print("Magnitude Array:", avg_motion_array)
    print("Angle Array:", avg_angle_array)

    # Set image hue to optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2

    # Set image value to optical flow magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

    # Converts HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

    # Opens a new window and displays the output frame 
    cv.imshow("Dense Optical Flow", rgb) 

    # Updates previous frame
    prev_gray_frame = curr_gray_frame

    if cv.waitKey(1) & 0xFF == ord('q'): 
        break

plt.plot(avg_motion_array, avg_angle_array)
plt.show()

video_capture.release() 
cv.destroyAllWindows() 
