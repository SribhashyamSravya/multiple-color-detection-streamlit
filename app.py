# ==========================================================
# STREAMLIT WEB APP - MULTIPLE COLOR DETECTION
# ==========================================================

import streamlit as st
import cv2
import numpy as np

# ----------------------------------------------------------
# Step 1: App Title and Description
# ----------------------------------------------------------
st.title("🎯 Real-Time Multiple Color Detection")
st.write("Detect Red, Green, Blue, Yellow with Center Tracking")

# Checkbox to Start / Stop Camera
run = st.checkbox('Start Camera')

# Placeholder for video frames
FRAME_WINDOW = st.image([])

# Open webcam
camera = cv2.VideoCapture(0)

# ----------------------------------------------------------
# Step 2: Define Color Ranges
# ----------------------------------------------------------
colors = {
    "Red": ([136, 87, 111], [180, 255, 255], (0, 0, 255)),
    "Green": ([25, 52, 72], [102, 255, 255], (0, 255, 0)),
    "Blue": ([94, 80, 2], [120, 255, 255], (255, 0, 0)),
    "Yellow": ([20, 100, 100], [30, 255, 255], (0, 255, 255))
}

# Kernel for noise removal
kernel = np.ones((5, 5), "uint8")

# ----------------------------------------------------------
# Step 3: Real-Time Processing Loop
# ----------------------------------------------------------
while run:

    success, frame = camera.read()

    if not success:
        break

    # Convert to HSV
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Process each color
    for color_name, (lower, upper, box_color) in colors.items():

        lower = np.array(lower, np.uint8)
        upper = np.array(upper, np.uint8)

        mask = cv2.inRange(hsvFrame, lower, upper)
        mask = cv2.dilate(mask, kernel)

        contours, _ = cv2.findContours(mask,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:

            area = cv2.contourArea(contour)

            if area > 500:

                x, y, w, h = cv2.boundingRect(contour)

                # Draw bounding box
                cv2.rectangle(frame,
                              (x, y),
                              (x + w, y + h),
                              box_color,
                              2)

                # Calculate center
                center_x = int(x + w / 2)
                center_y = int(y + h / 2)

                # Draw center point
                cv2.circle(frame,
                           (center_x, center_y),
                           5,
                           (255, 255, 255),
                           -1)

                # Display color name
                cv2.putText(frame,
                            f"{color_name}",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            box_color,
                            2)

    # Convert BGR to RGB (Streamlit requires RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display frame in Streamlit
    FRAME_WINDOW.image(frame)

# ----------------------------------------------------------
# Step 4: Release Camera When Stopped
# ----------------------------------------------------------
camera.release()
st.write("Camera Stopped")