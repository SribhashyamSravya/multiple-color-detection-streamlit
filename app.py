import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.title("🎯 Real-Time Multiple Color Detection")
st.write("Detect Red, Green, Blue, Yellow with Center Tracking")

colors = {
    "Red": ([136, 87, 111], [180, 255, 255], (0, 0, 255)),
    "Green": ([25, 52, 72], [102, 255, 255], (0, 255, 0)),
    "Blue": ([94, 80, 2], [120, 255, 255], (255, 0, 0)),
    "Yellow": ([20, 100, 100], [30, 255, 255], (0, 255, 255))
}

kernel = np.ones((5, 5), "uint8")


class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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

                    cv2.rectangle(img,
                                  (x, y),
                                  (x + w, y + h),
                                  box_color, 2)

                    center_x = int(x + w / 2)
                    center_y = int(y + h / 2)

                    cv2.circle(img,
                               (center_x, center_y),
                               5,
                               (255, 255, 255),
                               -1)

                    cv2.putText(img,
                                f"{color_name}",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                box_color,
                                2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(key="example",
                video_processor_factory=VideoProcessor)