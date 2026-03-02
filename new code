import streamlit as st
import cv2
import torch
import numpy as np
import time
from threading import Thread

# Video Stream Class for threaded capture
class VideoStream:
    def __init__(self, src=0, resolution=(640, 480)):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.stopped = False
        self.frame = None
        
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
        
    def update(self):
        while not self.stopped:
            (grabbed, frame) = self.stream.read()
            if not grabbed:
                self.stopped = True
                break
            self.frame = frame
            time.sleep(0.01)  # Small delay to reduce CPU usage
            
    def read(self):
        return self.frame
        
    def stop(self):
        self.stopped = True
        self.stream.release()

# Set page configuration
st.set_page_config(
    page_title="Smart Imaging Dashboard",
    page_icon="🎥",
    layout="wide"
)

# Load YOLOv5 model (Optimized for Jetson GPU)
@st.cache_resource
def load_model():
    # Check for CUDA to leverage Jetson Nano's GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, device=device)
    model.classes = [0]  # Filter for people only (class 0)
    return model

# Function to process frame with YOLOv5
def process_frame(frame, model, restricted_area, people_counter):
    if frame is None:
        return None, False
        
    # Convert frame to RGB for YOLOv5
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model(rgb_frame)
    
    # Get detections
    detections = results.pandas().xyxy[0]
    
    # Filter for people (class 0)
    people_detections = detections[detections['class'] == 0]
    
    # Count people
    people_count = len(people_detections)
    people_counter['count'] = people_count
    
    # Draw bounding boxes and check for restricted area intrusion
    intrusion_detected = False
    
    # Draw restricted area (Red box)
    cv2.rectangle(frame, (restricted_area[0], restricted_area[1]), 
                 (restricted_area[2], restricted_area[3]), (0, 0, 255), 2)
    
    for _, detection in people_detections.iterrows():
        # Get bounding box coordinates
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        conf = detection['confidence']
        label = f"Person: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Check if person is in restricted area
        person_center_x = (x1 + x2) // 2
        person_center_y = (y1 + y2) // 2
        
        if (restricted_area[0] < person_center_x < restricted_area[2] and 
            restricted_area[1] < person_center_y < restricted_area[3]):
            intrusion_detected = True
            # Highlight the intruding person with a red box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    # Add people count to frame
    cv2.putText(frame, f"People Count: {people_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame, intrusion_detected

def main():
    st.title("Smart Imaging Dashboard")
    
    # Sidebar configuration
    st.sidebar.title("Settings")
    
    # USB camera index selection
    usb_camera_index = st.sidebar.number_input(
        "USB Camera Index",
        min_value=0,
        max_value=10,
        value=0, # Usually 0 for the first connected USB webcam
        step=1,
        help="If 0 fails, try 1 or 2 depending on your Jetson's /dev/video* mappings."
    )
    
    # Define restricted area (default values)
    st.sidebar.subheader("Restricted Area Settings")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        x1 = st.sidebar.number_input("X1", value=100, min_value=0, max_value=1000)
        y1 = st.sidebar.number_input("Y1", value=100, min_value=0, max_value=1000)
    with col2:
        x2 = st.sidebar.number_input("X2", value=400, min_value=0, max_value=1000)
        y2 = st.sidebar.number_input("Y2", value=400, min_value=0, max_value=1000)
    
    restricted_area = [x1, y1, x2, y2]
    
    # Load model
    with st.spinner("Loading YOLOv5 model..."):
        model = load_model()
        st.sidebar.success("Model loaded successfully!")
    
    # Initialize people counter
    people_counter = {'count': 0}
    
    # Create placeholder for video feed
    video_placeholder = st.empty()
    
    # Create metrics placeholders
    col1, col2 = st.columns(2)
    people_count_metric = col1.metric("People Count", 0)
    status_metric = col2.metric("Status", "Safe")
    
    # Initialize camera using threaded video capture
    video_stream = VideoStream(src=usb_camera_index, resolution=(640, 480)).start()
    
    # Give the camera sensor time to warm up
    time.sleep(2.0)
    
    # Start streaming
    try:
        while True:
            # Get frame from threaded video stream
            frame = video_stream.read()
            
            if frame is None:
                st.error(f"Error: Failed to capture image from camera (index: {usb_camera_index}).")
                st.info("Check if your USB webcam is plugged in and recognized via 'ls -l /dev/video*'.")
                break
            
            # Process frame
            processed_frame, intrusion_detected = process_frame(frame, model, restricted_area, people_counter)
            
            # Update metrics
            people_count_metric.metric("People Count", people_counter['count'])
            
            # Update visual status metric for intrusions
            if intrusion_detected:
                status_metric.metric("Status", "⚠️ INTRUSION DETECTED", delta="- Alert")
            else:
                status_metric.metric("Status", "Safe", delta=None)
            
            # Convert to RGB for Streamlit display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame
            video_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)
            
            # Small delay to reduce CPU usage
            time.sleep(0.01)
            
    except Exception as e:
        st.error(f"Error during video processing: {e}")
        
    finally:
        # Release resources safely on exit
        video_stream.stop()

if __name__ == "__main__":
    main()
