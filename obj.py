import streamlit as st
import cv2
import torch
import torchvision
import torchvision.transforms as T
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

# 1. REPLACE YOUR load_model FUNCTION WITH THIS:
@st.cache(allow_output_mutation=True)
def load_model():
    # This version is compatible with Torchvision 0.9.1
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device

# 2. REPLACE YOUR process_frame FUNCTION WITH THIS:
def process_frame(frame, model, restricted_area, people_counter):
    if frame is None:
        return None, False
    
    # Prepare image
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = T.ToTensor()(img_rgb)
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
    
    # Run Inference
    with torch.no_grad():
        # SSD models expect a list of tensors
        predictions = model([img_tensor])
    
    # SSD returns: boxes, labels, scores
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()
    
    people_count = 0
    intrusion_detected = False

    # Draw restricted area
    cv2.rectangle(frame, (int(restricted_area[0]), int(restricted_area[1])), 
                 (int(restricted_area[2]), int(restricted_area[3])), (0, 0, 255), 2)

    for i in range(len(pred_labels)):
        # In COCO dataset (which this model uses), Class 1 is "Person"
        if pred_labels[i] == 1 and pred_scores[i] > 0.5:
            people_count += 1
            x1, y1, x2, y2 = pred_boxes[i].astype(int)
            
            # Draw person box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Check for intrusion
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if (restricted_area[0] < cx < restricted_area[2] and 
                restricted_area[1] < cy < restricted_area[3]):
                intrusion_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    people_counter['count'] = people_count
    return frame, intrusion_detected
    
    # Loop through detections
    for i in range(len(boxes)):
        # Check if it's a Person (COCO class 1) and confidence > 50%
        if labels[i] == 1 and scores[i] > 0.50:
            people_count += 1
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = boxes[i].astype(int)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"Person: {scores[i]:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Check if person is in restricted area
            person_center_x = (x1 + x2) // 2
            person_center_y = (y1 + y2) // 2
            
            if (restricted_area[0] < person_center_x < restricted_area[2] and 
                restricted_area[1] < person_center_y < restricted_area[3]):
                intrusion_detected = True
                # Highlight the intruding person with a red box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    people_counter['count'] = people_count
    
    # Add people count to frame
    cv2.putText(frame, f"People Count: {people_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame, intrusion_detected

def main():
    st.title("Smart Imaging Dashboard")
    
    st.sidebar.title("Settings")
    
    usb_camera_index = st.sidebar.number_input(
        "USB Camera Index", min_value=0, max_value=10, value=0, step=1
    )
    
    st.sidebar.subheader("Restricted Area Settings")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        x1 = st.sidebar.number_input("X1", value=100, min_value=0, max_value=1000)
        y1 = st.sidebar.number_input("Y1", value=100, min_value=0, max_value=1000)
    with col2:
        x2 = st.sidebar.number_input("X2", value=400, min_value=0, max_value=1000)
        y2 = st.sidebar.number_input("Y2", value=400, min_value=0, max_value=1000)
    
    restricted_area = [x1, y1, x2, y2]
    
    with st.spinner("Loading AI Model (No Ultralytics needed!)..."):
        model = load_model()
        st.sidebar.success("Model loaded successfully!")
    
    people_counter = {'count': 0}
    video_placeholder = st.empty()
    
    col1, col2 = st.columns(2)
    people_count_metric = col1.metric("People Count", 0)
    status_metric = col2.metric("Status", "Safe")
    
    video_stream = VideoStream(src=usb_camera_index, resolution=(640, 480)).start()
    time.sleep(2.0)
    
    try:
        while True:
            frame = video_stream.read()
            
            if frame is None:
                st.error(f"Camera Error on index {usb_camera_index}. Waiting...")
                time.sleep(1)
                continue
                
            processed_frame, intrusion_detected = process_frame(frame, model, restricted_area, people_counter)
            
            people_count_metric.metric("People Count", people_counter['count'])
            
            if intrusion_detected:
                status_metric.metric("Status", "⚠️ INTRUSION DETECTED", delta="- Alert")
            else:
                status_metric.metric("Status", "Safe", delta=None)
            
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)
            
            time.sleep(0.01)
            
    except Exception as e:
        st.error(f"Stream interrupted. Please refresh.")
        
    finally:
        video_stream.stop()

if __name__ == "__main__":
    main()
