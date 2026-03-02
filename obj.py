cat << 'EOF' > obj.py
import streamlit as st
import cv2
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import time
from threading import Thread

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
            time.sleep(0.01) 
            
    def read(self):
        return self.frame
        
    def stop(self):
        self.stopped = True
        self.stream.release()

st.set_page_config(page_title="Smart Imaging", layout="wide")

@st.cache(allow_output_mutation=True)
def load_model():
    # Using ResNet50. This is bulletproof for Torchvision 0.9.1.
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device

def process_frame(frame, model, device, restricted_area, people_counter):
    if frame is None:
        return None, False
        
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = T.ToTensor()(img_rgb).to(device)
    
    with torch.no_grad():
        predictions = model([img_tensor])
        
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    
    people_count = 0
    intrusion_detected = False
    
    cv2.rectangle(frame, (int(restricted_area[0]), int(restricted_area[1])), 
                 (int(restricted_area[2]), int(restricted_area[3])), (0, 0, 255), 2)
    
    for i in range(len(labels)):
        if labels[i] == 1 and scores[i] > 0.5:
            people_count += 1
            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if (restricted_area[0] < cx < restricted_area[2] and 
                restricted_area[1] < cy < restricted_area[3]):
                intrusion_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    people_counter['count'] = people_count
    return frame, intrusion_detected

def main():
    st.title("Smart Imaging Dashboard")
    cam_index = st.sidebar.number_input("Camera Index", value=0, step=1)
    
    x1 = st.sidebar.number_input("X1", value=100)
    y1 = st.sidebar.number_input("Y1", value=100)
    x2 = st.sidebar.number_input("X2", value=400)
    y2 = st.sidebar.number_input("Y2", value=400)
    restricted_area = [x1, y1, x2, y2]
    
    model, device = load_model()
    st.sidebar.success("Model Loaded Successfully!")
    
    video_placeholder = st.empty()
    col1, col2 = st.columns(2)
    p_metric = col1.metric("People Count", 0)
    s_metric = col2.metric("Status", "Safe")
    
    video_stream = VideoStream(src=cam_index).start()
    time.sleep(2.0)
    
    people_counter = {'count': 0}
    
    try:
        while True:
            frame = video_stream.read()
            if frame is None:
                continue
                
            processed_frame, intrusion = process_frame(frame, model, device, restricted_area, people_counter)
            
            p_metric.metric("People Count", people_counter['count'])
            if intrusion:
                s_metric.metric("Status", "⚠️ INTRUSION", delta="- ALERT", delta_color="inverse")
            else:
                s_metric.metric("Status", "Safe", delta=None)
            
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB")
            
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        video_stream.stop()

if __name__ == "__main__":
    main()
EOF
