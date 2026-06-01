from ultralytics import YOLO
import time
model = YOLO('yolo11x-pose.pt')
start_time=time.time()
results = model.track(source="video/record_2026-05-30_13-55-47.avi", conf=0.35,show=False, save=True,device=0,verbose=True)
print(time.time() - start_time)