from ultralytics import YOLO
model=  YOLO('runs/detect/ppe-2class-5/weights/last.pt')
model.train(data='Dataset/data6/data.yaml', epochs=100, batch=16, imgsz=640, name='ppe-2class',workers=0,patience=20)