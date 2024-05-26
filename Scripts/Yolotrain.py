from ultralytics import YOLO

model = YOLO('yolov8m-seg.pt')
results = model.train(data='train.yaml', epochs=125, imgsz=640, batch=8)