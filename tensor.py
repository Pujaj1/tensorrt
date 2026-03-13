import depthai as dai
import cv2
import torch
import time
import numpy as np

names_path = '/home/puja/yolov5/runs/train/exp2/weights/coco.names'
with open(names_path, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]
    
pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.Camera).build()
cam_out = cam.requestOutput(size=(640, 640), type=dai.ImgFrame.Type.BGR888i)
q_rgb = cam_out.createOutputQueue(maxSize=4, blocking=False)

engine_path = '/home/puja/depthai-core/examples/python/DetectionNetwork/yolov5.engine' 
model = torch.hub.load('ultralytics/yolov5', 'custom', path=engine_path)
model.classes = None  

pipeline.start()

print("Streaming... Press 'q' to exit.")
while pipeline.isRunning():
    start_time = time.time()
    frame_msg = q_rgb.get()
    if frame_msg is None:
        continue
    frame = frame_msg.getCvFrame()  
    results = model(frame, size=640)

    for detection in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2 = map(int, detection[:4])
        conf = detection[4]
        cls  = int(detection[5])
        label_text = class_names[cls] if cls < len(class_names) else f"ID {cls}"
        label = f"{label_text} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
    cv2.imshow("OAK YOLOv5 Segmentation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
pipeline.stop()
