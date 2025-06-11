from ultralytics import YOLO

model = YOLO('models/best.pt')  # Load a pretrained YOLOv8 model

results = model.predict('sumber_videos/Base1.mp4',save=True)
print(results[0])
print('=====================')
if results[0].boxes is not None:
    for box in results[0].boxes:
        print(box)
else:
    print("No detections were made.")