from ultralytics import YOLO


model = YOLO("yolov8m.pt")  # Download pre triained checkpint from Ultralytics

# Train the model with 2 GPUs
results = model.train(
    data="/data/ubuntu/secai/src/sec_ai/detection/data.yaml",
    epochs=100,
    imgsz=640,
)
