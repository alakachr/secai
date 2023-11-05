from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(
    data="/data/ubuntu/secai/data/Prestamo_no_consensuado.v5i.yolov8/data.yaml",
    epochs=100,
    imgsz=640,
)
