from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO('yolov9c.pt')

# Train the model
results = model.train(
    data='dataset.yaml',
    epochs=250,
    imgsz=640,
    batch=16,
    device=0,
    cache=False  # cache=False is important!
)
