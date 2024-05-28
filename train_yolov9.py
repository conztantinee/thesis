from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO('yolov9c.pt')

# Train the model
results = model.train(
    data='data.yaml',
    epochs=250,
    imgsz=640,
    batch=-1,
    device=0,
    cache=False  # cache=False is important!
)
