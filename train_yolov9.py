import torch
from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='data.yaml',
    epochs=250,
    imgsz=640,
    batch=8,  # Set a smaller batch size
    device=1,
    cache=False,  # cache=False is important!
    amp=True  # Enable Automatic Mixed Precision
)

# Training loop with GPU cache cleaning
num_epochs = 250

for epoch in range(num_epochs):
    # Perform training for one epoch
    model.train_one_epoch()
    
    # Clean GPU cache
    torch.cuda.empty_cache()
    
    # (Optional) Print memory usage to monitor the GPU memory
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1024**2} MB")
    print(f"Cached GPU memory: {torch.cuda.memory_reserved() / 1024**2} MB")

# Validate the model
model.validate()
