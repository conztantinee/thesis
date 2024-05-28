import torch
from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO('yolov9c.pt')

# Train the model
results = model.train(
    data='dataset.yaml',
    epochs=250,
    imgsz=640,
    batch=-1,
    device=0,
    cache=False  # cache=False is important!
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
