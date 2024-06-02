import torch
from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO('yolov9c.pt')

# Wrap the model with DataParallel to use multiple GPUs
model = torch.nn.DataParallel(model)

# Train the model
results = model.train(
    data='dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=32,  
    device=0,  
    cache=False,  
)

# Training loop with GPU cache cleaning
num_epochs = 50

for epoch in range(num_epochs):
    # Perform training for one epoch
    model.train_one_epoch()
    
    # Clean GPU cache
    torch.cuda.empty_cache()
    
    # Print memory usage to monitor the GPU memory
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} memory summary:")
        print(torch.cuda.memory_summary(device=i, abbreviated=False))

# Validate the model
model.validate()
