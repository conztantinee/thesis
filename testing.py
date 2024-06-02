from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import cv2

# Loading the trained model
model = YOLO('/Users/kamal/Desktop/thesis/thesis/runs/detect/train4/weights/best.pt')

# Directory containing test images
test_images_dir = '/Users/kamal/Desktop/thesis/thesis/test/images'

# Listing all image files in the test directory
test_images = [os.path.join(test_images_dir, img) for img in os.listdir(test_images_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]

# Inference on Test Set
results = []
for img_path in test_images:
    result = model(img_path)
    results.append(result)

# Evaluation

metrics = model.val(data='/Users/kamal/Desktop/thesis/thesis/dataset.yaml')  # YAML file containing test dataset configuration

# Retrieve evaluation metrics from results_dict
metrics_dict = metrics.results_dict

# Print evaluation metrics
print("Evaluation Metrics:")
print(f"Precision: {metrics_dict['metrics/precision(B)']:.4f}")
print(f"Recall: {metrics_dict['metrics/recall(B)']:.4f}")
print(f"mAP@0.5: {metrics_dict['metrics/mAP50(B)']:.4f}")
print(f"mAP@0.5:0.95: {metrics_dict['metrics/mAP50-95(B)']:.4f}")

# Visualize some results
for img_path, result in zip(test_images[:5], results):  # Visualize the first 5 test images
    img = cv2.imread(img_path)
    for detection in result:
        for box in detection.boxes:
            # Extracting box coordinates, class label, and confidence
            xyxy = box.xyxy[0].cpu().numpy().astype(int)  # Ensure we get a flat array
            cls = int(box.cls[0].cpu().numpy())  # Convert to integer
            conf = float(box.conf[0].cpu().numpy())  # Convert to float
            # Draw bounding boxes on the image
            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(img, f'{cls} {conf:.2f}', (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
