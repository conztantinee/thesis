import cv2
import torch
import numpy
import pytesseract
from PIL import Image
from ultralytics import YOLO
import os

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

model = YOLO('runs/detect/train5/weights/best.pt')

image_folder_path = 'test/images'


output_file_path = 'output.TXT'

# Function to detect license plates
def detect_license_plate(image):
    results = model(image)
    return process_results(results)

def process_results(results):
    all_bboxes = []

    
    for result in results:
        
        if hasattr(result, 'xyxy'):
            bboxes = result.xyxy[0].numpy()  
            all_bboxes.extend(bboxes)  
    
    return all_bboxes

# Function to extract text from license plates
def extract_text_from_plate(plate_image):
    gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    _, binary_plate = cv2.threshold(gray_plate, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(binary_plate, config='--psm 6')
    return text.strip()

# list to store detected license plate numbers
detected_plates = []

# processing images
for image_file in os.listdir(image_folder_path):
    if image_file.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder_path, image_file)
        img = Image.open(image_path)
        
        # license plates
        bboxes = detect_license_plate(img)
        if bboxes is None:
            print(f"No bounding boxes found in image: {image_file}")
            continue
               
        car_image = cv2.imread(image_path)
           
        # Loop
        for bbox in bboxes:
            print("Bounding box coordinates:", bbox[:6])
            x1, y1, x2, y2, conf, cls = map(int, bbox[:6])  
            if x1 < x2 and y1 < y2:  
                plate_img = car_image[y1:y2, x1:x2]  
                plate_number = extract_text_from_plate(plate_img)  
                if plate_number:
                    detected_plates.append(plate_number)
                    print(f"Detected License Plate Number: {plate_number}")
                else:
                    print("OCR failed to detect text.")
            

# saving results
with open(output_file_path, 'w') as f:
    f.write(','.join(detected_plates))

#print(f"Detected license plate numbers have been saved")
    