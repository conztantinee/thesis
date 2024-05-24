import os
import glob
import random

def generate_and_shuffle_train_file(image_directory, output_file):
    image_paths = glob.glob(os.path.join(image_directory, '**', '*.jpg'), recursive=True)

    random.shuffle(image_paths)

    with open(output_file, 'w') as f:
        for path in image_paths:
            f.write(f"{path}\n")

if __name__ == "__main__":
    image_directory = 'train' 
    output_file = 'train.txt'  

    generate_and_shuffle_train_file(image_directory, output_file)
