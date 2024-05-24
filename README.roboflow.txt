
License Plates - v1 2024-05-09 4:55am
==============================

This dataset was exported via roboflow.com on May 9, 2024 at 2:56 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 595 images.
Plates are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 2 versions of each source image:
* 50% probability of horizontal flip
* Random shear of between -15° to +15° horizontally and -15° to +15° vertically
* Random exposure adjustment of between -14 and +14 percent
* Random Gaussian blur of between 0 and 2 pixels


