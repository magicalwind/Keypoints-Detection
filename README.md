# Keypoints-Detection
Training a convolutional neural network to perform facial keypoint detection, for futher use of face tracking，facial expression analysis and etc.
The image data is Youtube Faces dataset (https://www.cs.tau.ac.il/~wolf/ytfaces/)

- ## Load and Visualize data.ipynb and data_load.py
Visualize image data extracted from youtube faces dataset. Use CV2 to resize, rotate and randomcrop the orignial image for a more robust training model. And, view some sample images and their corresponding key-points.

- ## models.py
Main structure of the models. CNN with Batch norm and dropout layer. 

- ## train.ipynb
Train the models. Visulize and compare the results before and after the training. 

- ## Facial Keypoints Detection Pipeline.ipynb
Select a random image, use cv2 haar cascade classifier to detect and crop the face region of images. Random padding the cropped image and implement pre-trained model to detect and visualize the key-points.
