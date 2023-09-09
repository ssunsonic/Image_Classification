# ECS171

Chihuahua vs Muffins


## Preprocessing Methodology
After conducting basic EDA on our image data, we plan to first crop/resize images that include irrelevant background noise. For example, a few of the chihuahua images contain objects unrelated to the face, which is the main feature we should be extracting and analyzing. Resizing the images to 640x480 also allows the model to run faster on smaller, cropped images. We also considered image rotation if necessary. Once the images contain the main features based on our shrewd discretion, we plan on greyscaling for greater feature detection (i.e. edge/corner detection contrast between classes).  


## Post-Preprocessing and Modeling Draft
For image data, we first split the images into training and validation sets. We stayed true to image removal, resizing (adjusted to 50 X 50), and grayscaling for the preprocessing section. After this, we constructed a Keras Sequential Model with 3 Convolutional Layers, with ReLU activation functions for the Convolutions and a sigmoid at the end for binary classification. Our model did better than we initially expected, however it showed later signs of overfitting. Out of 10 epochs, the model started overfitting around 6 epochs.


** note, added the copy of image processing final model to add code so we can see the layers of each image as it goes through the model
