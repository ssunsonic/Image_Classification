# ECS171

Chihuahua vs Muffins


## Preprocessing Methodology
After conducting basic EDA on our image data, we plan to first crop/resize images that include irrelevant background noise. For example, a few of the chihuahua images contain objects unrelated to the face, which is the main feature we should be extracting and analyzing. Resizing the images also allows the model to run faster on smaller, cropped images. We all considered image rotation if necessary. Once the images contain the main features based on our shrewd discretion, we plan on greyscaling for greater feature detection (i.e. edge/corner detection contrast between classes).  
