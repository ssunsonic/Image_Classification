# ECS171 - Machine Learning Final Project

Chihuahua vs Muffins

![alt text](https://raw.githubusercontent.com/ssunsonic/ML_Project/2dd8ae4f30be02ea6600b77dd9026678903eab39/figures/Kaggle%20Chihuahuas%20vs%20Muffins.jpg)

([original source](https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification))

## Introduction
There is a growing prevalence in machine learning models for everyday utilization in part due to its associated convenience of automation, efficiency, and identification.  In the case of identification, machine learning models are expected to correctly discern between specified objects of interest; this could potentially be the difference between being able to identify a patient's diagnosis in time for live-saving treatment. However, for these benefits to be truly realized, it is necessary to ensure that these models can first achieve the bare minimum. For the purposes of this project, we use Kaggle's Chihuahua vs Muffins dataset to observe how well these models perform in a low-stakes environment such as differentiating images of chihuahuas and muffins. The dataset contains 6,000 different images taken from Google Images, which contain almost equally split cases and no duplicates. We chose this dataset due to its simplicity and the amusing nature of examining supposed similarities between chihuahuas and muffins. If a machine learning model cannot execute simple tasks, how can we expect to apply them to more complex scenarios? Our research utilizes a two convolutional neural networks (CNN) and k-nearest neighbors (KNN) approach to be compared against each other. The two CNNs differ in the amount of convolutional layers and nodes included, however only one model is used for the final comparison due to complications with overfitting. Despite potential overlap in features among the images, the models should be able to recognize unique features between the two categories and subsequently identify them correctly. Training the different models should allow us to observe how machine learning models may be used to complete efficient, automated identification tasks and if there are any differences between the type of model being used.

## Methods
### Data Exploration
To explore our data, we chose to do a manual analysis of the dataset, carefully looking at each image as well as the set as a whole.

### Preprocessing
When preprocessing our data, we focused on Data Transformation and Reduction methods.

### Model 1
For the first model in our project, we built a CNN with 3 convolutional layers to extract the features of each image. Each convolutional layer uses ```relu``` as its activation function. We also made sure to include a hidden layer with 64 nodes, and because we are predicting binary outcomes, the output layer uses a ```sigmoid``` activation function.

Implementation of Model 1:
```
model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
```
### Model 2
Similar to Model 1, Model 2 is a CNN instead with 2 convolution layers. Both layers use ```relu```, however, the hidden layer in Model two has 32 nodes. Again, the outcome layer utilizes a ```sigmoid``` activation function due to how we are predicting binary outcomes.

Implementation of Model 2:
```
model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
```
### Model 3
The third and final model uses k-nearest neighbors (KNN) as an alternative approach. Despite trying to use 3, 4, and 5 neighbors for the model, the accuracy remained the same.

Implementation of Model 3:
```
model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
```

## Results
### Data Exploration
When we initially looked into the dataset, there were three main observations we made regarding extraneous variables, variance between the amount of images per category, and assorted image dimensions. Firstly, there were several images that did not fall into either of the two class categories: muffin or chihuahua. Additionally, we noticed that there are 2199 muffin pictures while there are 3718 chihuahua pictures. Our last observation was the various sizes of the images found in the dataset. The figure below depicts the distribution of image sizes across the dataset prior to standardization. 

![alt text](https://github.com/ssunsonic/ML_Project/blob/main/figures/Figure%201.png)

### Preprocessing
Our preprocessing consists of three steps: (1) data splitting, (2) data transformation, and (3) data reduction. Before preprocessing the images themselves, we did a 80:20 split on the data into a training and testing set. Then to transform the data, we standardized the image dimensions across both sets to 50x50. As for data reduction, we opted to convert the images to greyscale. All of these steps were completed as shown in the following code block:

```
# splitting the data
ds_train = tf.keras.utils.image_dataset_from_directory(destiny,
            validation_split=0.2, subset = 'training', seed = 1)
ds_val = tf.keras.utils.image_dataset_from_directory(destiny,
            validation_split=0.2, subset = 'validation', seed = 1)

# size of images we want to resize to
size = (50,50)

# resizing all images
ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
ds_val = ds_val.map(lambda image, label: (tf.image.resize(image, size), label))

# greyscaling all images
ds_train = ds_train.map(lambda image, label: (tf.image.rgb_to_grayscale(image), label))
ds_val = ds_val.map(lambda image, label: (tf.image.rgb_to_grayscale(image), label))
```

### Model 1

### Model 2

### Model 3

![alt text](https://github.com/ssunsonic/ML_Project/blob/main/figures/KNN%20Classification%20Report.PNG?raw=true)

## Discussion
### Data Exploration
We chose to perform a manual data exploration analysis to gain a better understanding of the data for ourselves. Furthermore, we felt that it would've taken much longer to implement an automated process for certain things such as the removal of irrelevant images from the dataset. Although we started with 6000 images from the original dataset, we ended up with 4320. We admit that this was a lot of images to remove which could subsequently affect the model, but we felt that it was necessary to remove the images as they were, afterall, irrelevant to our objective.

After removing several extraneous variables, we wanted to go through the data once more to ensure there was not a class imbalance. While we found a discrepancy of 481 images between the two (with there being more images of chihuahuas than muffins), we ultimately felt that this imbalance was not large enough for a need to compensate for.

Lastly, we wanted to check the image sizes. As shown in the figure in the Results section, we did find a variety of image sizes, so we chose to reduce the size of all images to maintain consistency while also reducing computation cost. This is further discussed in the following preprocessing stage.

### Preprocessing
**Data Split**: Unfortunately, we found it difficult to split the dataset into training and testing after preprocessing the images, so we ended up splitting everything first. We acknowledge that this is generally bad practice and would've looked more into splitting the dataset if we had more time.

**Data Transformation**: Because convolutional neural networks require that all images used for training be of the same size, we decided to standardize this. Thus, we chose to resize the images to 50x50 which would also minimize the runtime for our computers while still preserving the quality of the images. Regardless, we do recognize that other sizes of images would be better for model accuracy, however, we felt limited by both our personal computer and Google Colab resources. 

**Data Reduction**: By converting the images to greyscale, we were able to simplify having three color channels into one. As with standardizing the data, converting the images also helped to limit the overall computational cost. We felt that preserving the shapes within the images were more important than the colors in the images. Greyscaling moreover helps with edge detection as the differences in color intensity is more drastic between pixels.

**Additional Preprocessing**: In our original preprocessing methodology, we had considered applying more techniques for image augmentation such as image rotation. While we did not have enough time to do so, we had originally considered its implementation as image rotation would help diversify the training set. Rotating the image does not change the information on it, and for humans it is easy to tell they are the same photo. However, the algorithm does not perceive this the same way humans do and will consider it a separate case.   

### Model 1

### Model 2

### Model 3

## Conclusion
As the upcoming generation in an age of emergent technologies, things such as machine learning models are becoming increasingly important to be aware of. There are still many things unknown about the capabilities of machine learning, but all of that starts from a base case. Through our project, we were able to personally observe how to implement machine learning models to do something as simple as discerning between a chihuahua or a muffin. We were able to experience how difficult certain implementations were, how different models behaved, and how important each preceding step was in the process. As mentioned in our discussion section, there were definitely aspects that could have been explored more if we had access to more time. We had underestimated how long it would take to implement certain features. For example, we were unable to implement image rotation which may have affected the training data and subsequently the overall results. We were also unable to print out classification reports for the CNNs.

Further directions of this project could be to go back and figure out ways to achieve the things we couldn't end up doing. Additionally, we could expand the model to take in images of other dogs or baked goods. 

## Colaboration
**Allison Peng** (Code/Organizer): Facilitated group meetings and contributed to exploratory data analysis. Assisted in coding of preprocessing and a portion of the writeup of the project. 

**Danish Rizvi** (Code/Nice Guy): Cleaned up data and preprocessed. Showcased model capability by displaying images and using the CNN to label them with classifications. 

**Eric Sun** (Code/Formatter): Contributed to constructing CNN model architecture and development/testing.  Assisted in coding preprocessing techniques, EDA, and model accuracy. Formatted code in Google Colab and organized descriptions/file structure in GitHub for an easy to read, followable manner. 

**Franklin Hong** (Code):

**Mariel Cecilio** (Writer): Wrote and formatted the entire final write up, taking into account and revising group input.

---

## Preprocessing Methodology
After conducting basic EDA on our image data, we plan to first crop/resize images that include irrelevant background noise. For example, a few of the chihuahua images contain objects unrelated to the face, which is the main feature we should be extracting and analyzing. Resizing the images to 640x480 also allows the model to run faster on smaller, cropped images. We also considered image rotation if necessary. Once the images contain the main features based on our shrewd discretion, we plan on greyscaling for greater feature detection (i.e. edge/corner detection contrast between classes).  


## Post-Preprocessing and Modeling Draft
For image data, we first split the images into training and validation sets. We stayed true to image removal, resizing (adjusted to 50 X 50), and grayscaling for the preprocessing section. After this, we constructed a Keras Sequential Model with 3 Convolutional Layers, with ReLU activation functions for the Convolutions and a sigmoid at the end for binary classification. Our model did better than we initially expected, however it showed later signs of overfitting. Out of 10 epochs, the model started overfitting around 6 epochs.


** note, added the copy of image processing final model to add code so we can see the layers of each image as it goes through the model
