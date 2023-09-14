# ECS171 - Machine Learning Final Project

Chihuahua vs Muffins

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
![alt text]()

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

---

## Preprocessing Methodology
After conducting basic EDA on our image data, we plan to first crop/resize images that include irrelevant background noise. For example, a few of the chihuahua images contain objects unrelated to the face, which is the main feature we should be extracting and analyzing. Resizing the images to 640x480 also allows the model to run faster on smaller, cropped images. We also considered image rotation if necessary. Once the images contain the main features based on our shrewd discretion, we plan on greyscaling for greater feature detection (i.e. edge/corner detection contrast between classes).  


## Post-Preprocessing and Modeling Draft
For image data, we first split the images into training and validation sets. We stayed true to image removal, resizing (adjusted to 50 X 50), and grayscaling for the preprocessing section. After this, we constructed a Keras Sequential Model with 3 Convolutional Layers, with ReLU activation functions for the Convolutions and a sigmoid at the end for binary classification. Our model did better than we initially expected, however it showed later signs of overfitting. Out of 10 epochs, the model started overfitting around 6 epochs.


** note, added the copy of image processing final model to add code so we can see the layers of each image as it goes through the model
