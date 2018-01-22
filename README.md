# Image-Categorization

Image Categorization on Stanford Dog Dataset using tensor flow and Keras


## Dataset 

To build our image classification model, we used Stanford Dog dataset which contains 20580 images of
120 different dog breeds where each image is annotated with a object class label. The dataset is extremely
challenging due to a variety of reasons. First, being a fine-grained image categorization problem, there is 
little inter-class variation.For example the Siberian Husky and Alaskan Malamute share very similar 
characteristics. Second, there is a very large intra-class variation i.e. images within a class have 
different ages, colors and poses. Third, variations in background of the images.



## Data Processing

The first step as part of pre-processing was to re-size the images to 64*64 pixels. This is to have uniformity in the
size of images and also considering the computational restrictions. The next step was to add a little bit of vari-
ance in our data since the images in the dataset were very organized and contained no noise. As we want our
model to classify images with noise we artificially added noise by a random combination of zoom and horizontal
flip. The dataset was then split into train and test. As this is a classification task we took into consideration the
proportions of our target classes and made sure the train set has equal number of images (100) from each class 
resulting in 12000 images in train set and the remaining 8580 images in the test set.



## CNN Architecture

A CNN has three layers. The first is a Convolutional layer where a sliding window (filter) is used to extract
features from the image. Considering the non-linear nature of real world images we introduced non-linearity
into our model by passing a Relu function into the convolutional layer. The second layer is a Pooling layer, that
reduces the dimensions, which makes the model efficient and prevents over-fitting. We used max pooling as it
has shown to work better in practice. The final layer is a Fully Connected layer that uses a softmax activation
function in the output layer. The purpose of the Fully Connected layer is to use the features extracted by the
previous layers to classify the input image into various classes based on the training dataset.

We have taken the pre-trained InceptionV3 model excluding the top layer, and fine-tuned it on our set of
classes (120) by adding a ’Relu’,’Softmax’ layer on top of it. In order to input the images into a InceptionV3
model we have converted the test images to 128*128 pixels. Once the images are converted and model weights
are freezed, we used the model to predict the accuracy on test dataset.


