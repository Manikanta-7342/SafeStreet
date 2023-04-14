# TrafficSign

One of the applications where deep learning is being used extensively is Autonomous Driving. The autonous driving mechanism needs to detect innumerable objects (pedestrians, other cars, obstacles etc) and make decisions. One of the smaller problems is to detect traffic signs and make decisions accordingly. We are going to solve problems of detecting traffic signs on the road. We are going to use The German Traffic Sign Recognition Benchmark(GTSRB) dataset.

The training dataset contains around 39,000 images while test dataset contains around 12,000 images containing 43 different classes. We will be using Convolutional Neural Networks(CNN) to solve this problem using Keras framework and tensorflow as backend.

#Preprocessing Images

Now, we will use preprocess our images. We need preprocessing for two reasons:

*To normalize intensity across all images i.e if an image is overexposed or underexposed, we will make it well-exposed using histogram equilization. As you can see in above pictures, we have many such images.
*To resize all images to same size.

#Building Convolutional Neural Network Model

We will now build our sequential CNN model with following specifications:

*6 convolutional layer followed by one hidden layer and one output layer(fully connected or dense layer).
*Dropout layers for regularization to avoid overfitting
*Relu activation function for all convolutional layers
*Softmax activation function for output layer as it is a multi-class classification problem
*Flatten layer for reshaping the output of the convolutional layer

#Conclusion

We created a Convolutional Neural Network (CNN) model to classify traffic sign images. We started with exploring our dataset of German traffic signs. Then we performed pre-processing of images (Histogram equalization and rescaling to same size) to make them suitable for CNN. We built a simple CNN model using Keras with 6 convolutional layer followed by one hidden layer, one output layer(fully connected or dense layer). We used dropout layers to avoid overfitting. After that we trained our model with our training dataset. The evaluation of model resulted in 97.3% accuracy. We used data augmentation techniques to further improve accuracy to 98.4%. The human accuracy for this dataset is 98.84%. Pretty Close!
