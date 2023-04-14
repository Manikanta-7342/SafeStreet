# TrafficSign

One of the applications where deep learning is being used extensively is Autonomous Driving. The autonous driving mechanism needs to detect innumerable objects (pedestrians, other cars, obstacles etc) and make decisions. One of the smaller problems is to detect traffic signs and make decisions accordingly. I'm going to solve problems of detecting traffic signs on the road.

The training dataset contains around 39,000 images while test dataset contains around 12,000 images containing 43 different classes. I will be using Convolutional Neural Networks(CNN) to solve this problem using Keras framework and tensorflow as backend.

# Preprocessing Images

We need preprocessing for two reasons:

* To normalize intensity across all images i.e if an image is overexposed or underexposed, we will make it well-exposed using histogram equilization. As you can see in above pictures, we have many such images.
* To resize all images to same size.

# Building Convolutional Neural Network Model

We will now build our sequential CNN model with following specifications:

* 6 convolutional layer followed by one hidden layer and one output layer(fully connected or dense layer).
* Dropout layers for regularization to avoid overfitting
* Relu activation function for all convolutional layers
* Softmax activation function for output layer as it is a multi-class classification problem
* Flatten layer for reshaping the output of the convolutional layer

# Intel OneAPI Edge

On average, it took approximately 50 minutes to an hour to train the model on our local systems. However, certain systems encountered issues and were unable to execute the process at all. To address this issue, I decided to migrate our code to the Intel DevCloud and leverage the capabilities of the Intel Analytics toolkit, which includes tools such as oneDNN and oneDAL. By utilizing these tools, we were able to optimize our model training process and achieve better results.

The optimization of (oneDNN) and (oneDAL) in __TensorFlow 11.0 and sklearnex versions__ respectively played a significant role in achieving the required training output in just 35 minutes, making it the most notable tool among all. This optimization significantly boosted the speed of our training process. We are also exploring the possibility of further enhancing the speed by using the openvino toolkit, which has the potential to reduce the training time to just 20 minutes.

# Conclusion

I created a Convolutional Neural Network (CNN) model to classify traffic sign images. I started with exploring our dataset of German traffic signs. Then I performed pre-processing of images (Histogram equalization and rescaling to same size) to make them suitable for CNN. We built a simple CNN model using __tensorflow 11.0 version which is optimised with oneDNN library__ with 6 convolutional layer followed by one hidden layer, one output layer(fully connected or dense layer). I used dropout layers to avoid overfitting. After that I trained our model with our training dataset. The evaluation of model resulted in __97.3% accuracy__. I used data augmentation techniques to further improve accuracy to __98.4%__ and, __The training time required by the PC was almost twice as long as the time taken by the oneAPI toolkit in DevCloud__. The human accuracy for this dataset is 98.84%. Pretty Close!
