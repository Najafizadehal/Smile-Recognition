# Smile-Recognition
This code is a simple classifier model designed to detect smiles in images. The Local Binary Pattern (LBP) algorithm is used to extract features from the images, and then a simple neural network is used to classify the images based on these extracted features. The following steps describe how the code works:

First, the necessary libraries for running the code are imported.
Then, a function is defined using the MTCNN library to detect faces in the images.
Another function is defined for preprocessing the images. In this function, the LBP algorithm is used to extract features from the images. The images are then resized and flattened, and divided by 255 to make them usable as input for the neural network.
In the classification_model function, a simple neural network is defined for classifying the images. This neural network consists of 3 layers, using relu and softmax activation functions.
Finally, the preprocessing and classification_model functions are used to preprocess and classify the training and testing data, and the resulting model is saved with the name "smile_classifier.h5".
