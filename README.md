# DA6401_DL_Assignment2_Part_A
# Overview :
This project focuses on building and experimenting with a Convolutional Neural Network (CNN) for image classification using a subset of the iNaturalist dataset. As part of the assignment, the goal is to design a lightweight yet modular CNN model tailored for a 10-class classification task.

## Model Architecture :
The model is a Convolutional Neural Network (CNN) with the following structure:

* Convolutional Layers: 5 layers with different kernel sizes and number of filters.

* Batch Normalization: Used after each convolutional and dense layer to improve training.

* Activation Functions: ReLU, GELU, SiLU, or Mish can be used after each convolution layer.

* Max Pooling: Applied after every activation to reduce image dimensions.

* Dropout: Added in both convolutional and dense layers to reduce overfitting.

* Dense Layer: One fully connected layer with either 128 or 256 neurons.

* Output Layer: Final layer with 10 neurons, one for each class in the iNaturalist dataset.

# Hyperparameters :
* Number of filters in each layer : [32,32,32,32,32],[64,64,64,64,64],[32,64,128,256,512]
* Activation function for the Convolutional layers: ReLU, GELU, SiLU, Mish
* Dropout: 0.2, 0.3
* Data augmentation: Yes, No
* Batch normalisation: Yes, No
* Number of nodes in dense layer : [128, 256]
* Learning_rate: 0.001,0.0005
# Training Process :
I split the training data into 80:20 for training and validation.
I use Random sweep configuration for hyperparameter tuning.Then I selected best hyperparameter configuration based on validation accuracy.
# Best Hyperparameters :
After tuning the hyperparameters through a combination of random and Bayesian optimization sweeps, the best-performing model configuration was selected based on validation accuracy. The final model was then retrained using this optimal setup.

* Optimizer: Adam

* Learning Rate: 0.0001

* Batch Size: 64

* Activation Function: Mish

* Dropout: 0.2

* Batch Normalization: Enabled

* Number of Epochs: 10

* Filter Configuration: [32, 64, 128, 256, 512]

* Fully Connected Layer Size: 256

Using this configuration, the model achieved the following performance on the unseen test set:

Test Accuracy: 34.17 %

Test Loss: 1.8902

# Model Evaluation :
I evaluated the best-performing model—based on validation accuracy—on the test dataset to assess its generalization performance. The final test accuracy achieved by this model was 34.17%.


# DA6401_DL_Assignment2_Part_B : Fine-Tuning a Pre-Trained Model (ResNet50)
# Overview :
In this section, I fine-tuned a pre-trained ResNet50 model using the iNaturalist dataset. The model was initialized with weights learned from ImageNet and then adapted to suit the specific number of categories present in the iNaturalist dataset.

Training a deep neural network from scratch on a large dataset can be very resource-intensive, as it involves learning millions of parameters. Fine-tuning a model that has already been trained on a similar task helps speed up the process and often improves performance by leveraging learned features.

# Fine-Tuning Strategies :
My strategies are :
* Freezing all layers except the final classification layer.
* Freezing the initial k layers and fine-tuning the remaining ones.
* Freezing the fully connected layers while training the convolutional layers.

# Results :
As part of the model fine-tuning process, I investigated the impact of freezing different proportions of the ResNet50 network on validation accuracy. The results are summarized below:

* When 30% of the layers were frozen and the remaining layers fine-tuned, the model achieved the highest validation accuracy of 87.6%.

* Freezing 50% of the layers resulted in a slight drop, with a validation accuracy of 86.85%.

* With 80% of the layers frozen, the validation accuracy further declined to 85.20%.

These results indicate that freezing fewer layers and allowing more of the network to adapt to the iNaturalist dataset leads to better performance. This is expected, as deeper layers tend to capture more task-specific features and benefit more from fine-tuning on the target dataset.


