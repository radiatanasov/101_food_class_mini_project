# 101 Food Classes Mini Project

The goal of this project is to surpass the results of the original Food101 paper using only 10% of the data. You can read the original paper [here](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf). Our approach will be incremental: start small, get a model working, evaluate our experiments, and then gradually scale up.

![scalling_data](images/1.png)

## What will this project contain:
- Training a feature extraction transfer learning model on 10% of the Food101 training data
- Fine-tuning the feature extraction model
- Saving and loading our trained model
- Evaluating the performance of the Food Vision model trained on 10% of the training data
- Finding the model's most incorrect predictions

## 101 Food Classes: Working with less data

The original Food101 dataset contains 1000 images per class (750 in the training set and 250 in the test set), totaling 101,000 images.

Instead of jumping directly into modeling with this extensive dataset, we'll continue our approach of experimentation by first evaluating how our previously effective models perform with just 10% of the training data.

This means we'll develop a model for each of the 101 food classes using 75 training images and evaluate its performance on 250 test images.

The data comes in the common image classification data format of:

```
10_food_classes_10_percent
├───train
│   ├───pizza
│   │   ├───123124.jpg
│   │   ├───123445.jpg
│   │   └───...
│   └───steak
│       ├───54321.jpg
│       ├───12345.jpg
│       └───...
└───test
    ├───pizza
    │   ├───111234.jpg
    │   ├───523234.jpg
    │   └───...
    └───steak
        ├───123124.jpg
        ├───123123.jpg
        └───...
```

## Train a model with transfer learning on 10% of 101 food classes

More specifically, our goal will be to see if we can beat the baseline from the original Food101 paper (50.76% accuracy on 101 classes) with 10% of the training data and the following modeling setup:

- A ModelCheckpoint callback to save our progress during training, this means we could experiment with further training later without having to train from scratch every time
- Data augmentation built right into the model
- A headless (no top layers) EfficientNetB0 architecture from tf.keras.applications as our base model
- A Dense layer with 101 hidden neurons (same as the number of food classes) and softmax activation as the output layer
- Categorical crossentropy as the loss function since we're dealing with more than two classes
- The Adam optimizer with the default settings
- Fitting for 5 full passes on the training data while evaluating on 15% of the test data

## Analysis

For the analysis, I used a confusion matrix and a classification report, from which I extracted the F1 score and class name to determine which classes are best recognized by the models.

## Visualizing predictions on test images

Specifically, it'll:

- Read in a target image filepath using tf.io.read_file().
- Turn the image into a Tensor using tf.io.decode_image().
- Resize the image to be the same size as the images our model has been trained on (224 x 224) using tf.image.resize().
- Scale the image to get all the pixel values between 0 & 1 if necessary.

## Contact
If you have any questions or suggestions, you can contact me at radi2035@gmail.com.

