from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from keras import models, layers, losses
import cv2
import seaborn as sns
import tensorflow as tf


# whether you want to save the model after training
save_model = True
# wether you want to used the last saved model
using_saved_model = False

# mnist or patch
mode = 'mnist'


def plot_digits(X, y):
    plt.figure(figsize=(20,6))
    for i in range(10):
        if np.where(y==f"{i}")[0].size > 0:
            index = np.where(y==f"{i}")[0][0]
            digit_sub = plt.subplot(2, 5, i + 1)
            digit_sub.imshow(np.reshape(X[index], (dim_row,dim_col)), cmap="gray")
            digit_sub.set_xlabel(f"Digit {y[index]}")
    plt.show()


def plot_history(hist):
    plt.plot(hist.history['accuracy'], label='accuracy')
    plt.plot(hist.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()


def plot_confusion_matrix(X_test, y_test):
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    conf_matrix = confusion_matrix(y_test, predicted_classes)
    sns.heatmap(conf_matrix, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title('Confusion Matrix')
    plt.show()


def resize_images(images, width, height):
    # define size and initialize the output variable
    size = (width, height)
    output_images = []

    # resize each image
    for image in images:
        output_images.append(cv2.resize(image, size))

    return np.array(output_images)


batch_size = 40

# load data from mnist
X_mnist, y_mnist = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser='auto')
y_mnist = np.array(y_mnist, dtype=int)
dim_row = 28
dim_col = 28
# unflatten mnist images
X_mnist = np.array([np.reshape(xf, (dim_row, dim_col)) for xf in X_mnist])

# load data from TSP
with open(f"{os.path.dirname(os.path.realpath(__file__))}/all_drawings.json", 'r') as file:
    data = json.load(file)
X_tsp = np.array([d[0] for d in data])
y_tsp = np.array([int(d[1]) for d in data])

if mode == 'mnist':
    print("setting up mnist training. Train: MNIST. Test: TSP")
    iters = 15
    batch_size = 200

    dim_row = 28
    dim_col = 28

    # resize the tsp images
    X_tsp = resize_images(X_tsp, dim_row, dim_col)

    #old split
    # X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, test_size=0.25)

    #nex split
    # train on mnist
    X_train = X_mnist
    y_train = y_mnist

    # test with touchpad data
    X_test = X_tsp
    y_test = y_tsp

else:
    print("setting up tsp training. Train: TSP, Test: TSP")
    dim_row = 27
    dim_col = 19
    iters = 100

    X_train, X_test, y_train, y_test = train_test_split(X_tsp, y_tsp, test_size=0.12)
    batch_size = 20

# print shape of train and test data
print("train data shape:")
print(X_train.shape)
print(y_train.shape)
print("test data shape:")
print(X_test.shape)
print(y_test.shape)

# to train or not to train
if using_saved_model:
    # load saved model, should have a check if it exsists, but im lazy today
    if mode == "mnist":
        model = tf.keras.models.load_model('saved_model/my_model.keras')
    else:
        model = tf.keras.models.load_model('saved_model/my_tsp_model.keras')
    print("loaded model")
else:
    # setup NN for model
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(dim_row, dim_col, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    # configure model
    the_optimizer = "adam"
    model.compile(optimizer=the_optimizer,
                  loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # train model
    hist = model.fit(X_train, y_train, epochs=iters, batch_size=batch_size, validation_split=0.15)

    # plot accuracy over time
    plot_history(hist)

# save model when you want
if save_model and using_saved_model != True:
    # save model after evaluating
    if mode == "mnist":
        model.save('saved_model/my_model.keras')
    else:
        model.save('saved_model/my_tsp_model.keras')
    print("saved model")

# evaluate model
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
print(f"Test loss: {test_loss} Test accuracy: {test_acc}")

plot_confusion_matrix(X_test, y_test)



