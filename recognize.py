import matplotlib.pyplot as plt
import os
import cv2
import keras
import numpy as np
import random
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from sklearn import model_selection, metrics
from scipy import ndimage
import pandas as pd


def get_best_shift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)

    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)

    return shiftx, shifty


def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


def shift_according_to_center_of_mass(img):
    img = cv2.bitwise_not(img)

    # Centralize the image according to center of mass
    shiftx, shifty = get_best_shift(img)
    shifted = shift(img, shiftx, shifty)
    img = shifted

    img = cv2.bitwise_not(img)
    return img


batch_size = 64
num_classes = 3
epochs = 1

img_rows, img_cols = 224, 224
PATH = 'dataset'
CATEGORIES = ['blank', 'fist', 'palm']
training_data = []

for category in CATEGORIES:
    path = os.path.join(PATH, category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
        image_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        resized_array = cv2.resize(image_array, (img_rows, img_cols), interpolation=cv2.INTER_LANCZOS4)
        _, resized_array = cv2.threshold(resized_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        resized_array = shift_according_to_center_of_mass(resized_array)
        training_data.append([resized_array, class_num])
print(len(training_data))

fig = plt.figure(figsize=(9, 8))
rows, columns = 5, 10
ax = []
for i in range(columns * rows):
    # create subplot and append to ax
    ax.append(fig.add_subplot(rows, columns, i + 1))
    ax[-1].set_title("ax:" + str(training_data[1990 + i][1]))  # set title
    plt.imshow(training_data[1990 + i][0], cmap='gray')
    plt.axis("off")
plt.show()

random.shuffle(training_data)

X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X, dtype="uint8")
X = X.reshape(len(training_data), img_cols, img_rows, 1)
y = np.array(y)

print("len x: ", len(X))
print("len y: ", len(y))

ts = 0.3
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=ts, random_state=42)

model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(img_rows, img_cols, 1)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print("test_acc: ", test_acc, " test_loss: ", test_loss)
predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)
df = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred),
             columns=["Predicted blank","Predicted fist","Predicted palm"],
             index=["blank", "fist", "palm"])
print(df)

model.save('models/model2.hdf5')
model.save_weights('models/digitRecognition2.h5')