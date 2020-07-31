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
import preprocessing
from keras.preprocessing.image import ImageDataGenerator

batch_size = 64
num_classes = 3
epochs = 10

img_rows, img_cols = 224, 224
PATH = 'dataset'
CATEGORIES = ['fist', 'palm']
training_data = []

for category in CATEGORIES:
    path = os.path.join(PATH, category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
        path2 = os.path.join(path, img)
        orig = cv2.imread(path2)
        image = preprocessing.pre_process_image(path2)
        orig, image = preprocessing.find_corners_of_largest_polygon(image, orig)
        print(image.shape)
        training_data.append([image, class_num])
print(len(training_data))

random.shuffle(training_data)

fig = plt.figure(figsize=(9, 8))
rows, columns = 5, 10
ax = []
for i in range(columns * rows):
    # create subplot and append to ax
    ax.append(fig.add_subplot(rows, columns, i + 1))
    ax[-1].set_title("ax:" + str(training_data[0 + i][1]))  # set title
    plt.imshow(training_data[0 + i][0], cmap='gray')
    plt.axis("off")
plt.show()

X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X, dtype="uint8")
X = X.reshape(-1, img_cols, img_rows, 1)
y = np.array(y)

print("len x: ", len(X))
print("len y: ", len(y))

ts = 0.3
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=ts, random_state=42)

datagen = ImageDataGenerator(horizontal_flip=True,
                             rotation_range=45,
                             zoom_range=[1.5, 1.0],
                             )
datagen.fit(X_train)

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

model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                    epochs=epochs, verbose=2, validation_data=(X_test, y_test),
                    steps_per_epoch=X_train.shape[0] // batch_size)

test_loss, test_acc = model.evaluate(X_test, y_test)
print("test_acc: ", test_acc, " test_loss: ", test_loss)
predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)
df = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred),
                  columns=["Predicted fist", "Predicted palm"],
                  index=["fist", "palm"])
print(df)

model.save('models/model5.hdf5')
model.save_weights('models/digitRecognition5.h5')
