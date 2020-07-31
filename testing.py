import matplotlib.pyplot as plt
import os
import cv2
from keras.models import load_model
import numpy as np
import random
from sklearn import model_selection, metrics
from scipy import ndimage
import pandas as pd
import preprocessing


batch_size = 64
num_classes = 3
epochs = 5

img_rows, img_cols = 224, 224
PATH = 'test'
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
    ax[-1].set_title("ax:" + str(training_data[490 + i][1]))  # set title
    plt.imshow(training_data[490 + i][0], cmap='gray')
    plt.axis("off")
plt.show()

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

model = load_model("models/model5.hdf5")

test_loss, test_acc = model.evaluate(X, y)
print("test_acc: ", test_acc, " test_loss: ", test_loss)
predictions = model.predict(X)

y_pred = np.argmax(predictions, axis=1)
df = pd.DataFrame(metrics.confusion_matrix(y, y_pred),
             columns=["Predicted fist","Predicted palm"],
             index=["fist", "palm"])
print(df)

indices = np.nonzero(y_pred != y)[0]
print(indices," ",type(indices))
print(predictions.shape)
for ind in enumerate(indices):
    print(predictions[i][:])

plt.rcParams['figure.figsize'] = (7,14)
figure_evaluation = plt.figure()

for i, incorrect in enumerate(indices):
    plt.subplot(6,3,i+10)
    plt.imshow(X[incorrect].reshape(224,224), cmap='gray', interpolation='none')
    plt.title(
      "Predicted {}, Truth: {}".format(y_pred[incorrect],
                                       y[incorrect]))
    plt.xticks([])
    plt.yticks([])
plt.show()