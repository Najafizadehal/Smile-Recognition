import glob
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from mtcnn import MTCNN
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from joblib import dump

detector = MTCNN()


def face_detector(img):
    try:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = detector.detect_faces(rgb_img)[0]
        x, y, w, h = out['box']

        return img[y:y + h, x:x + w]

    except:
        pass


def preprocessing(path):
    data = []
    labels = []
    for i, item in enumerate(glob.glob(path)):
        img = cv2.imread(item)
        face = face_detector(img)

        if face is None:
            continue

        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = local_binary_pattern(face, 24, 3)
        face = cv2.resize(face, (64, 64))
        face = face.flatten()
        face = face / 255.0

        data.append(face)

        label = item.split('\\')[-2]
        labels.append(label)

        if i % 100 == 0:
            print(f'[INFO]: {i} / 4000 images processed')

    data = np.array(data)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_model(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(Dense(128, input_dim=4096, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(np.unique(y_train)), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    # Evaluate the model on test set
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Accuracy: %.2f' % (accuracy*100))

    return model


X_train, X_test, y_train, y_test = preprocessing('files\\*\\*')

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Convert labels to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = classification_model(X_train, y_train, X_test, y_test)

model.save('smile_classifier.h5')
