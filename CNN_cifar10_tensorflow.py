import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging


from keras import datasets, models, metrics, layers, preprocessing
import tensorflow as tf


cifar10 = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data() #train test segreagation

#classes
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]


#Image preprocess
train_images = train_images / 255.0
test_images = test_images / 255.0

#Draw images with matplot lib


ImgNum = int(input("Please Enter image number: "))

plt.figure()
plt.imshow(train_images[ImgNum])
plt.xlabel(class_names[train_labels[ImgNum][0]])
plt.colorbar()
plt.grid(False)
plt.show()


#Creat the model

model = tf.keras.models.Sequential(name='ConvNet')
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape = (32,32,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))


#compile the model 

model.compile(optimizer='adam', 

loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

metrics=['accuracy'])

#Fit the model

history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images,test_labels))

test_loss, test_accuracy = model.evaluate(test_images,test_labels, verbose=2)


#Check model prediction
image = test_images[4]
prediction = model.predict(np.array([image]))
class_names[np.argmax(prediction)]

#Data Augmentation

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

#Pick image to transform
test_img = train_images[14]
img = tf.keras.preprocessing.image.img_to_array(test_img)
img.reshape((1,) + img.shape)

i = 0

for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):
    plt.figure(i)
    plot = plt.imshow(tf.keras.preprocessing.image.img_to_array(batch[0]))
    i += 1
    if i > 4:
        break

plt.show()