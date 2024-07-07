import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist #load data set

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() #train test segreagation

class_names = ['T-shirt/top','Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#Draw images with matplot lib

plt.figure()
plt.imshow(train_images[2])
plt.colorbar()
plt.grid(False)
plt.show()

#Image preprocessing
train_images = train_images / 255.0
test_images = test_images / 255.0


#create the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), #Layer 1 input
    keras.layers.Dense(128, activation='relu'), # Layer 2 hidden
    keras.layers.Dense(10, activation='softmax') #Layer 3 ouput
])

#compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics= ['accuracy']
)

#fit the model
model.fit(x=train_images,y=train_labels, epochs=10,shuffle=True)

#Check accuracy
test_loss, test_accuracy = model.evaluate(x=test_images, y=test_labels, verbose=1)
print("Test accuracy: ", test_accuracy)

COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label, class_names):

    prediction = model.predict(np.array([image]))
    predicted_class = class_names[np.argmax(prediction)]
    showimage(image, class_names[correct_label], predicted_class)


def showimage(img,label,guess):
 
    fig, ax = plt.subplots(figsize=(5, 5))
    
    cax = ax.imshow(img, cmap=plt.cm.binary)
    
    ax.set_title(f"Expected: {label}", color='black')
    ax.set_xlabel(f"Guess: {guess}", color='black')
    
    fig.colorbar(cax)
    ax.grid(False)

    
    plt.show() 


def get_number():
    while True:
        num = input("Pick a number: ")
        if num.isdigit():
            num = int(num)
            if 0 <= num <= 1000:
                return num
            else:
                print("Try again...")


num  = get_number()
predict(model, test_images[num], test_labels[num] ,class_names)
