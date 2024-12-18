import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.expand_dims(x_train, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)
x_train = tf.image.resize_with_crop_or_pad(x_train, 32, 32)
x_test = tf.image.resize_with_crop_or_pad(x_test, 32, 32)
x_train = x_train/255
x_test = x_test/255
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
model = models.Sequential([
    layers.Conv2D(6, kernel_size=(5,5), strides=1, padding='valid', activation='relu', input_shape=(32,32,1)),

    layers.AveragePooling2D(pool_size=(2,2), strides=2),

    layers.Conv2D(16, kernel_size=(5,5), strides=1, padding='valid', activation='relu'),

    layers.AveragePooling2D(pool_size=(2,2), strides=2),

    layers.Flatten(),

    layers.Dense(120, activation='relu'),

    layers.Dense(84, activation='relu'),

    layers.Dense(10, activation='softmax')
])
model.summary()
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128,
                    epochs=10, validation_split=0.2, verbose=2)
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
# Plot the training and validation accuracy and loss over epochs
epochs = range(1, len(history.history['accuracy']) + 1)

# Plot accuracy
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['loss'], label='Training Loss')
plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
