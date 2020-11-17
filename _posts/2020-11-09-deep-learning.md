---
layout: post
title:  "Insects Image Classification"
date:   2020-11-09
image:  images/HW7/logo.jpg
tags:   [study]
---

Hello, welcome to my blog! This post will share the data manipulation of **[Image Classification][Image Classification]**.

Please visit my **[GitHub][GitHub]** for more information. 

# Introduction

Insects classification.

# Preparations
We should first import the necessary packages. For the first model, I basically follow the official guide from tensorflow.
{% highlight python %}
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
{% endhighlight %}

# Insects classification
## load data
{% highlight python %}
import pathlib
train_dir = pathlib.Path("insects/train")
test_dir = pathlib.Path("insects/test")
{% endhighlight %}

We can then take a look of the insects in train and test directories.
{% highlight python %}
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.savefig('train.png')
{% endhighlight %}
![1]({{site.baseurl}}/images/HW7/train.png)

{% highlight python %}
plt.figure(figsize=(10, 10))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.savefig('test.png')
{% endhighlight %}
![2]({{site.baseurl}}/images/HW7/test.png)

## Configure the dataset for performance
{% highlight python %}
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
{% endhighlight %}

## Create the model
{% highlight python %}
num_classes = 3

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
{% endhighlight %}

## Compile the model
{% highlight python %}
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
{% endhighlight %}

## Model summary
{% highlight python %}
model.summary()
{% endhighlight %}
        Model: "sequential_1"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        rescaling_1 (Rescaling)      (None, 180, 180, 3)       0         
        _________________________________________________________________
        conv2d_3 (Conv2D)            (None, 180, 180, 16)      448       
        _________________________________________________________________
        max_pooling2d_3 (MaxPooling2 (None, 90, 90, 16)        0         
        _________________________________________________________________
        conv2d_4 (Conv2D)            (None, 90, 90, 32)        4640      
        _________________________________________________________________
        max_pooling2d_4 (MaxPooling2 (None, 45, 45, 32)        0         
        _________________________________________________________________
        conv2d_5 (Conv2D)            (None, 45, 45, 64)        18496     
        _________________________________________________________________
        max_pooling2d_5 (MaxPooling2 (None, 22, 22, 64)        0         
        _________________________________________________________________
        flatten_1 (Flatten)          (None, 30976)             0         
        _________________________________________________________________
        dense_2 (Dense)              (None, 128)               3965056   
        _________________________________________________________________
        dense_3 (Dense)              (None, 3)                 387       
        =================================================================
        Total params: 3,989,027
        Trainable params: 3,989,027
        Non-trainable params: 0
        _________________________________________________________________

<br>

## Train the model
{% highlight python %}
epochs=10
history = model.fit(
  train_ds,
  validation_data=test_ds,
  epochs=epochs
)
{% endhighlight %}

        Epoch 1/10
        32/32 [==============================] - 8s 252ms/step - loss: 0.9014 - accuracy: 0.6487 - val_loss: 0.6370 - val_accuracy: 0.7722
        Epoch 2/10
        32/32 [==============================] - 8s 247ms/step - loss: 0.4568 - accuracy: 0.8234 - val_loss: 0.5376 - val_accuracy: 0.8333
        Epoch 3/10
        32/32 [==============================] - 8s 249ms/step - loss: 0.3298 - accuracy: 0.8803 - val_loss: 0.3938 - val_accuracy: 0.8722
        Epoch 4/10
        32/32 [==============================] - 8s 251ms/step - loss: 0.2366 - accuracy: 0.9097 - val_loss: 0.2523 - val_accuracy: 0.9111
        Epoch 5/10
        32/32 [==============================] - 8s 250ms/step - loss: 0.1741 - accuracy: 0.9441 - val_loss: 0.2255 - val_accuracy: 0.9167
        Epoch 6/10
        32/32 [==============================] - 8s 251ms/step - loss: 0.1165 - accuracy: 0.9598 - val_loss: 0.1234 - val_accuracy: 0.9556
        Epoch 7/10
        32/32 [==============================] - 8s 251ms/step - loss: 0.1012 - accuracy: 0.9607 - val_loss: 0.1412 - val_accuracy: 0.9556
        Epoch 8/10
        32/32 [==============================] - 8s 251ms/step - loss: 0.0674 - accuracy: 0.9715 - val_loss: 0.0765 - val_accuracy: 0.9778
        Epoch 9/10
        32/32 [==============================] - 8s 252ms/step - loss: 0.0339 - accuracy: 0.9912 - val_loss: 0.0464 - val_accuracy: 0.9833
        Epoch 10/10
        32/32 [==============================] - 8s 254ms/step - loss: 0.0259 - accuracy: 0.9961 - val_loss: 0.0339 - val_accuracy: 0.9944

<br>

## Visualize training results
{% highlight python %}
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
{% endhighlight %}

![3]({{site.baseurl}}/images/HW7/viz1.png)

[Image Classification]: https://www.insectimages.org/index.cfm
[GitHub]: https://github.com/eveyimi/eveyimi.github.io

