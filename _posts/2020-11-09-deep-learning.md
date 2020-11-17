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

# Transfer learning with TensorFlow Hub
**[TensorFlow Hub][TensorFlow Hub]** is a repository of pre-trained TensorFlow models. In this section, I am going to use image classification model from TensorFlow Hub and do transfer learning to fine-tune a model for image classes. We can use a model from TFHub to train a custom image classier by retraining the top layer of the model to recognize the classes in our dataset.

## Set up
{% highlight python %}
import numpy as np
import time
import PIL.Image as Image
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
{% endhighlight %}

## An ImageNet classifier
### Download the classifier
Use hub.KerasLayer to load a MobileNetV2 model from TensorFlow Hub. Any compatible image classifier model from tfhub.dev will work here.
{% highlight python %}
classifier_model ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4" 
IMAGE_SHAPE = (224, 224)
classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
])
{% endhighlight %}
### Decode the predictions
Take the predicted class ID and fetch the ImageNet labels to decode the predictions.
{% highlight python %}
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())
{% endhighlight %}

## Transfer learning
### Load data
{% highlight python %}
batch_size = 32
img_height = 224
img_width = 224
train_dir = pathlib.Path("insects/train")
test_dir = pathlib.Path("insects/test")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_dir,
  image_size=(img_height, img_width),
  batch_size=batch_size)
// Found 1019 files belonging to 3 classes.
class_names = np.array(train_ds.class_names)
print(class_names)
// ['beetles' 'cockroach' 'dragonflies']
{% endhighlight %}

TensorFlow Hub's conventions for image models is to expect float inputs in the [0, 1] range. Use the Rescaling layer to achieve this.
{% highlight python %}
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
{% endhighlight %}

Make sure to use buffered prefetching so we can yield data from disk without having I/O become blocking.
{% highlight python %}
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
{% endhighlight %}

{% highlight python %}
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break
{% endhighlight %}
(32, 224, 224, 3)
(32,)

### Run the classifier on a batch of images
{% highlight python %}
result_batch = classifier.predict(train_ds)
predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]
predicted_class_names
{% endhighlight %}
    array(['bonnet', 'cockroach', 'damselfly', ..., 'spindle', 'lacewing','slug'], dtype='<U30')

{% highlight python %}
plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
    plt.subplot(6,5,n+1)
    plt.imshow(image_batch[n])
    plt.title(predicted_class_names[n])
    plt.axis('off')
_ = plt.suptitle("ImageNet predictions")
plt.show()
plt.savefig('batch_pred.png')
{% endhighlight %}

![3]({{site.baseurl}}/images/HW7/batch_pred.png)

The results might seem not that good. Let's move to the next section.

### Download the headless model
{% highlight python %}
feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4" 
feature_extractor_layer = hub.KerasLayer(
    feature_extractor_model, input_shape=(224, 224, 3), trainable=False)
feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)
{% endhighlight %}
        (32, 1280)

{% highlight python %}
num_classes = len(class_names)

model = tf.keras.Sequential([
  feature_extractor_layer,
  tf.keras.layers.Dense(num_classes)
])

model.summary()
{% endhighlight %}
        Model: "sequential_1"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        keras_layer_2 (KerasLayer)   (None, 1280)              2257984   
        _________________________________________________________________
        dense (Dense)                (None, 3)                 3843      
        =================================================================
        Total params: 2,261,827
        Trainable params: 3,843
        Non-trainable params: 2,257,984
        _________________________________________________________________

<br>

{% highlight python %}
predictions = model(image_batch)
predictions.shape
{% endhighlight %}
        TensorShape([32, 3])

<br>

We then compile and fit the model.
{% highlight python %}
model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []
    
    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()

batch_stats_callback = CollectBatchStats()
history = model.fit(train_ds, epochs=2,
                    callbacks=[batch_stats_callback])
{% endhighlight %}
        Epoch 1/2
        32/32 [==============================] - 8s 239ms/step - loss: 0.1042 - acc: 1.0000
        Epoch 2/2
        32/32 [==============================] - 8s 236ms/step - loss: 0.0410 - acc: 1.0000

<br>

We then visualize the results.
{% highlight python %}
plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(batch_stats_callback.batch_losses)
plt.savefig('loss.png')
{% endhighlight %}

![3]({{site.baseurl}}/images/HW7/loss.png)

{% highlight python %}
plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(batch_stats_callback.batch_acc)
plt.savefig('acc.png')
{% endhighlight %}

![3]({{site.baseurl}}/images/HW7/acc.png)

{% highlight python %}
predicted_batch = model.predict(image_batch)
predicted_id = np.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]
{% endhighlight %}

{% highlight python %}
plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
    plt.subplot(6,5,n+1)
    plt.imshow(image_batch[n])
    plt.title(predicted_label_batch[n].title())
    plt.axis('off')
_ = plt.suptitle("Model predictions")
plt.savefig('model_pred.png')
{% endhighlight %}

![3]({{site.baseurl}}/images/HW7/model_pred.png)

[Image Classification]: https://www.insectimages.org/index.cfm
[GitHub]: https://github.com/eveyimi/eveyimi.github.io

