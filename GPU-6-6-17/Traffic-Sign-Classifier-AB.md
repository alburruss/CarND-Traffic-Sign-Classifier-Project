
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Build a Traffic Sign Recognition Classifier

In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 

In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.

The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.


>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

---
## Step 0: Load The Data


```python
# edit: June 1 2017
# web download data: for VM only
import requests, zipfile, io

r = requests.get("https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip")
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

```


```python
# Load pickled data
import pickle

# Fill this in based on where you saved the training and testing data

training_file = "train.p"
validation_file= "valid.p"
testing_file = "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
```

---

## Step 1: Dataset Summary & Exploration

The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**

Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas


```python
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Visualizations will be shown in the notebook.
%matplotlib inline

image = X_train[0]

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

n_valid = len(X_valid)

# TODO: What's the shape of an traffic sign image?
# matplotlib triple(length, width, layers)
image_shape = image.shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(y_train)

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Number of validation examples =", n_valid)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```

    Number of training examples = 34799
    Number of testing examples = 12630
    Number of validation examples = 4410
    Image data shape = (32, 32, 3)
    Number of classes = 34799


### Include an exploratory visualization of the dataset

Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc.

The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.

**NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections.


```python
# print sample from X_train
image = X_train[101]
# print out sample img, dimensions and plot
print('This image is:', type(image), 'with dimesions:', image.shape)
print('Label of training img 101', y_train[101])
plt.imshow(image)
```

    This image is: <class 'numpy.ndarray'> with dimesions: (32, 32, 3)
    Label of training img 101 41





    <matplotlib.image.AxesImage at 0x7f981816bc88>




![png](output_9_2.png)



```python
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
# grey --> contrast stretching: normalize -->  reshape
# print sample from X_train

from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity

s = (32, 32, 3)
out = np.empty(s)
img = X_train[101]

# print out array data for img
print('img')
print('This image is:', type(img), 'with dimesions:', img.shape)
print('Label ', y_train[101])

# grey
img2 = rgb2gray(img)


# print out array data for greyscale transformed img
print('img2')
print('This image is:', type(img2), 'with dimesions:', img2.shape)
print('Label ', y_train[101])
print('Greyscale transformed data')
#for j in range(32):
#    print(np.asarray(img2[j]))


# resize
img3 = np.asarray(img2).reshape((32, 32, 1))


# print out sample img, dimensions and plot
print('img3')
print('This image is:', type(img3), 'with dimesions:', img3.shape)
print('Label ', y_train[101])

# img Normalization: rescale_intensity
img_rescale = rescale_intensity(img2, in_range = 'image', out_range = (0.2, 0.8) )
print('img_rescale')
print('This image is:', type(img_rescale), 'with dimesions:', img_rescale.shape)
print('Label ', y_train[101])
print('rescale_intensity transformed data')
#for j in range(32):
#    print(np.asarray(img_rescale[j]))

plt.imshow(img_rescale, cmap='gray')

```

    img
    This image is: <class 'numpy.ndarray'> with dimesions: (32, 32, 3)
    Label  41
    img2
    This image is: <class 'numpy.ndarray'> with dimesions: (32, 32)
    Label  41
    Greyscale transformed data
    img3
    This image is: <class 'numpy.ndarray'> with dimesions: (32, 32, 1)
    Label  41
    img_rescale
    This image is: <class 'numpy.ndarray'> with dimesions: (32, 32)
    Label  41
    rescale_intensity transformed data





    <matplotlib.image.AxesImage at 0x7f980a628ba8>




![png](output_10_2.png)



```python
# print sample from X_valid
img = X_valid[100]
# print out sample img, dimensions and plot
print('This image is:', type(img), 'with dimesions:', img.shape)
print('Label 0 ', y_valid[100])

# grey
img2 = rgb2gray(img)

# img Normalization: rescale_intensity
img_rescale = rescale_intensity(img2, in_range = 'image', out_range = (0.2, 0.8) )

plt.imshow(img_rescale, cmap='gray')
```

    This image is: <class 'numpy.ndarray'> with dimesions: (32, 32, 3)
    Label 0  31





    <matplotlib.image.AxesImage at 0x7f980a593b38>




![png](output_11_2.png)



```python
# print sample from X_train
img = X_test[0]
# print out sample img, dimensions and plot
print('This image is:', type(img), 'with dimesions:', img.shape)
print('Label 0 ', y_test[0])

# grey
img2 = rgb2gray(img)

# img Normalization: rescale_intensity
img_rescale = rescale_intensity(img2, in_range = 'image', out_range = (0.2, 0.8) )

plt.imshow(img_rescale, cmap='gray')
```

    This image is: <class 'numpy.ndarray'> with dimesions: (32, 32, 3)
    Label 0  16





    <matplotlib.image.AxesImage at 0x7f980a501198>




![png](output_12_2.png)


----

## Step 2: Design and Test a Model Architecture

Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 

With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 

There are various aspects to consider when thinking about this problem:

- Neural network architecture (is the network over or underfitting?)
- Play around preprocessing techniques (normalization, rgb to grayscale, etc)
- Number of examples per label (some have more than others).
- Generate fake data.

Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

### Pre-process the Data Set (normalization, grayscale, etc.)

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.


```python
### Preprocess the data here. Preprocessing steps could include normalization, 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

# shuffle training data set
from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)

# print sample img
image = X_train[101]
#print out sample img, dimensions and plot
print('This image is:', type(image), 'with dimesions:', image.shape)
print('Label of training img 101 is ', y_train[101])

plt.imshow(image)

```

    This image is: <class 'numpy.ndarray'> with dimesions: (32, 32, 3)
    Label of training img 101 is  11





    <matplotlib.image.AxesImage at 0x7f97fe1e62b0>




![png](output_16_2.png)



```python
from skimage.color import rgb2gray
from skimage import exposure

# validation data set

X_valid_G = []

# pre-process routine

for i in range(len(X_valid)):
    img = X_valid[i]
    
    # grey
    img2 = rgb2gray(img)
       
    # img Normalization: rescale_intensity
    eq_img = rescale_intensity(img2, in_range = 'image', out_range = (0.2, 0.8) )

    # reshape
    img4 = np.asarray(eq_img).reshape((32, 32, 1))

    X_valid_G.append( img4 )

print(len(X_valid_G))
print(X_valid_G[0].shape)
```

    4410
    (32, 32, 1)



```python
# testing dataset

X_test_G = []

# pre-process routine
for i in range(len(X_test)):
    img_t = X_test[i]
    
    # convert X_train imgs to grayscale
    img_t2 = rgb2gray(img_t)
   
    # img Normalization: rescale_intensity
    eq_t2img = rescale_intensity(img_t2, in_range = 'image', out_range = (0.2, 0.8) )

    # reshape
    img_t4 = np.asarray(eq_t2img).reshape((32, 32, 1))
    
    X_test_G.append( img_t4 )

print(len(X_test_G))
print(X_test_G[0].shape)
```

    12630
    (32, 32, 1)



```python
# training data set

X_train_G = []

# convert X_train imgs to grayscale
for i in range(len(X_train)):
    img_r = X_train[i]
    
    # greyscale transform
    img_r2 = rgb2gray(img_r)

    # img Normalization: rescale_intensity
    eq_r2_img = rescale_intensity(img_r2, in_range = 'image', out_range = (0.2, 0.8) )
    
    # reshape
    img_r4 = np.asarray(eq_r2_img).reshape((32, 32, 1))

    X_train_G.append( img_r4 )

print(len(X_train_G))
print(X_train_G[0].shape)
```

    34799
    (32, 32, 1)


### Model Architecture


```python
### Define your architecture here.
### Feel free to use as many code cells as needed.

import tensorflow as tf

EPOCHS = 20
BATCH_SIZE = 32
```


```python
# filter depth 32, 64

from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    dropout = 0.70
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x32.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 32), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(32))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x32. Output = 14x14x32.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x64.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 64), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(64))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x64. Output = 5x5x64.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x64. Output = 1600.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 1600. Output = 800.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(1600, 800), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(800))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)
    # apply dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # SOLUTION: Layer 4: Fully Connected. Input = 800. Output = 400.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(800, 400), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(400))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)
    # apply dropout
    fc2 = tf.nn.dropout(fc2, dropout)
    
    # SOLUTION: Layer 5: Fully Connected. Input = 400. Output = 200.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(400, 200), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(200))
    fc3    = tf.matmul(fc2, fc3_W) + fc3_b
    
    # SOLUTION: Activation.
    fc3    = tf.nn.relu(fc3)
    # apply dropout
    fc3 = tf.nn.dropout(fc3, dropout)
                        
    # SOLUTION: Layer 6: Fully Connected. Input = 200. Output = 100.
    fc4_W  = tf.Variable(tf.truncated_normal(shape=(200, 100), mean = mu, stddev = sigma))
    fc4_b  = tf.Variable(tf.zeros(100))
    fc4    = tf.matmul(fc3, fc4_W) + fc4_b
    
    # SOLUTION: Activation.
    fc4    = tf.nn.relu(fc4)
    # apply dropout
    fc4 = tf.nn.dropout(fc4, dropout)
    
    # SOLUTION: Layer 7: Fully Connected. Input = 100. Output = 48.
    fc5_W  = tf.Variable(tf.truncated_normal(shape=(100, 48), mean = mu, stddev = sigma))
    fc5_b  = tf.Variable(tf.zeros(48))
    fc5    = tf.matmul(fc4, fc5_W) + fc5_b
    
    # SOLUTION: Activation.
    fc5    = tf.nn.relu(fc5)
    # apply dropout
    fc5 = tf.nn.dropout(fc5, dropout)
    
    # SOLUTION: Layer 8: Fully Connected. Input = 48. Output = 24.
    fc6_W  = tf.Variable(tf.truncated_normal(shape=(48, 24), mean = mu, stddev = sigma))
    fc6_b  = tf.Variable(tf.zeros(24))
    fc6    = tf.matmul(fc5, fc6_W) + fc6_b
    
    # SOLUTION: Activation.
    fc6    = tf.nn.relu(fc6)
    # apply dropout
    fc6 = tf.nn.dropout(fc6, dropout)
    
    # SOLUTION: Layer 9: Fully Connected. Input = 24. Output = 10.
    fc7_W  = tf.Variable(tf.truncated_normal(shape=(24, 10), mean = mu, stddev = sigma))
    fc7_b  = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc6, fc7_W) + fc7_b
    
    return logits
```

### Train, Validate and Test the Model

A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.


```python
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

# features and labels
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)
```


```python
# training pipeline

rate = 0.1

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
```


```python
# Model evaluation

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
```


```python
# Train the Model
from sklearn.utils import shuffle

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train_G)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train_G, y_train = shuffle(X_train_G, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_G[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid_G, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
```

    Training...
    
    EPOCH 1 ...
    Validation Accuracy = 0.646
    
    EPOCH 2 ...
    Validation Accuracy = 0.646
    
    EPOCH 3 ...
    Validation Accuracy = 0.646
    
    EPOCH 4 ...
    Validation Accuracy = 0.646
    
    EPOCH 5 ...
    Validation Accuracy = 0.646
    
    EPOCH 6 ...
    Validation Accuracy = 0.646
    
    EPOCH 7 ...
    Validation Accuracy = 0.646
    
    EPOCH 8 ...
    Validation Accuracy = 0.646
    
    EPOCH 9 ...
    Validation Accuracy = 0.646
    
    EPOCH 10 ...
    Validation Accuracy = 0.646
    
    EPOCH 11 ...
    Validation Accuracy = 0.646
    
    EPOCH 12 ...
    Validation Accuracy = 0.646
    
    EPOCH 13 ...
    Validation Accuracy = 0.646
    
    EPOCH 14 ...
    Validation Accuracy = 0.646
    
    EPOCH 15 ...
    Validation Accuracy = 0.646
    
    EPOCH 16 ...
    Validation Accuracy = 0.646
    
    EPOCH 17 ...
    Validation Accuracy = 0.646
    
    EPOCH 18 ...
    Validation Accuracy = 0.646
    
    EPOCH 19 ...
    Validation Accuracy = 0.646
    
    EPOCH 20 ...
    Validation Accuracy = 0.646
    
    Model saved



```python
# Evaluate the model

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test_G, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
```

    Test Accuracy = 0.625


---

## Step 3: Test a Model on New Images

To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

### Load and Output the Images


```python
### Load the images and plot them here.
### Feel free to use as many code cells as needed.

```

### Predict the Sign Type for Each Image


```python
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
```

### Analyze Performance


```python
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
```

### Output Top 5 Softmax Probabilities For Each Image Found on the Web

For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 

The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.

`tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.

Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tk.nn.top_k` is used to choose the three classes with the highest probability:

```
# (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])
```

Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:

```
TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))
```

Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.


```python
### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
```

---

## Step 4: Visualize the Neural Network's State with Test Images

 This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.

 Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.

For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.

<figure>
 <img src="visualize_cnn.png" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above)</p> 
 </figcaption>
</figure>
 <p></p> 



```python
### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it maybe having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
```

### Question 9

Discuss how you used the visual output of your trained network's feature maps to show that it had learned to look for interesting characteristics in traffic sign images


**Answer:**

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

### Project Writeup

Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 
