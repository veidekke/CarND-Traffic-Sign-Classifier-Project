
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Build a Traffic Sign Recognition Classifier

In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n',
    '**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 

In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.

The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains 'Stand Out Suggestions' for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the 'stand out suggestions', you can include the code in this Ipython notebook and also discuss the results in the writeup file.


>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

---
## Step 0: Load The Data


```python
# Load pickled data
import pickle
import os

# Jupyter notebook directory
path = os.getcwd()
path += '/traffic-signs-data/'

training_file = path + 'train.p'
validation_file = path + 'valid.p' 
testing_file = path + 'test.p'

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
- `'sizes'` is a list containing tuples, (width, height) representing the the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**

Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas


```python
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

import numpy as np


# Number of training examples
n_train = len(X_train)

# Number of testing examples
n_test = len(X_test)

# Number of validation examples
n_valid = len(X_valid)

# What's the shape of a traffic sign image?
image_shape = X_train[0].shape

# How many unique classes/labels there are in the dataset
n_classes = len(np.unique(y_train))

print('Number of training examples =', n_train)
print('Number of testing examples =', n_test)
print('Number of validation examples =', n_valid)
print('Image data shape =', image_shape)
print('Number of classes =', n_classes)
```

    Number of training examples = 34799
    Number of testing examples = 12630
    Number of validation examples = 4410
    Image data shape = (32, 32, 3)
    Number of classes = 43


### Include an exploratory visualization of the dataset

Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc.

The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.

**NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections.


```python
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import random
# Visualizations will be shown in the notebook.
%matplotlib inline

# Read sign names from CSV file
from pandas.io.parsers import read_csv
signnames = read_csv('signnames.csv').values[:, 1]

# Display random image and its label
index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.xlabel(signnames[y_train[index]])
plt.imshow(image)
```




    <matplotlib.image.AxesImage at 0x110aaa358>




![png](output_8_1.png)



```python
# Count traffic sign categories in dataset

data = np.zeros(43)

for i in range(len(X_train)):
    data[y_train[i]] += 1
```


```python
# Plot bar chart of traffic sign counts in dataset

n = 43

index = np.arange(n)
width = 1

fig, ax = plt.subplots(figsize=(20, 10))
ax.bar(index, data, width)

ax.set_ylabel('Count')
ax.set_xlabel('Traffic Sign Category')
ax.set_title('Counts of traffic signs in dataset')
ax.set_xticks(index + width / 2)
ax.set_xticklabels(signnames, rotation=90)

plt.show()
```


![png](output_10_0.png)


----

## Step 2: Design and Test a Model Architecture

Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

There are various aspects to consider when thinking about this problem:

- Neural network architecture
- Play around preprocessing techniques (normalization, rgb to grayscale, etc)
- Number of examples per label (some have more than others).
- Generate fake data.

Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

**NOTE:** The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play!

### Pre-process the Data Set (normalization, grayscale, etc.)

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.


```python
### Preprocess the data here. Preprocessing steps could include normalization, converting to grayscale, etc.
### Feel free to use as many code cells as needed.


# Convert to grayscale    
X_train = np.sum(X_train/3, axis=3, keepdims=True)
X_valid = np.sum(X_valid/3, axis=3, keepdims=True)
X_test = np.sum(X_test/3, axis=3, keepdims=True)

# Normalize to values between 0.0 and 1.0
X_train = (X_train - 127) / 255 
X_valid = (X_valid - 127) / 255
X_test = (X_test - 127) / 255
```


```python
# Visualize the preprocessed images

# Display random image and its label
index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.xlabel(signnames[y_train[index]])
plt.imshow(image, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x113339748>




![png](output_15_1.png)


### Model Architecture


```python
import tensorflow as tf
from sklearn.utils import shuffle


X_train, y_train = shuffle(X_train, y_train)

EPOCHS = 200
BATCH_SIZE = 128
```


```python
### Define your architecture here.
### Feel free to use as many code cells as needed.

from tensorflow.contrib.layers import flatten

def Sermanet(x):    
    
    # ------ Layer 1 -----
    
    # Convolutional. Input = 32x32x1. Output = 32x32x12
    conv1_W = tf.get_variable('conv1_W', shape=(3, 3, 1, 12), initializer = tf.contrib.layers.xavier_initializer())
    conv1_b = tf.Variable(tf.zeros(12))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b

    # Activation
    conv1 = tf.nn.relu(conv1)
    # Pooling. Input = 32x32x12. Output = 16x16x12
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    
    # ------ Layer 2 -----
    
    # Convolutional. Input = 16x16x12. Output = 12x12x16
    conv2_W = tf.get_variable('conv2_W', shape=(5, 5, 12, 16), initializer = tf.contrib.layers.xavier_initializer())
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # Activation
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 12x12x16. Output = 6x6x16
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    conv2 = tf.nn.dropout(conv2, keep_prob)
    
    
    # ------ Layer 3 -----
    
    # Convolutional. Input = 6x6x16. Output = 2x2x32
    conv3_W = tf.get_variable('conv3_W', shape=(5, 5, 16, 32), initializer = tf.contrib.layers.xavier_initializer())
    conv3_b = tf.Variable(tf.zeros(32))
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
    
    # Activation
    conv3 = tf.nn.relu(conv3)
    
    # Pooling. Input = 2x2x32. Output = 1x1x32
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # Dropout
    conv3 = tf.nn.dropout(conv3, keep_prob)
    
    
    # ------ Concat -----
    
    # 1. layer output
    # Pooling & flatten. Input = 16x16x12. Output = 768
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    pool1 = flatten(pool1)
    
    # 2. layer output
    # Pooling & flatten. Input = 6x6x16. Output = 144
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    pool2 = flatten(pool2)
    
    # 3. layer output
    # Flatten. Input = 1x1x32. Output = 32
    conv3 = flatten(conv3)
    
    # Concatenate. Input = 768, 144, 32. Output = 944
    flat = tf.concat(1, [pool1, pool2, conv3])
    
    
    # ------ Layer 4 -----
    
    # Fully Connected + Relu. Input = 944. Output = 472
    fc1_W = tf.get_variable('fc1_W', shape=(944, 472), initializer = tf.contrib.layers.xavier_initializer())
    fc1_b = tf.Variable(tf.zeros(472))
    fc1 = tf.matmul(flat, fc1_W) + fc1_b    
    fc1 = tf.nn.relu(fc1)
    
    # Dropout
    fc1 = tf.nn.dropout(fc1, keep_prob)
    
    
    # ------ Layer 5 -----
    
    # Fully Connected. Input = 472. Output = 43
    with tf.variable_scope('fc2'):
        fc2_W = tf.get_variable('fc2_W', shape=(472, 43), initializer = tf.contrib.layers.xavier_initializer())
    fc2_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc1, fc2_W) + fc2_b
    return logits
```

### Train, Validate and Test the Model

A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

#### Setup Features


```python
tf.reset_default_graph() 

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

keep_prob = tf.placeholder(tf.float32) # probability to keep units
```

#### Training Pipeline


```python
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

rate = 0.0002
l2_rate = 0.01

logits = Sermanet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
with tf.variable_scope('fc2', reuse=True):
    l2_regularization_loss = tf.nn.l2_loss(tf.get_variable('fc2_W'))
loss_operation = tf.reduce_mean(cross_entropy) + l2_rate * l2_regularization_loss
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
```

#### Model Evaluation


```python
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
```

#### Model Training


```python
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    try:
        tf.train.Saver().restore(sess, './lenet')
    except Exception as e:
        print('Error restoring previous model: ', e)
        pass
    
    print('Training...\n')
    
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        training_accuracy = evaluate(X_train, y_train)
        print('EPOCH {} ...'.format(i+1))
        print('Validation Accuracy = {:.3f}'.format(validation_accuracy))
        print('Training Accuracy = {:.3f}\n'.format(training_accuracy))
        
        if validation_accuracy > 0.955:
            break
        if i % 3 == 0:
            saver.save(sess, './lenet')
            print('Model saved')
        
    saver.save(sess, './lenet')
    print('Model saved\n')
```

    Training...
    



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-266-f7dd9f8ab95d> in <module>()
         17             end = offset + BATCH_SIZE
         18             batch_x, batch_y = X_train[offset:end], y_train[offset:end]
    ---> 19             sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
         20 
         21         validation_accuracy = evaluate(X_valid, y_valid)


    /Users/B/miniconda3/envs/carnd-term1/lib/python3.5/site-packages/tensorflow/python/client/session.py in run(self, fetches, feed_dict, options, run_metadata)
        764     try:
        765       result = self._run(None, fetches, feed_dict, options_ptr,
    --> 766                          run_metadata_ptr)
        767       if run_metadata:
        768         proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)


    /Users/B/miniconda3/envs/carnd-term1/lib/python3.5/site-packages/tensorflow/python/client/session.py in _run(self, handle, fetches, feed_dict, options, run_metadata)
        962     if final_fetches or final_targets:
        963       results = self._do_run(handle, final_targets, final_fetches,
    --> 964                              feed_dict_string, options, run_metadata)
        965     else:
        966       results = []


    /Users/B/miniconda3/envs/carnd-term1/lib/python3.5/site-packages/tensorflow/python/client/session.py in _do_run(self, handle, target_list, fetch_list, feed_dict, options, run_metadata)
       1012     if handle is None:
       1013       return self._do_call(_run_fn, self._session, feed_dict, fetch_list,
    -> 1014                            target_list, options, run_metadata)
       1015     else:
       1016       return self._do_call(_prun_fn, self._session, handle, feed_dict,


    /Users/B/miniconda3/envs/carnd-term1/lib/python3.5/site-packages/tensorflow/python/client/session.py in _do_call(self, fn, *args)
       1019   def _do_call(self, fn, *args):
       1020     try:
    -> 1021       return fn(*args)
       1022     except errors.OpError as e:
       1023       message = compat.as_text(e.message)


    /Users/B/miniconda3/envs/carnd-term1/lib/python3.5/site-packages/tensorflow/python/client/session.py in _run_fn(session, feed_dict, fetch_list, target_list, options, run_metadata)
       1001         return tf_session.TF_Run(session, options,
       1002                                  feed_dict, fetch_list, target_list,
    -> 1003                                  status, run_metadata)
       1004 
       1005     def _prun_fn(session, handle, feed_dict, fetch_list):


    KeyboardInterrupt: 



```python
with tf.Session() as sess:
    try:
        tf.train.Saver().restore(sess, './lenet')
    except Exception as e:
        print('Error restoring previous model: ', e)
        pass
    
    print('Training Accuracy = {:.3f}'.format(evaluate(X_train, y_train)))
    print('Validation Accuracy = {:.3f}'.format(evaluate(X_valid, y_valid)))
    print('Test Accuracy = {:.3f}'.format(evaluate(X_test, y_test)))
```

    Training Accuracy = 1.000
    Validation Accuracy = 0.951
    Test Accuracy = 0.944


## ---

## Step 3: Test a Model on New Images

To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

### Load and Output the Images


```python
### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import matplotlib.image as mpimg


# Expected labels
expected_labels = [13, 19, 1, 9, 12]

# Number of test images
test_image_count = 5

test_images = []

_, axs = plt.subplots(1, test_image_count)

# Load and plot images
for i in range(test_image_count):
    image = mpimg.imread('signs/sign{}.jpg'.format(i + 1))
    test_images.append(image)
    axs[i].set_xlabel(signnames[expected_labels[i]], rotation=90)
    axs[i].imshow(image)

test_images = np.asarray(test_images)
```


![png](output_32_0.png)


### Predict the Sign Type for Each Image


```python
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

# Preprocessing

# Convert to grayscale    
test_images = np.sum(test_images / 3, axis=3, keepdims=True)

# Normalize to values between 0.0 and 1.0
test_images = (test_images - 127) / 255 
```


```python
# Calculate (top 5) predictions 
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    try:
        tf.train.Saver().restore(sess, './lenet')
    except Exception as e:
        print('Error restoring previous model: ', e)
        pass

    softmax = sess.run(tf.nn.softmax(logits), feed_dict={x: test_images, keep_prob: 1.0})
    top_k = sess.run(tf.nn.top_k(softmax, 5), feed_dict={x: test_images, keep_prob: 1.0})
    
for i in range(test_image_count):
    print('Predicted sign: {} [{:.2f}% certainty]'.format(signnames[top_k[1][i][0]], top_k[0][i][0] * 100))
    print('Expected sign: ', signnames[expected_labels[i]], '\n')
```

    Predicted sign: Yield [100.00% certainty]
    Expected sign:  Yield 
    
    Predicted sign: Dangerous curve to the left [100.00% certainty]
    Expected sign:  Dangerous curve to the left 
    
    Predicted sign: Speed limit (30km/h) [77.89% certainty]
    Expected sign:  Speed limit (30km/h) 
    
    Predicted sign: No passing [100.00% certainty]
    Expected sign:  No passing 
    
    Predicted sign: Priority road [100.00% certainty]
    Expected sign:  Priority road 
    


### Analyze Performance


```python
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    try:
        tf.train.Saver().restore(sess, './lenet')
    except Exception as e:
        print('Error restoring previous model: ', e)
        pass

    new_image_accuracy = evaluate(test_images, expected_labels)
    print('New image accuracy = {:.3f}\n'.format(new_image_accuracy))
```

    New image accuracy = 1.000
    


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

index = np.arange(5)
width = 1

for i in range(test_image_count):
    _, axs = plt.subplots(1, 2, figsize=(6, 2))
    axs[0].set_xlabel(signnames[expected_labels[i]], rotation=90)
    axs[0].imshow(test_images[i].squeeze(), cmap='gray')
    
    axs[1].bar(index, top_k[0][i], width)
    axs[1].set_ylabel('Certainty')
    axs[1].set_xlabel('Traffic Sign Category')
    axs[1].set_title('Top 5 predictions')
    axs[1].set_xticks(index + width / 2)
    axs[1].set_xticklabels(signnames[top_k[1][i]], rotation=90)
```


![png](output_40_0.png)



![png](output_40_1.png)



![png](output_40_2.png)



![png](output_40_3.png)



![png](output_40_4.png)



```python
# Load and plot images of signs that the model mistakingly predicted for new sign no. 3
_, axs = plt.subplots(1, 4)

axs[0].set_xlabel(signnames[y_train[47]], rotation=90)
axs[0].imshow(X_train[47].squeeze(), cmap='gray')

axs[1].set_xlabel(signnames[y_train[25]], rotation=90)
axs[1].imshow(X_train[25].squeeze(), cmap='gray')

axs[2].set_xlabel(signnames[y_train[6]], rotation=90)
axs[2].imshow(X_train[6].squeeze(), cmap='gray')

axs[3].set_xlabel(signnames[y_train[4]], rotation=90)
axs[3].imshow(X_train[4].squeeze(), cmap='gray')
```




    <matplotlib.image.AxesImage at 0x1140ff588>




![png](output_41_1.png)


---

## Step 4: Visualize the Neural Network's State with Test Images

 This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.

 Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.

For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.

<figure>
 <img src='visualize_cnn.png' width='380' alt='Combined Image' />
 <figcaption>
 <p></p> 
 <p style='text-align: center;'> Your output should look something like this (above)</p> 
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
    activation = tf_activation.eval(session=sess,feed_dict={x : [image_input], keep_prob: 1.0})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation='nearest', vmin =activation_min, vmax=activation_max, cmap='gray')
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation='nearest', vmax=activation_max, cmap='gray')
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation='nearest', vmin=activation_min, cmap='gray')
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation='nearest', cmap='gray')
    

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    try:
        tf.train.Saver().restore(sess, './lenet')
    except Exception as e:
        print('Error restoring previous model: ', e)
        pass    

    outputFeatureMap(test_images[0], logits)
            
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-283-5dd8ea347352> in <module>()
         39         pass
         40 
    ---> 41     outputFeatureMap(test_images[0], logits)
         42 


    <ipython-input-283-5dd8ea347352> in outputFeatureMap(image_input, tf_activation, activation_min, activation_max, plt_num)
         14     # If you get an error tf_activation is not defined it maybe having trouble accessing the variable from inside a function
         15     activation = tf_activation.eval(session=sess,feed_dict={x : [image_input], keep_prob: 1.0})
    ---> 16     featuremaps = activation.shape[3]
         17     plt.figure(plt_num, figsize=(15,15))
         18     for featuremap in range(featuremaps):


    IndexError: tuple index out of range


### Question 9

Discuss how you used the visual output of your trained network's feature maps to show that it had learned to look for interesting characteristics in traffic sign images


**Answer:**

Sadly, I could not get it to work.

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n',
    '**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

### Project Writeup

Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 
