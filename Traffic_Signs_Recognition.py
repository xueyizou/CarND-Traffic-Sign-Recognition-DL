# -*- coding: utf-8 -*-
# Load pickled data
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import os
from sklearn.utils import shuffle
import cnn_helper

#%%

# TODO: fill this in based on where you saved the training and testing data
training_file = 'dataset/train.p'
testing_file =  'dataset/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)


X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

#%%

### To start off let's do a basic data summary.

# TODO: number of training examples
n_train = X_train.shape[0]

# TODO: number of testing examples
n_test = X_test.shape[0]

# TODO: what's the shape of an image?
image_shape = X_train[0].shape

# TODO: how many classes are in the dataset
n_classes = np.unique(y_train).size

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

#%%
### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.some suggestions include:
### plotting traffic signs images, plotting the count of each sign

img = X_train[np.random.randint(0,n_train,15)]
img_show = np.vstack((np.hstack((img[0],img[1],img[2],img[3],img[4])),
                      np.hstack((img[5],img[6],img[7],img[8],img[9])),
                      np.hstack((img[10],img[11],img[12],img[13],img[14]))))

fig, axes = plt.subplots(nrows=2,ncols=1)
axes[0].imshow(img_show)
axes[1].hist(y_train, n_classes)
plt.show()


#%%
###preprocessing
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=1)
n_train = X_train.shape[0]
n_val = X_val.shape[0]
print("Now divide the training data as %d training items and %d validation items." % (n_train, n_val))

##normalization
#X_train= (X_train-128)/128.0
#X_val= (X_val-128)/128.0
#X_test = (X_test-128)/128.0

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255 - 0.5
X_val = X_val / 255 - 0.5
X_test = X_test / 255 - 0.5


#%%
def run_training():
  features = tf.placeholder("float", shape=[None]+ list(X_train.shape[1:]))
  labels = tf.placeholder("int64", shape=(None,))
  keep_prob = tf.placeholder("float")
  # Build a Graph that computes predictions from the inference model.
  logits = cnn_helper.inference(features,conv1_out=32,conv2_out=64, fc1_out=256, keep_prob =keep_prob)
  # Add to the Graph the Ops for loss calculation.
  loss_op= cnn_helper.loss(logits, labels)
  # Add the Op to compare the logits to the labels during evaluation.
  evaluation_op = cnn_helper.evaluation(logits, labels)
  # Add to the Graph the Ops that calculate and apply gradients.
  train_op = cnn_helper.training(loss_op, 0.01)
  # Add the variable initializer Op.
  init = tf.global_variables_initializer()

  # Build the summary Tensor based on the TF collection of Summaries.
  summary = tf.merge_all_summaries()
  # Create a saver for writing training checkpoints.
  saver = tf.train.Saver()

  # Create a session for running Ops on the Graph.
  sess = tf.Session()
  # Instantiate a SummaryWriter to output summaries and the Graph.
  summary_writer = tf.train.SummaryWriter("./log", sess.graph)
  # And then after everything is built:
  # Run the Op to initialize the variables.
  sess.run(init)

  epochs =50
  batch_size=256
  global_step=0
  checkpoint_file = os.path.join("./log/", 'model.ckpt')
  for epoch in range(epochs):
        # training
        X_train_temp, y_train_temp = shuffle(X_train, y_train)
        start_time = time.time()
        for offset in range(0, X_train_temp.shape[0], batch_size):
            end = offset + batch_size
            feed_dict={features: X_train_temp[offset:end], labels: y_train_temp[offset:end], keep_prob:0.5}
            sess.run(train_op, feed_dict=feed_dict)

            global_step +=1
            if global_step %100==0:
              summary_str = sess.run(summary, feed_dict=feed_dict)
              summary_writer.add_summary(summary_str, global_step)
              summary_writer.flush()
            if global_step %2000==0:
              saver.save(sess, checkpoint_file, global_step)


        val_loss, val_acc = cnn_helper.do_eval(X_val, y_val, batch_size, sess, loss_op, evaluation_op, features, labels, keep_prob)
        print("Epoch", epoch+1)
        print("Time: %.3f seconds" % (time.time() - start_time))
        print("Validation Loss =", val_loss)
        print("Validation Accuracy =", val_acc)
        print("==================================")

  summary_str = sess.run(summary, feed_dict=feed_dict)
  summary_writer.add_summary(summary_str, global_step)
  summary_writer.flush()
  saver.save(sess, checkpoint_file)

  test_loss, test_acc = cnn_helper.do_eval(X_test, y_test, batch_size, sess, loss_op, evaluation_op, features, labels, keep_prob)
  print("Test Loss =", test_loss)
  print("Test Accuracy =", test_acc)
  print("")
  sess.close()

#%%
def run_prediction():
  from scipy.misc import imread, imresize
  imgs = []
  fig, axes = plt.subplots(nrows=1,ncols=5)

  for i in range(5):
    path = './test_images/{0}.jpg'.format(i+1)
    img = imread(path)
    imgs.append(img)
    axes[i].imshow(img)
  plt.show()

  X = np.empty(shape=[5,32,32,3])
  for i in range(5):
    img = imresize(imgs[i], (32,32))
    img_normalized = img/255.0 -0.5
    X[i] = img_normalized

  features = tf.placeholder("float", [None, 32, 32, 3])
  keep_prob = tf.placeholder("float")
  # Build a Graph that computes predictions from the inference model.
  logits = cnn_helper.inference(features,conv1_out=32,conv2_out=64, fc1_out=256, keep_prob =keep_prob)
  probs = tf.nn.softmax(logits)
  top_k_op = tf.nn.top_k(probs, 5)

  saver = tf.train.Saver()
  path ="./log_model/model.ckpt"
  if os.path.exists("./log/model.ckpt.index"):
    path ="./log/model.ckpt"
  with tf.Session() as sess:
    saver.restore(sess, path)
    print("Model restored")
    values, indices = sess.run(top_k_op, feed_dict={features:X, keep_prob:1})

  import csv
  reader = csv.reader(open('signnames.csv', newline=''), delimiter=',')
  sign_map = {row[0]: row[1] for row in reader}

  for i in range(5):
    plt.figure(i)
    plt.imshow(np.uint8(imgs[i]))
    plt.show()
    probs = values[i]
    signs = indices[i]
    for k in range(probs.shape[0]):
      print("%.4f: %s" % (probs[k], sign_map[str(signs[k])]))


#%%
mode = "predict"  #"predict" "train"
if __name__ == '__main__':
  if mode == "train":
    if tf.gfile.Exists("./log/"):
      tf.gfile.DeleteRecursively("./log/")
    tf.gfile.MakeDirs("./log/")
    start_time = time.time()
    run_training()
    training_time = time.time() - start_time
    print("total time for training: %.2f sec" % training_time)
  else:
    run_prediction()




