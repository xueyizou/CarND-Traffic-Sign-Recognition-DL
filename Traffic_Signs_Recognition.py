# -*- coding: utf-8 -*-
# Load pickled data
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import os
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
import preprocessor
#to gray scale
X_train = preprocessor.to_grayscale(X_train)
X_test = preprocessor.to_grayscale(X_test)

#augment the training data using translation and rotation
X_train, y_train = preprocessor.augment_data(X_train, y_train)
n_train = X_train.shape[0]
print("!!!!!Number of augmented training examples =", n_train)

##normalization
X_train= preprocessor.normalize(X_train)
X_test = preprocessor.normalize(X_test)

##one-hot encoding
y_train_one_hot = preprocessor.one_hot_encoding(y_train,n_classes)
y_test_one_hot = preprocessor.one_hot_encoding(y_test,n_classes)

#%%
def run_training():
  x = tf.placeholder("float", shape=[None, X_train.shape[1], X_train.shape[2], X_train.shape[3]])
  y_ = tf.placeholder("float", shape=[None, n_classes])
  keep_prob = tf.placeholder("float")
  # Build a Graph that computes predictions from the inference model.
  logits = cnn_helper.inference(x,hidden1_units=32, hidden2_units=64, fc_units=1024, keep_prob_ph =keep_prob)

  # Add to the Graph the Ops for loss calculation.
  loss_op= cnn_helper.loss(logits, y_)

  # Add to the Graph the Ops that calculate and apply gradients.
  train_op = cnn_helper.training(loss_op, 1e-4)

  # Add the Op to compare the logits to the labels during evaluation.
  eval_op = cnn_helper.evaluation(logits, y_)

  # Build the summary Tensor based on the TF collection of Summaries.
  summary = tf.merge_all_summaries()

  # Add the variable initializer Op.
  init = tf.global_variables_initializer()

  # Create a saver for writing training checkpoints.
  saver = tf.train.Saver()

  # Create a session for running Ops on the Graph.
  sess = tf.Session()

  # Instantiate a SummaryWriter to output summaries and the Graph.
  summary_writer = tf.train.SummaryWriter("./log", sess.graph)

  # And then after everything is built:

  # Run the Op to initialize the variables.
  sess.run(init)

  batch_size=256
  max_steps =20000
  # Start the training loop.
  for step in range(max_steps):
    start_time = time.time()
    # Fill a feed dictionary with the actual set of images and labels
    # for this particular training step.
    batch = cnn_helper.get_next_batch_train(X_train, y_train_one_hot, batch_size)
    feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}

    # Run one step of the model.  The return values are the activations
    # from the `train_op` (which is discarded) and the `loss` Op.  To
    # inspect the values of your Ops or variables, you may include them
    # in the list passed to sess.run() and the value tensors will be
    # returned in the tuple from the call.
    _, loss_value = sess.run([train_op, loss_op],feed_dict=feed_dict)

    duration = time.time() - start_time

    # Write the summaries and print an overview fairly often.
    if step % 100 == 0:
      # Print status to stdout.
      print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
      # Update the events file.
      summary_str = sess.run(summary, feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, step)
      summary_writer.flush()

    # Save a checkpoint 
    if (step + 1) % 1000 == 0 or (step + 1) == max_steps:
      checkpoint_file = os.path.join("./log/", 'model.ckpt')
      saver.save(sess, checkpoint_file, global_step=step)
      
    # Evaluate the model periodically.
    if (step + 1) % 2000 == 0 or (step + 1) == max_steps:
      print('Training Data Eval:')
      cnn_helper.do_eval(sess, eval_op, x, y_, keep_prob, X_train[0:39209], y_train_one_hot[0:39209], 1000)
      # Evaluate against the test set.
      print('Test Data Eval:')
      cnn_helper.do_eval(sess, eval_op, x, y_, keep_prob, X_test,  y_test_one_hot, 1000)
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
  
  X = np.empty(shape=[5,32,32,1])
  for i in range(5):      
    img = imresize(imgs[i], (32,32))
    img_gray = preprocessor.to_grayscale(img)
    img_gray_normalized = preprocessor.normalize(img_gray)
    X[i] = img_gray_normalized
  
  x = tf.placeholder("float", [None, 32, 32, 1])
  keep_prob = tf.placeholder("float")
  # Build a Graph that computes predictions from the inference model.
  logits = cnn_helper.inference(x,hidden1_units=32, hidden2_units=64, fc_units=1024, keep_prob_ph =keep_prob)
  probs = tf.nn.softmax(logits)
  top_k_op = tf.nn.top_k(probs, 5)

  saver = tf.train.Saver()
  
  with tf.Session() as sess:
    saver.restore(sess, "./log_model/model.ckpt-19999")
    print("Model restored")
    values, indices = sess.run(top_k_op, feed_dict={x:X, keep_prob:1})
    
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
mode = "train"  #"predict" "train"
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
    
                
        

