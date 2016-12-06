import numpy as np
import tensorflow as tf

def _weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name) 

def _bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

def _conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def _max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

  
def get_next_batch_train(X_train, y_train_one_hot, batchsize=256):
  indices = np.random.randint(0,X_train.shape[0],batchsize)
  Xs = X_train[indices]
  ys = y_train_one_hot[indices]
  return (Xs, ys)   
  
def inference(images_ph, hidden1_units=32, hidden2_units=64, fc_units=1024, keep_prob_ph=1.0):
  """Build the model up to where it may be used for inference.

  Args:
    images_ph: Images placeholder.
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.
    fc_units: Size of the fully connected layer.
    keep_prob_ph: drop_out keep probability placeholder

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  with tf.name_scope('hidden1'):
    W_conv1 = _weight_variable([5, 5, 1, hidden1_units], 'weights')
    b_conv1 = _bias_variable([hidden1_units], 'biases')  
    h_conv1 = tf.nn.tanh(_conv2d(images_ph, W_conv1) + b_conv1)
    h_pool1 = _max_pool_2x2(h_conv1)
    
  with tf.name_scope('hidden2'):  
    W_conv2 = _weight_variable([5, 5, hidden1_units, hidden2_units], 'weights')
    b_conv2 = _bias_variable([hidden2_units], 'biases')  
    h_conv2 = tf.nn.tanh(_conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = _max_pool_2x2(h_conv2)
  
  
  with tf.name_scope('fully_connect1'):
    W_fc1 = _weight_variable([8 * 8 * hidden2_units+8*8*hidden1_units, fc_units], 'weights')
    b_fc1 = _bias_variable([fc_units], 'biases')  
    h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*hidden2_units])
    h_pool1_pool = _max_pool_2x2(h_pool1)
    h_pool1_pool_flat = tf.reshape(h_pool1_pool, [-1, 8*8*hidden1_units])
    h_combine = tf.concat(1, [h_pool2_flat,h_pool1_pool_flat])
    h_fc1 = tf.nn.tanh(tf.matmul(h_combine, W_fc1) + b_fc1)
  
#    W_fc1 = weight_variable([8 * 8 * hidden2_units, fc_units], 'weights')
#    b_fc1 = bias_variable([fc_units], 'biases')  
#    h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*hidden2_units])
#    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  with tf.name_scope('drop_out'): 
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_ph)
  
  with tf.name_scope('softmax_linear'):
    W_fc2 = _weight_variable([fc_units, 43], 'weights')
    b_fc2 = _bias_variable([43], 'biases')  
    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
 
  return logits


def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, 1-hot - [batch_size, NUM_CLASSES].

  Returns:
    loss: Loss tensor of type float.
  """
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss


def training(loss, learning_rate):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.scalar_summary('loss', loss)
  # Create the gradient descent optimizer with the given learning rate.
#  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, 1-hot - [batch_size, NUM_CLASSES].
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct_prediction, tf.int32))


def do_eval(sess,
            evaluation,
            images_placeholder,
            labels_placeholder,
            keep_prob_placeholder,
            data_set,
            label_set,
            batch_size):
  """Runs one evaluation against the full train/test of data.

  Args:
    sess: The session in which the model has been trained.
    evaluation: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    keep_prob_placeholder: The drop-out probability placeholder
    data_set: The set of images to evaluate. 
    label_set: The set of labels to evaluate.
    batch_size: size of images to be evaluated at one step.
  """
  true_count = 0  # Counts the number of correct predictions.
  num_examples = data_set.shape[0]
  steps = int(num_examples/batch_size)  
  for i in range(steps):
    feed_dict={images_placeholder: data_set[i*batch_size:(i+1)*batch_size], labels_placeholder: label_set[i*batch_size:(i+1)*batch_size], keep_prob_placeholder: 1.0}
    true_count += sess.run(evaluation, feed_dict=feed_dict)
  
  feed_dict={images_placeholder: data_set[steps*batch_size:num_examples], labels_placeholder: label_set[steps*batch_size:num_examples], keep_prob_placeholder: 1.0}
  true_count += sess.run(evaluation, feed_dict=feed_dict)
  accuracy = float(true_count) / num_examples
  print('  Num examples: %d  Num correct: %d  Accuracy @ 1: %0.04f' %
        (num_examples, true_count, accuracy))


