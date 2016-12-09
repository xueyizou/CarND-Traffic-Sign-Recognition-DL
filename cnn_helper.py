import tensorflow as tf

def _weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def _bias_variable(shape, name):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial, name=name)

def _conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def _max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')


def inference(features, conv1_out=32, conv2_out=64, fc1_out=128, keep_prob=1.0):
  """Build the model up to where it may be used for inference.

  Args:
    features: Images placeholder.
    conv1_out: Size of the first hidden layer.
    conv2_out: Size of the second hidden layer.
    fc1_out: Size of the fully connected layer.
    keep_prob: drop_out keep probability placeholder

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  #30*30
  with tf.name_scope('conv1'):
    W_conv1 = _weight_variable([3, 3, 3, conv1_out], 'weights')
    b_conv1 = _bias_variable([conv1_out], 'biases')
    h_conv1 = tf.nn.relu(_conv2d(features, W_conv1) + b_conv1)

  #15*15
  with tf.name_scope('max_pool1'):
    h_pool1 = _max_pool_2x2(h_conv1)

  #13*13
  with tf.name_scope('conv2'):
    W_conv2 = _weight_variable([3, 3, conv1_out, conv2_out], 'weights')
    b_conv2 = _bias_variable([conv2_out], 'biases')
    h_conv2 = tf.nn.relu(_conv2d(h_pool1, W_conv2) + b_conv2)

  with tf.name_scope('flatten'):
    h_flatten = tf.reshape(h_conv2, [-1, 13*13*conv2_out])

  with tf.name_scope('drop_out'):
    h_drop = tf.nn.dropout(h_flatten, keep_prob)

  with tf.name_scope('fully_connect1'):
    W_fc1 = _weight_variable([13 * 13 * conv2_out, fc1_out], 'weights')
    b_fc1 = _bias_variable([fc1_out], 'biases')
    h_fc1 = tf.nn.relu(tf.matmul(h_drop, W_fc1) + b_fc1)

  with tf.name_scope('softmax_linear'):
    W_fc2 = _weight_variable([fc1_out, 43], 'weights')
    b_fc2 = _bias_variable([43], 'biases')
    logits = tf.matmul(h_fc1, W_fc2) + b_fc2

  return logits

def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, 1-hot - [batch_size, NUM_CLASSES].

  Returns:
    loss: Loss tensor of type float.
  """
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss

def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, 1-hot - [batch_size, NUM_CLASSES].
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  correct_prediction = tf.equal(tf.argmax(logits,1), labels)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct_prediction, tf.int32))

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



def do_eval(X, y, batch_size, sess, loss_op, evaluation_op, features, labels, keep_prob_ph):
    total_loss = 0
    total_correct= 0
    for offset in range(0, X.shape[0], batch_size):
        end = offset + batch_size
        X_batch = X[offset:end]
        y_batch = y[offset:end]

        loss, correct_count = sess.run([loss_op, evaluation_op], feed_dict={features: X_batch, labels: y_batch, keep_prob_ph:1.0})
        total_loss += (loss * X_batch.shape[0])
        total_correct += correct_count

    return total_loss/X.shape[0], total_correct/X.shape[0]
