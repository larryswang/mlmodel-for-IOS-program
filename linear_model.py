import os
import tensorflow as tf
import numpy as np

def linear_model(x):
    # x is the time series input
    inter = tf.layers.dense(inputs = x, units = 1)
    pred = tf.nn.softmax(inter)
    return pred

def train():
    # generate toy data, it is 100 sets of data, each consists of 6 data
    input_data = np.ones((100,6))
    input_label = np.ones((100,1))

    # instantiate the model in the default graph
    x = tf.placeholder(tf.float32, [None, 6])
    print 'image_input: ', x
    pred = linear_model(x)
    print 'pred output:', pred
  
    # Add training components to it
    # 1 set of data goes to 1 of 5 labels
    y = tf.placeholder(tf.float32, [None, 1])
  
    # Define training hyper-parameters
    learning_rate = 0.01
    training_epochs = 25
    batch_size = 20
    display_step = 1
  
    # Define Cross Entropy loss
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
    # Use Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
  
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
  
    # Use a saver to save checkpoints
    saver = tf.train.Saver()
    # Training starts here
    with tf.Session() as sess:
      sess.run(init)
      # Training cycle
      for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = 5
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = input_data[i * batch_size: (i+1) * batch_size, :]
            batch_ys = input_label[i * batch_size: (i+1) * batch_size, :]
          
            # Fit training using batch data
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                         y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
      
      print 'Training Done. Now save the checkpoint...'
      save_dir = './checkpoints'
      save_path = os.path.join(save_dir, 'model.ckpt')
      if not os.path.exists(save_dir):
        os.mkdir(save_dir)
      save_path = saver.save(sess, save_path)
      tf.train.write_graph(sess.graph, './', 'model.pbtxt')
      print"print all variables:"
      print tf.global_variables()

if __name__ == '__main__':
    # Read the data
    train()

