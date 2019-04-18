import tensorflow as tf
from base_nn import BaseNN
import os


def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1],
                                                  padding='SAME'),b), name=name)

def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME', name=name)

def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

def init_bias(shape, name):
    return tf.Variable(tf.random_normal(shape), name=name)


class ConvNet(BaseNN):
    def construct(self):
        """
        Construction phase. It defines the operations of the network

        """
        # Placeholders for input, output and dropout
        input_x = tf.placeholder(tf.float32, [None, self.parameters['width'], self.parameters['height']],
                                 name="input_x")
        input_y = tf.placeholder(tf.float32, [None, self.parameters['num_classes']], name="input_y")

        # Stores the probability of keeping a neuron in the dropout layer
        dropout_hidden_keep_prob = tf.placeholder(tf.float32, name="dropout_hidden_keep_prob")

        img_expanded = tf.reshape(input_x, shape=[-1, self.parameters['width'], self.parameters['height'], 1])
        print "Input IMG: {}".format(input_x)
        print "IMG expanded: {}".format(img_expanded)


        with tf.name_scope("img-conv-maxpool-%s" % self.parameters['filter_sizes'][0]):
            filter_shape = [self.parameters['filter_sizes'][0],
                            self.parameters['filter_sizes'][0],
                            1,
                            self.parameters['num_filters'][0]]
            W_1_img = tf.get_variable("W1_img", shape=filter_shape,
                                      initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b_1_img = init_bias([self.parameters['num_filters'][0]], "b")

            print "IMG W l1: {}".format(W_1_img)
            print "IMG b l1: {}".format(b_1_img)

            # Convolution Layer
            conv1_img = conv2d('conv1', img_expanded, W_1_img, b_1_img)
            print "IMG Conv 1: {}".format(conv1_img)

            # Max Pooling (down-sampling)
            pool1_img = max_pool('pool1', conv1_img, k=2)
            print "IMG Pooling 1: {}".format(pool1_img)

            # Apply Normalization
            norm1_img = norm('norm1', pool1_img, lsize=4)
            print "IMG Norm 1: {}".format(norm1_img)

        with tf.name_scope("img-conv-maxpool-%s" % self.parameters['filter_sizes'][1]):
            filter_shape = [self.parameters['filter_sizes'][1],
                            self.parameters['filter_sizes'][1],
                            self.parameters['num_filters'][0],
                            self.parameters['num_filters'][1]]
            W_2_img = tf.get_variable("W2_img", shape=filter_shape,
                                      initializer=tf.contrib.layers.xavier_initializer_conv2d())

            b_2_img = init_bias([self.parameters['num_filters'][1]], "b")
            print "IMG W l2: {}".format(W_2_img)
            print "IMG b l2: {}".format(b_2_img)

            # Apply dropout
            # drop1_img = tf.nn.dropout(norm1_img, dropout_hidden_keep_prob)
            # Convolution Layer
            conv2_img = conv2d('conv2', norm1_img, W_2_img, b_2_img)
            print "IMG Conv 2: {}".format(conv2_img)

            # Max Pooling (down-sampling)
            pool2_img = max_pool('pool2', conv2_img, k=2)
            print "IMG Pooling 2: {}".format(pool2_img)

            # Apply Normalization
            norm2_img = norm('norm2', pool2_img, lsize=4)
            print "IMG Norm 2: {}".format(norm2_img)

        with tf.name_scope("img-conv-maxpool-%s" % self.parameters['filter_sizes'][2]):
            filter_shape = [self.parameters['filter_sizes'][2],
                            self.parameters['filter_sizes'][2],
                            self.parameters['num_filters'][1],
                            self.parameters['num_filters'][2]]
            W_3_img = tf.get_variable("W3_img", shape=filter_shape,
                                      initializer=tf.contrib.layers.xavier_initializer_conv2d())

            b_3_img = init_bias([self.parameters['num_filters'][2]], "b")
            print "IMG W l2: {}".format(W_3_img)
            print "IMG b l2: {}".format(b_3_img)

            # Apply dropout
            # drop2_img = tf.nn.dropout(norm2_img, dropout_hidden_keep_prob)
            # Convolution Layer
            conv3_img = conv2d('conv3', norm2_img, W_3_img, b_3_img)
            print "IMG Conv 3: {}".format(conv3_img)

            # Max Pooling (down-sampling)
            pool3_img = max_pool('pool3', conv3_img, k=2)
            print "IMG Pooling 3: {}".format(pool3_img)

            # Apply Normalization
            norm3_img = norm('norm3', pool3_img, lsize=4)
            print "IMG Norm 3: {}".format(norm3_img)

        with tf.name_scope("img-fully-connected-layer"):
            W_4_img = tf.get_variable("W4_img",
                                         shape=[int(norm3_img.shape[1]) * int(norm3_img.shape[2]) * int(norm3_img.shape[3]),
                              self.parameters["hidden_neurons"]],
                                      initializer=tf.contrib.layers.xavier_initializer())

            b_4_img = init_bias([self.parameters["hidden_neurons"]], "b")
            print "IMG W fully-connected layer: {}".format(W_4_img)
            print "IMG b fully-connected layer: {}".format(b_4_img)

            drop3_img = tf.nn.dropout(norm3_img, dropout_hidden_keep_prob)
            dense1_img = tf.reshape(drop3_img, [-1, W_4_img.get_shape().as_list()[0]])
            # Relu activation
            dense1_img = tf.nn.relu(tf.add(tf.matmul(dense1_img, W_4_img), b_4_img))
            print "IMG Fully-connected ReLU: {}".format(dense1_img)

        with tf.name_scope("output"):
            W_output = init_weights([self.parameters["hidden_neurons"], self.parameters["num_classes"]], "W")
            b_output = init_bias([self.parameters["num_classes"]], "b")
            print "W output layer: {}".format(W_output)
            print "b output layer: {}".format(b_output)

            # Apply dropout
            drop_output = tf.nn.dropout(dense1_img, dropout_hidden_keep_prob)
            # Output, class prediction
            scores = tf.nn.xw_plus_b(drop_output, W_output, b_output, name="scores")
            print "Scores: {}".format(scores)
            predictions = tf.argmax(scores, 1, name="predictions")
            softmax_probabilities = tf.nn.softmax(scores, name="probabilities")

        # Loss and accuracy
        # Measures the error that our network makes. Cross-entropy loss
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss_and_accuracy"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=input_y)
            loss = tf.reduce_mean(losses, name="loss")

            correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # Training procedure
        with tf.name_scope("train"):
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(self.parameters['learning_rate'], name="Adam")
            grads_and_vars = optimizer.compute_gradients(loss)

            if self.parameters['gradient_clipping'] == True:
                capped_gvs = [
                    (tf.clip_by_value(grad, self.parameters['min_gradient'], self.parameters['max_gradient']), var)
                    for
                    grad, var in
                    grads_and_vars]
                train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step,
                                                     name="train_op")
            else:
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step,
                                                     name="train_op")

        with tf.name_scope("summaries"):
            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries, name="grad_summaries_merged")

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", loss)
            acc_summary = tf.summary.scalar("accuracy", accuracy)
            acc_summary = tf.summary.scalar("accuracy", accuracy)

            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged],
                                                name="train_summary_op")
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary],
                                              name="dev_summary_op")

    def predict(self, img_raw):
        input_x = self.network_graph.get_operation_by_name("input_x").outputs[0]
        dropout_hidden_keep_prob = self.network_graph.get_operation_by_name("dropout_hidden_keep_prob").outputs[0]

        predictions = self.network_graph.get_operation_by_name("output/predictions").outputs[0]
        probabilities = self.network_graph.get_operation_by_name("output/probabilities").outputs[0]

        ypred, probs = self.network_session.run([predictions, probabilities], {input_x: img_raw,
                                                                               dropout_hidden_keep_prob: 1.0})

        return ypred[0], probs[0]



if __name__ == "__main__":
    model = ConvNet(os.path.dirname(
        os.path.abspath(__file__)) + "/parameters/parameters_cnn_3c_128x128.json")
    model.init_directories("CNN_IMG_3C_128x128")
    model.init_graph()

    tfrecords_filepath = "/path/to/tfrecords/"
    model.train([tfrecords_filepath + "training0.tfrecords",
                 tfrecords_filepath + "training1.tfrecords",
                 tfrecords_filepath + "training2.tfrecords",
                 tfrecords_filepath + "training3.tfrecords",
                 tfrecords_filepath + "training4.tfrecords",
                 tfrecords_filepath + "training5.tfrecords",
                 tfrecords_filepath + "training6.tfrecords",
                 tfrecords_filepath + "training7.tfrecords",
                 tfrecords_filepath + "training8.tfrecords"
                 ],
                [
                    tfrecords_filepath + "training9.tfrecords"])

