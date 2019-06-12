import tensorflow as tf


def attention(inputs, attention_size, time_major=False, return_alphas=False):
    #    u = tf.layers.dense(inputs, 
    #                         attention_size, 
    #                         activation=tf.tanh, 
    #                         use_bias=True, 
    #                         bias_initializer=tf.zeros_initializer())    

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.transpose(inputs, perm=[1, 0, 2])
        
    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer


    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    q_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    #b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    
    
    # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
    #  the shape of `v` is (B,T,D)*(D,A)*(B,T,D)*(D,A)=(B,T,T), where A=attention_size
    u = tf.tanh(tf.matmul(tf.tensordot(inputs, q_omega, axes=1),tf.tensordot(inputs, w_omega, axes=1),transpose_b=True))        
    # For each of the timestamps its vector of size A from `u` is reduced with `u_omega` vector
    #uu = tf.tensordot(u, u_omega, axes=1, name='uu')  # (B,T,A)*(A)=(B,T) shape
    alphas = tf.nn.softmax(u, name='alphas')         # (B,T,T) shape
    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,T,D) * (B,T,T) = (B,T,D) shape
    output = tf.reduce_sum(inputs* tf.expand_dims(alphas, -1), 1)


    if not return_alphas:
        return output
    else:
    	return output, alphas
