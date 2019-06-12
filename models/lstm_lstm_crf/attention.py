import tensorflow as tf


def attention(inputs, nwords,attention_size, time_major=False, return_alphas=False):
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
    
    TensorArr = tf.TensorArray(tf.int32, 1, dynamic_size=True, infer_shape=False)
    b = TensorArr.unstack(inputs)
    d = []
    for i in range(0,20):
        d.append(b.read(i))
    s_inputs=[]    
    time = tf.reduce_max(nwords)
    with tf.Session() as sess:
        time = sess.run(time)
    for j in range(time):
        k = []
        for i in range (0,20):
            k.append(d[i][j])
            k = tf.stack(k, axis=0)
            k = tf.expand_dims(k, -2)
            s_inputs.append(k)
    
    # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
    #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
    outputs = []
    for s_i in s_inputs:
        u = tf.tanh(tf.tensordot(tf.expand_dims(s_i, -2), q_omega, axes=1)+tf.tensordot(inputs, w_omega, axes=1))        
        # For each of the timestamps its vector of size A from `u` is reduced with `u_omega` vector
        uu = tf.tensordot(u, u_omega, axes=1, name='uu')  # (B,T,A)*(A)=(B,T) shape
        alphas = tf.nn.softmax(uu, name='alphas')         # (B,T) shape
        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs* tf.expand_dims(alphas, -1), 1)
        outputs.append(output)
    outputs = tf.stack(outputs, axis=1)

    if not return_alphas:
        return outputs
    else:
    	return outputs, alphas
