import tensorflow as tf

with tf.GradientTape() as tape:  
    x  = tf.Variable([[3, 3, 3], [3, 3, 3]], dtype=float, shape=(2, 3))  
    tape.watch(x)  
    y = tf.cast(x*x, dtype=float) 
    flat_y = tf.reshape(y, (-1,))
    
    

def _random_vector(C, B):
      # creates a random vector of dimension C with a norm of C^(1/2)
        if C == 1:
          return tf.ones((B,))
        v = tf.random.uniform((B, C))
        vnorm = tf.linalg.norm(v, axis=1, keepdims=True)
        return tf.math.divide(v, vnorm)

vec = _random_vector(3, 2) # vector of the vector-Jacobian product
flat_vec = tf.reshape(vec, (-1, ))

grads = tape.gradient(flat_y,x, output_gradients=flat_vec)
print(grads) # prints the vector-Jacobian product, [4.,12.,36.]
print(tf.linalg.norm(vec, axis=1, keepdims=True))