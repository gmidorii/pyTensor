import tensorflow as tf

def placeholder():
    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0)
    print(node1, node2)

    session = tf.Session()
    print(session.run([node1, node2]))

    node3 = tf.add(node1, node2)
    print(node3)
    print("session run(node3)", session.run(node3))

    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b
    print(session.run(adder_node, {a: 3, b: 4.5}))
    print(session.run(adder_node, {a: [1, 3], b: [2, 5]}))

    add_and_triple = adder_node * 3
    print(session.run(add_and_triple, {a: 3, b: 4.5}))

def variable():
    sess = tf.Session()
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    
    # must initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)

    print(sess.run(linear_model, {x:[1,2,3,4]}))

    # loss function
    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)
    print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

    # assign
    fixW = tf.assign(W, [-1.])
    fixB = tf.assign(b, [1.])
    sess.run([fixW, fixB])
    print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
    
def train():
    sess = tf.Session()
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    # must initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)
    # loss function
    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)

    # optimizer
    opt = tf.train.GradientDescentOptimizer(0.01)
    train = opt.minimize(loss)
    for i in range(1000):
        sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

    print(sess.run([W, b]))

def main():
    #placeholder()
    #variable()
    train()
    
    

if __name__ == "__main__":
  main()
