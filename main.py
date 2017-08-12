import tensorflow as tf

def main():
  node1 = tf.constant(3.0, dtype=tf.float32)
  node2 = tf.constant(4.0)
  print(node1, node2)

  session = tf.Session()
  print(session.run([node1, node2]))

  node3 = tf.add(node1, node2)
  print(node3)
  print("session run(node3)", session.run(node3))

if __name__ == "__main__":
  main()
