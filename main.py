import tensorflow as tf
import src.TransitionMatrix as tm

if __name__ == "__main__":
    sess = tf.Session()
    m = tm.TransitionMatrix(0.5, 0.5, 0.2, 0.2)
    prob = m.tr_prob("A000", 10)
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(prob))