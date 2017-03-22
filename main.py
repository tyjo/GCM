import tensorflow as tf
import src.TransitionMatrix as tm

if __name__ == "__main__":
    sess = tf.Session()
    m = tm.TransitionMatrix(2.5, 1.5, 1.6, 2.3)
    #prob = m.tr_prob("A000", 0.1)
    P = m.tr_matrix(1)
    init = tf.global_variables_initializer()
    sess.run(init)
    prt = tf.Print(P,[P],summarize=64)
    sess.run(prt)