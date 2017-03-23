import tensorflow as tf
import src.TransitionMatrix as tm
import src.PhyloTree as tree

if __name__ == "__main__":
    """
    sess = tf.Session()
    m = tm.TransitionMatrix(0.1, 0.5, 3, 1)
    #prob = m.tr_prob("A000", 0.1)
    P = m.tr_matrix(100)
    init = tf.global_variables_initializer()
    fw = tf.summary.FileWriter("./log")
    fw.add_graph(P.graph)
    sess.run(init)
    #prt = tf.Print(P,[P],summarize=64)
    #sess.run(prt)
    sess.run(P)
    """

    # Simulating observations from a tree
    m = tm.TransitionMatrix(0.2, 0.1, 0.3, 0.5)
    root = tree.Node("root")
    child1 = tree.Node("child1", 20)
    child2 = tree.Node("child2", 20)
    root.left = child1
    root.right = child2
    phylo_tree = tree.PhyloTree(root, m)
    phylo_tree.simulate(3)