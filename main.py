import tensorflow as tf
import src.TransitionMatrix as tm
import src.PhyloTree as tree

if __name__ == "__main__":
    # Build transition matrix and compute
    m = tm.TransitionMatrix(0.2, 0.1, 0.3, 0.5)
    p1 = m.tr_matrix(10)
    p2 = m.tr_matrix(20)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(p1))
    print(sess.run(p2))

    # Build a phylogenetic tree
    m = tm.TransitionMatrix(0.2, 0.1, 0.3, 0.5)
    root = tree.Node("root")
    a1 = tree.Node("a1", 10)
    a2 = tree.Node("a2", 10)
    child1 = tree.Node("child1", 20, "AC")
    child2 = tree.Node("child2", 20, "AC")
    child3 = tree.Node("child3", 20, "TC")
    child4 = tree.Node("child4", 20, "TC")
    root.left = a1
    root.right = a2
    a1.left = child1
    a1.right = child2
    a2.left = child3
    a2.right = child4
    phylo_tree = tree.PhyloTree(root, m)

    # Simulate a sequence of length 3
    phylo_tree.simulate(3)

    # Estimate the parameters of the model given observations AC and TC above
    phylo_tree.estimate()
    phylo_tree.print_parameters()