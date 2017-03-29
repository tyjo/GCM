import tensorflow as tf
import src.TransitionMatrix as tm
import src.PhyloTree as tree

if __name__ == "__main__":
    # Build transition matrix and compute
    m = tm.TransitionMatrix(0.2, 0.1, 0.3, 0.5)
    print (m)
    p1 = m.tr_matrix(10)
    p2 = m.tr_matrix(20)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(p1))
    print(sess.run(p2))
    
    #tr1 = sess.run(tf.reduce_sum(p1))
    #print (tr1)
    #train=tf.train.GradientDescentOptimizer(0.01).minimize(tf.reduce_sum(m.tr_matrix(10)))
    #print ("train ", train)
    
    # Build a phylogenetic tree
    root = tree.Node("5")
    a1 = tree.Node("6", 3)
    a2 = tree.Node("7", 2)
    child1 = tree.Node("1", 1, "AC")
    child2 = tree.Node("2", 1, "AC")
    child3 = tree.Node("3", 2, "TC")
    child4 = tree.Node("4", 2, "TC")

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
