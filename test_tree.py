import numpy as np
import time
from sys import argv
from sys import exit

import src.TransitionMatrix as tm
import src.PhyloTree as tree

if __name__ == "__main__":
    if len(argv) < 2:
        print("please provide output file")
        exit(1)

    # Build a phylogenetic tree
    m = tm.TransitionMatrix(0.924, 1.283, 0.307, 0.662)
    root = tree.Node("root", 0) # specify initial state
    a1 = tree.Node("a1", 0.5)
    a2 = tree.Node("a2", 0.5)
    child1 = tree.Node("child1", 0.5)
    child2 = tree.Node("child2", 0.5)
    child3 = tree.Node("child3", 0.5)
    child4 = tree.Node("child4", 0.5)
    root.left = a1
    root.right = a2
    a1.left = child1
    a1.right = child2
    a2.left = child3
    a2.right = child4
    phylo_tree = tree.PhyloTree(root, m)
    phylo_tree.simulate(100000)
    phylo_tree.set_simulated_observations()
    estim = phylo_tree.estimate()
    exit(0)

    with open(argv[1], 'w') as f:
        f.write('')

    for i in range(50):
        param = [np.random.uniform(0.1, 0.9) for i in range(4)]
        true_param = param[:]
        m1 = tm.TransitionMatrix(param[0], param[1], param[2], param[3])
        phylo_tree = tree.PhyloTree(root, m1)
        phylo_tree.simulate(10000)
        phylo_tree.set_simulated_observations()
        start_time = time.time()
        estim = phylo_tree.estimate()
        total_time = (time.time() - start_time) / 60.

        # Maximize log likelihood starting from true parameters
        with open(argv[1], 'a') as f:
            f.write("Simulation: {}-a\n".format(i))
            f.write("True Parameters: {}\n".format(true_param))
            f.write("Initial Parameters: {}\n".format(param))
            f.write("Inferred Parameters: {}\n".format(estim[0]))
            f.write("Log Likelihood: {}\n".format(estim[1]))
            f.write("Running Time: {}\n".format(total_time))

        param = [np.random.uniform(0.1, 0.9) for i in range(4)]
        m2 = tm.TransitionMatrix(param[0], param[1], param[2], param[3])
        phylo_tree = tree.PhyloTree(root, m2)
        start_time = time.time()
        estim = phylo_tree.estimate()
        total_time = (time.time() - start_time) / 60.

        # Maximize log likelihood starting from random parameters
        with open(argv[1], 'a') as f:
            f.write("Simulation: {}-b\n".format(i))
            f.write("True Parameters: {}\n".format(true_param))
            f.write("Initial Parameters: {}\n".format(param))
            f.write("Inferred Parameters: {}\n".format(estim[0]))
            f.write("Log Likelihood: {}\n".format(estim[1]))
            f.write("Running Time: {}\n\n".format(total_time))