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
    length_factor=1
    # Tree: ((((hg19:0.006653,panTro2:0.006688):0.002482,gorGor1:0.008783):0.009697,ponAbe2:0.018183)
    hg19 = tree.Node("hg19", 0.006653*length_factor)
    panTro2 = tree.Node("panTro2", 0.006688*length_factor)
    gorGor1 = tree.Node("gorGor1", 0.008783*length_factor)
    ponAbe2 = tree.Node("ponAbe", 0.018183*length_factor)
    hg_pan = tree.Node("hg19:panTro2", 0.002482*length_factor)
    hg_pan.left = hg19
    hg_pan.right = panTro2
    hg_pan_gor = tree.Node("hg_pan:gorGor1", 0.009697*length_factor)
    hg_pan_gor.left = hg_pan
    hg_pan_gor.right = gorGor1
    hg_pan_gor_pon = tree.Node("hg_pan_gor:ponAbe2", 0*length_factor)
    hg_pan_gor_pon.left = hg_pan_gor
    hg_pan_gor_pon.right = ponAbe2


    with open(argv[1], 'w') as f:
        f.write('')
    frm=open("rate_matrix.txt", 'w')
    for i in range(4):
        param = [np.random.uniform(0.1, 0.9) for i in range(4)]
        #param=[0.2,0.5,0.1,0.2]
        true_param = param[:]
        m1 = tm.TransitionMatrix(param[0], param[1], param[2], param[3])
        tmp=m1.compute_rate_matrix()
        #for state in range(len(m1.states)):
        #    for state2 in range(len(m1.states)):
        #        frm.write("%.1f\t" %tmp[state][state2])
        #    frm.write ("\n")
        phylo_tree = tree.PhyloTree(hg_pan_gor_pon, m1)
        phylo_tree.simulate(100000)
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
        phylo_tree = tree.PhyloTree(hg_pan_gor_pon, m2)
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


