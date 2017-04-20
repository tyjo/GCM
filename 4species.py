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
    
    # Tree: ((((hg19:0.006653,panTro2:0.006688):0.002482,gorGor1:0.008783):0.009697,ponAbe2:0.018183)
    hg19 = tree.Node("hg19", 0.006653)
    panTro2 = tree.Node("panTro2", 0.006688)
    gorGor1 = tree.Node("gorGor1", 0.008783)
    ponAbe2 = tree.Node("ponAbe", 0.018183)
    hg_pan = tree.Node("hg19:panTro2", 0.002482)
    hg_pan.left = hg19
    hg_pan.right = panTro2
    hg_pan_gor = tree.Node("hg_pan:gorGor1", 0.009697)
    hg_pan_gor.left = hg_pan
    hg_pan_gor.right = gorGor1
    hg_pan_gor_pon = tree.Node("hg_pan_gor:ponAbe2")
    hg_pan_gor_pon.left = hg_pan_gor
    hg_pan_gor_pon.right = ponAbe2

    with open(argv[1], 'w') as f:
        f.write('true\tstart\tinferred\tlog_likelihood\ttime\n')

    for i in range(50):
        param = [np.random.uniform(0.1, 1) for i in range(4)]
        true_param = param[:]
        m = tm.TransitionMatrix(param[0], param[1], param[2], param[3])
        phylo_tree = tree.PhyloTree(hg_pan_gor_pon, m)
        phylo_tree.simulate(10000)
        phylo_tree.set_simulated_observations()
        start_time = time.time()
        estim = phylo_tree.estimate()
        total_time = (time.time() - start_time) / 60.
        with open(argv[1], 'a') as f:
            f.write("{}\t{}\t{}\t{}\t{}\n".format(true_param, param, estim[0], estim[1], total_time))


