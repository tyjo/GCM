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
    # Tree: ((((((((((hg19:0.006653,panTro2:0.006688):0.002482,gorGor1:0.008783):0.009697,ponAbe2:0.018183):0.040003,
    # rheMac2:0.008812):0.002489,papHam1:0.008723):0.045139,calJac1:0.066437):0.057049,tarSyr1:0.137822):0.010992,
    # (micMur1:0.092888,otoGar1:0.129500):0.035423):0.015348,
    #  tupBel1:0.186424)
    length_factor=10
    hg19 = tree.Node("hg19", 0.006653*length_factor)
    panTro2 = tree.Node("panTro2", 0.006688*length_factor)
    gorGor1 = tree.Node("gorGor1", 0.008783*length_factor)
    ponAbe2 = tree.Node("ponAbe2", 0.018183*length_factor)
    rheMac2 = tree.Node("rheMac2", 0.008812*length_factor)
    papHam1 = tree.Node("papHam1", 0.008723*length_factor)
    calJac1 = tree.Node("calJac1", 0.066437*length_factor)
    tarSyr1 = tree.Node("tarSyr1", 0.137822*length_factor)
    micMur1 = tree.Node("micMur1", 0.092888*length_factor)
    otoGar1 = tree.Node("otoGar1", 0.129500*length_factor)
    tupBel1 = tree.Node("tupBel1", 0.186424*length_factor)

    hg_pan = tree.Node("hg19:panTro2", 0.002482*length_factor)
    hg_pan.left = hg19
    hg_pan.right = panTro2
    hg_pan_gor = tree.Node("hg_pan:gorGor1", 0.009697*length_factor)
    hg_pan_gor.left = hg_pan
    hg_pan_gor.right = gorGor1
    hg_pan_gor_pon = tree.Node("hg_pan_gor:ponAbe2", 0.040003*length_factor)
    hg_pan_gor_pon.left = hg_pan_gor
    hg_pan_gor_pon.right = ponAbe2
    hg_pan_gor_pon_rhe = tree.Node("hg_pan_gor_pon:rheMac2", 0.002489*length_factor)
    hg_pan_gor_pon_rhe.left = hg_pan_gor_pon
    hg_pan_gor_pon_rhe.right = rheMac2
    hpgp_rhe_pap = tree.Node("hgpgp_rhe:papHam1", 0.045139*length_factor)
    hpgp_rhe_pap.left = hg_pan_gor_pon_rhe
    hpgp_rhe_pap.right = papHam1
    hpgprp_cal = tree.Node("hgpgpr:calJack1", 0.057049*length_factor)
    hpgprp_cal.left = hpgp_rhe_pap
    hpgprp_cal.right = calJac1
    hpgprpc_tar = tree.Node("hgpgprc:tarSyr1", 0.010992*length_factor)
    hpgprpc_tar.left = hpgprp_cal
    hpgprpc_tar.right = tarSyr1

    micMur_otoGar = tree.Node("micMur1:otoGar1", 0.035423*length_factor)

    hpgprpc_tar_micMur_otoGar = tree.Node("hpgprpc_tar:micMur_otoGar", 0.015348*length_factor)
    hpgprpc_tar_micMur_otoGar.left = hpgprpc_tar
    hpgprpc_tar_micMur_otoGar.right = micMur_otoGar

    root = tree.Node("root", 0.004886*length_factor)
    root.left = hpgprpc_tar_micMur_otoGar
    root.right = tupBel1


    with open(argv[1], 'w') as f:
        f.write('')

    for i in range(1):
        #param = [np.random.uniform(0.1, 0.9) for i in range(4)]
        param=[0.2,0.5,0.1,0.2]
        true_param = param[:]
        m1 = tm.TransitionMatrix(param[0], param[1], param[2], param[3])
        phylo_tree = tree.PhyloTree(root, m1)
        phylo_tree.simulate(1000)
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


