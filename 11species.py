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
    hg19 = tree.Node("hg19", 0.006653)
    panTro2 = tree.Node("panTro2", 0.006688)
    gorGor1 = tree.Node("gorGor1", 0.008783)
    ponAbe2 = tree.Node("ponAbe2", 0.018183)
    rheMac2 = tree.Node("rheMac2", 0.008812)
    papHam1 = tree.Node("papHam1", 0.008723)
    calJac1 = tree.Node("calJac1", 0.066437)
    tarSyr1 = tree.Node("tarSyr1", 0.137822)
    micMur1 = tree.Node("micMur1", 0.092888)
    otoGar1 = tree.Node("otoGar1", 0.129500)
    tupBel1 = tree.Node("tupBel1", 0.186424)

    hg_pan = tree.Node("hg19:panTro2", 0.002482)
    hg_pan.left = hg19
    hg_pan.right = panTro2
    hg_pan_gor = tree.Node("hg_pan:gorGor1", 0.009697)
    hg_pan_gor.left = hg_pan
    hg_pan_gor.right = gorGor1
    hg_pan_gor_pon = tree.Node("hg_pan_gor:ponAbe2", 0.040003)
    hg_pan_gor_pon.left = hg_pan_gor
    hg_pan_gor_pon.right = ponAbe2
    hg_pan_gor_pon_rhe = tree.Node("hg_pan_gor_pon:rheMac2", 0.002489)
    hg_pan_gor_pon_rhe.left = hg_pan_gor_pon
    hg_pan_gor_pon_rhe.right = rheMac2
    hpgp_rhe_pap = tree.Node("hgpgp_rhe:papHam1", 0.045139)
    hpgp_rhe_pap.left = hg_pan_gor_pon_rhe
    hpgp_rhe_pap.right = papHam1
    hpgprp_cal = tree.Node("hgpgpr:calJack1", 0.057049)
    hpgprp_cal.left = hpgp_rhe_pap
    hpgprp_cal.right = calJac1
    hpgprpc_tar = tree.Node("hgpgprc:tarSyr1", 0.010992)
    hpgprpc_tar.left = hpgprp_cal
    hpgprpc_tar.right = tarSyr1

    micMur_otoGar = tree.Node("micMur1:otoGar1", 0.035423)

    hpgprpc_tar_micMur_otoGar = tree.Node("hpgprpc_tar:micMur_otoGar", 0.015348)
    hpgprpc_tar_micMur_otoGar.left = hpgprpc_tar
    hpgprpc_tar_micMur_otoGar.right = micMur_otoGar

    root = tree.Node("root", 0.004886)
    root.left = hpgprpc_tar_micMur_otoGar
    root.right = tupBel1


    with open(argv[1], 'w') as f:
        f.write('')

    for i in range(100):
        param = [np.random.uniform(0.1, 0.9) for i in range(4)]
        true_param = param[:]
        m1 = tm.TransitionMatrix(param[0], param[1], param[2], param[3])
        phylo_tree = tree.PhyloTree(root, m1)
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
        phylo_tree = tree.PhyloTree(root,m2)
        estim = phylo_tree.estimate()

        # Maximize log likelihood starting from random parameters
        with open(argv[1], 'a') as f:
            f.write("Simulation: {}-b\n".format(i))
            f.write("True Parameters: {}\n".format(true_param))
            f.write("Initial Parameters: {}\n".format(param))
            f.write("Inferred Parameters: {}\n".format(estim[0]))
            f.write("Log Likelihood: {}\n".format(estim[1]))
            f.write("Running Time: {}\n\n".format(total_time))


