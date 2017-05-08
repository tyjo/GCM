import src.TransitionMatrix as tm
import src.PhyloTree as tree

m = tm.TransitionMatrix(0.83, 0.55, 0.49, 0.71)
root = tree.Node("root")
a1 = tree.Node("a1", 0.5)
a2 = tree.Node("a2", 0.5)
child1 = tree.Node("child1", 0.5, "GCCAGTCAACAAATTCGTGCACTAGGTAGGGTAATTTCCCCAGTCCTTAGTTCGCTACAAACTTCTTAACCATGATTAAGCCCTGGATTTGCTCAATACG")
child2 = tree.Node("child2", 0.5, "ACGACACAAAACATGAGTGGCGTTAGTCCGCTGATTTCCCTAGGCCTTATATTGCTACGGTCGTGTGCACCATGATCTTATAGAGGATTAACGGAATACG")
child3 = tree.Node("child3", 0.5, "ACAATTAAAGACCTTCATGGACAAAACAGCGCCATTTGATTTCTCGTCCGTTTATACCCCTGCTCAGAGCGCTGACTTACAGATGCAGTGGCTGCAACCC")
child4 = tree.Node("child4", 0.5, "ACACTACTCTAAATTCATGGACTAAAGCGCGCCATGTGATTTGTGGTCCTTTGATTACCATGATCTTTGCCCTGAACTACGGATGCATGGGCTGCTAAAG")
root.left = a1
root.right = a2
a1.left = child1
a1.right = child2
a2.left = child3
a2.right = child4

phylo_tree = tree.PhyloTree(root, m)
phylo_tree.estimate()
phylo_tree.print_parameters()