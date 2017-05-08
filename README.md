# Simulation and estimation under the Generalized Covarion Model

Parameter estimation by maximum likelihood under the generalized covarion model.

## Dependencies
1. Numpy >= 1.12.1
2. Python 3

## Simulation and Estimation
Simulating data consists of three parts:

1. Specifying the transition matrix
2. Specifying the phylogenetic tree
3. Simulating data on the tree

Similarly, estimating model parameters consists of three parts:

1. Specifying the transition matrix
2. Specifying the phylogenetic tree
3. Estimating parameters

### Specifying the transition matrix
The transition matrix takes 4 parameters: transition rate, transversion rate, on rate, and off rate. On rate specifies the rate of switches from OFF to ON, where off rate specifies the rate of switches from ON to OFF.

```python
import src.TransitionMatrix as tm

matrix = tm.TransitionMatrix(tr_rate, tv_rate, on_rate, off_rate)
```


### Building the phylogenetic tree
Phylogenetic trees are specified as nodes in the tree, each with a branch length that corresponds to the length to the node's ancester. The root node has length 0. The nodes are linked together by specifying left and right childen. A phylogenetic tree is defined through both a root node and transition matrix. To specify a tree with 2 observations, with common ancestor "root" 

```python
import src.PhyloTree as tree

root = tree.Node(name="root", length=0)
child1 = tree.Node(name="child1", length=0.2)
child2 = tree.Node(name="child2", length=0.2)
root.left = child1
root.right = child2
phylo_tree = tree.PhyloTree(root, matrix)
```

### Simulating data
Given a phylogenetic tree and transition matrix, data can by simulated by calling

```
phylo_tree.simulate(N)
```
where N is the length of the sequence.

### Inference
Inference proceeds by maximum likelihood. The initial parameters of the transition matrix are those that are passed to the constructor. Observed sequences are specified in the Node constructor. The example below infers parameters for a 4 species tree with 100 base pair sequences.

```python
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
```

Parameters provided to the transition matrix specify initial parameters passed to the numerical optimization routine.

This example is provided in ```sample.py```. To estimate parameters on this example, call

```python sample.py```

You should see

```
...
...
...
parameter estimates = [ 0.09128966  0.29990838  0.25255296  0.05355357]
log likelihood      = -472.739894676
parameter estimates = [ 0.09128615  0.29990093  0.25248618  0.05353279]
log likelihood      = -472.739894675
parameter estimates = [ 0.09128992  0.29990558  0.25252533  0.05354676]
log likelihood      = -472.739894675
parameter estimates = [ 0.09128992  0.29990558  0.25252533  0.05354676]
tr_rate:	 0.091289918036
tv_rate:	 0.299905582895
on_rate:	 0.252525328397
off_rate:	 0.0535467552549
```

## Example code
More usage examples are provided in ```test_tree.py```, ```4species.py```, and ```11species.py```. Each file simulates 50 replicates on the specified tree, and estimates parameters from i) the true parameter initialization, and; ii) a random initialization. Results are output to a file provided as a command line argument:

```
python test_tree.py test_output
```