# Simulation and estimation under the Generalized Covarion Model

Expectation maximization for the generalized covarion model.

## Simulation and Estimation
Simulating data consists of three parts:

1. Specifying the transition matrix
2. Specifying the phylogenetic tree
3. Simulating data on the tree

Similarly, estimating model parameters consists of three parts:

1. Specifying the transition matrix
2. Specifying the phylogenetic tree
3. Calling the EM algorithm

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
child1 = tree.Node(name="child1", length=20)
child2 = tree.Node(name="child2", length=20)
root.left = child1
root.right = child2
phylo_tree = tree.PhyloTree(root, matrix)
```

### Simulating data
Given a phylogenetic tree and transition matrix, data can by simulated by calling

```python
phylo_tree.simulate(N)
```
where N is the length of the sequence.

### Inference
Inference proceeds by expectation maximization. The initial parameters of the transition matrix are those that are passed to the constructor.

```python
import src.TransitionMatrix as tm
import src.PhyloTree as tree

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
phylo_tree.estimate()
phylo_tree.print_parameters()
```

See ```main.py``` for complete examples.