import numpy as np
import scipy.optimize
import os
import src.TransitionMatrix as tm

class Node:

    def __init__(self, name, length=0, observations=None):
        self.name = name
        self.length = length
        self.observations = observations
        self.left = None
        self.right = None

        # Numerical conditional expectation of the node given observations and
        # current parameter values.
        self.expectations_ = None

        # Store simulated values
        self.simulated_obs = None




class PhyloTree:

    def __init__(self, root, tr_matrix):
        """
        root: the root node of the phylogenetic tree
        tr_matrix: transition matrix
        """
        self.root = root
        self.tr_matrix = tr_matrix
        self.num_states = len(self.tr_matrix.states)
        # The length of the observed sequence in leaf nodes
        self.num_obs = 0
        # The initial distribution of the root node
        self.initial_distribution = np.array([ 1. / self.num_states for st in tr_matrix.states ])
        self.log_likelihood = 0

        assert root.left != None, "Trees must have at least one child node"
        self.check_tree_(root)
        #self.setup_(self.root)


    def check_tree_(self, root):
        """
        Make each node either and internal node with 2 children or
        a leaf node wih 0 children
        """
        if root == None:
            return

        elif root.left == None and root.right != None or \
             root.left != None and root.right == None:
             raise AssertionError("Leaf nodes must have 0 children.")

        if self.is_leaf_node(root) and root.observations != None and self.num_obs == 0:
                self.num_obs = len(root.observations)

        elif self.is_leaf_node(root) == None and root.observations != None and self.num_obs != len(root.observations):
                raise AssertionError("Leaf nodes must all have the same number of observations")

        self.check_tree_(root.left)
        self.check_tree_(root.right)


    def is_leaf_node(self, node): return node.left == None


    def compute_log_likelihood_(self, param):
        def compute_helper(root, tr_matrix):
            if root == None: return

            root.expectations_ = np.zeros((self.num_obs, self.num_states))
            if self.is_leaf_node(root.left):
                root.left.expectations_ = np.zeros((self.num_obs, self.num_states))
                root.right.expectations_ = np.zeros((self.num_obs, self.num_states))
                for i in range(self.num_obs):
                    #root.left.expectations_[i] = np.array([1. if root.left.observations[i] == self.tr_matrix.states[j] else 0. \
                    #                                   for j in range(self.num_states)])
                    #root.right.expectations_[i] = np.array([1. if root.right.observations[i] == self.tr_matrix.states[j] else 0. \
                    #                                   for j in range(self.num_states)])
                    root.left.expectations_[i] = np.array([1. if root.left.observations[i][0] == self.tr_matrix.states[j][0] else 0. \
                                                       for j in range(self.num_states)])
                    root.right.expectations_[i] = np.array([1. if root.right.observations[i][0] == self.tr_matrix.states[j][0] else 0. \
                                                       for j in range(self.num_states)])
                    left = tr_matrix.tr_matrix(root.left.length) * root.left.expectations_[i]
                    right = tr_matrix.tr_matrix(root.right.length) * root.right.expectations_[i]
                    root.expectations_[i] = left.sum(axis=1)*right.sum(axis=1)

            else:
                compute_helper(root.left, tr_matrix)
                compute_helper(root.right, tr_matrix)
            
                for i in range(self.num_obs):
                    left = tr_matrix.tr_matrix(root.left.length).dot(root.left.expectations_[i])
                    right = tr_matrix.tr_matrix(root.right.length).dot(root.right.expectations_[i])
                    root.expectations_[i] = left*right
                    #left = tr_matrix.tr_matrix(root.left.length)*root.left.expectations_[i]
                    #right = tr_matrix.tr_matrix(root.right.length)*root.right.expectations_[i]
                    #root.expectations_[i] = (left*right).sum()
                    #print(root.expectations_[i])
            
            #print(root.expectations_)
                #for j in range(self.num_states):
                #    for k in range(self.num_states):
                #        for l in range(self.num_states):
                #            root.expectations_[i][j] += left[j][k]*right[j][l]

        tr_matrix = tm.TransitionMatrix(param[0], param[1], param[2], param[3])
        compute_helper(self.root, tr_matrix)
        self.log_likelihood = np.log((self.root.expectations_*self.initial_distribution).sum(axis=1)).sum()
        assert(self.log_likelihood < 0.)
        #print(self.root.expectations_)
        #print(np.log(self.root.expectations_.sum(axis=1)).sum())
        #return
        print(self.log_likelihood)
        print(param)
        return -np.log(self.root.expectations_.sum(axis=1)).sum()


    def maximize_log_likelihood_(self):
        """
        Maximizes the expected complete log conditional.
        Maximization proceeds by gradient ascent.
        """
        np.set_printoptions(edgeitems=16)
        scipy.optimize.minimize(self.compute_log_likelihood_, 
                                np.array([self.tr_matrix.tr_rate,
                                          self.tr_matrix.tv_rate,
                                          self.tr_matrix.on_rate,
                                          self.tr_matrix.off_rate]),
                                method="nelder-mead")
                                #method="cobyla",
                                #options={"rhobeg":0.4},
                                #constraints = [ {'type': 'ineq', 'fun': lambda x: x > 0.001},
                                #                {'type': 'ineq', 'fun': lambda x: x > 0.001},
                                #                {'type': 'ineq', 'fun': lambda x: x > 0.001},
                                #                {'type': 'ineq', 'fun': lambda x: x > 0.001}])
        print("\t", self.log_likelihood)


    def simulate_(self, node):
        """
        Helper function to simulate observations.
        """
        if node.left == None:
            return

        node.left.simulated_obs = []
        node.right.simulated_obs = []
        for obs in node.simulated_obs:
            index = self.tr_matrix.states.index(obs)
            row1 = self.tr_matrix.tr_matrix(node.left.length)[index]
            row2 = self.tr_matrix.tr_matrix(node.right.length)[index]
            
            # check matrix exponential calculation
            assert(sum(row1) - 1 < 0.01)
            assert(sum(row2) - 1 < 0.01)

            i1 = np.random.multinomial(1, row1).argmax()
            i2 = np.random.multinomial(1, row2).argmax()

            if self.is_leaf_node(node.left):
                #node.left.simulated_obs.append(self.tr_matrix.states[i1][0])
                #node.right.simulated_obs.append(self.tr_matrix.states[i2][0])
                node.left.simulated_obs.append(self.tr_matrix.states[i1])
                node.right.simulated_obs.append(self.tr_matrix.states[i2])

            else:
                node.left.simulated_obs.append(self.tr_matrix.states[i1])
                node.right.simulated_obs.append(self.tr_matrix.states[i2])

        print("{}\t{}".format(node.left.name, node.left.simulated_obs))
        print("{}\t{}".format(node.right.name, node.right.simulated_obs))
        #print("{}\t{}".format(node.left.name, node.left.simulated_obs))
        #print("{}\t{}".format(node.right.name, node.right.simulated_obs))

        self.simulate_(node.left)
        self.simulate_(node.right)


    def simulate(self, size):
        """
        Given a tree, simulates states along the branches.
        """
        print("Simulating tree...")
        self.root.simulated_obs = []
        for i in range(size):
            self.root.simulated_obs.append(np.random.choice(self.tr_matrix.states))

        print("{}\t{}".format(self.root.name, self.root.simulated_obs))
        if self.root.left != None:
            self.simulate_(self.root)


    def estimate(self):
        """
        Runs the expectation maximization algorithm to estimate
        model parameter.
        """
        np.seterr(under="raise")
        #print("Building computation graph...")
        print("Running...")
        self.maximize_log_likelihood_()
        param = np.array([self.tr_matrix.tr_rate, self.tr_matrix.tv_rate,
                          self.tr_matrix.on_rate, self.tr_matrix.off_rate])
        lk = self.compute_log_likelihood_(param)
        print("\tParameter estimates: ", param)
        return (lk, param)
        

    def print_parameters(self):
        print("tr_rate:\t", self.tr_matrix.tr_rate)
        print("tv_rate:\t", self.tr_matrix.tv_rate)
        print("on_rate:\t", self.tr_matrix.on_rate)
        print("off_rate:\t", self.tr_matrix.off_rate)
