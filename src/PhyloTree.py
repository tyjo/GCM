import numpy as np
import tensorflow as tf
import src.TransitionMatrix

class Node:

    def __init__(self, name, length=0, observations=None):
        self.name = name
        self.length = length
        self.observations = observations
        
        self.expectations = None
        self.left = None
        self.right = None


class PhyloTree:

    def __init__(self, root, tr_matrix):
        """
        root: the root node of the phylogenetic tree
        tr_matrix: transition matrix
        """
        self.root = root
        self.check_tree_(root)
        self.tr_matrix = tr_matrix

        # Tensorflow session
        self.sess = None

        self.num_states = len(tr_matrix.states)
        # This should be the initial distribution
        self.root.expectations = [ 1. / self.num_states for st in tr_matrix.states ]
        self.setup_nodes_(root.left)
        self.setup_nodes_(root.right)

    def check_tree_(self, root):
        """
        Make each node either has 2 or 0 children
        """
        if root == None:
            return

        elif root.left == None and root.right != None or \
           root.left != None and root.right == None:
           raise AssertionError("Leaf nodes must have 0 children.")

        self.check_tree_(root.left)
        self.check_tree_(root.right)

    def setup_nodes_(self, node):
        """
        Sets up vector for expecations
        """
        if node == None:
            return

        node.expecations = [ 0. for st in range(self.num_states)]
        self.setup_nodes_(node.left)
        self.setup_nodes_(node.right)

    def calculate_expectations_(self, root):
        """
        Calculates the expected state of each ancestral node.
        This isn't quite right. These expecations need to be evaluations
        of graph.
        """

        if root.left == None or root.right == None:
            return

        # Given root, calculate expected left node and expected right node
        left.expectations = [0. for state in tr_matrix.states]
        right.expectations = [0. for state in tr_matrix.states]
        for fr in tr_matrix.states:
            for to in tr_matrix.states:
                left.expectations[to] += self.sess.run(tr_matrix.tr_prob(fr, to, left.length)) * root[fr]
                right.expectations[to] += self.sess.run(tr_matrix.tr_prob(fr, to, right.length)) * root[fr]

        self.calculate_expectations_(root.left)
        self.calculate_expectations_(root.right)

    def maximize_log_likelihood_(self):
        """
        Maximizes the expected complete log conditional.
        Maximization proceeds by gradient ascent.
        """

        log_likelihood = 0 # calculate here
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = optimizer.minimize(-loss)
        
        prv = sess.run(log_likelihood)
        nxt = 0
        while abs(nxt - prv) > 0.1:
            sess.run(train)

    def simulate(self, size):
        """
        Given a tree, simulates states along the branches.
        """
        print("Simulating tree...")
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.root.observations = []
        for i in range(size):
            index = np.random.multinomial(1, self.root.expectations).argmax()
            self.root.observations.append(self.tr_matrix.states[index])

        print("{}\t{}".format(self.root.name, self.root.observations))
        
        if self.root.left != None:
            self.simulate_(self.root)


    def simulate_(self, node):
        """
        Helper function to simulate observations.
        """

        if node.left == None:
            return

        node.left.observations = []
        node.right.observations = []
        for obs in node.observations:
            index = self.tr_matrix.states.index(obs)
            row1 = self.sess.run(self.tr_matrix.tr_matrix(node.left.length))[index]
            row2 = self.sess.run(self.tr_matrix.tr_matrix(node.right.length))[index]
            i1 = np.random.multinomial(1, row1).argmax()
            i2 = np.random.multinomial(1, row2).argmax()
            node.left.observations.append(self.tr_matrix.states[i1])
            node.right.observations.append(self.tr_matrix.states[i2])

        print("{}\t{}".format(node.left.name, node.left.observations))
        print("{}\t{}".format(node.right.name, node.right.observations))

        self.simulate_(node.left)
        self.simulate_(node.right)


    def estimate(self):
        """
        Runs the expectation maximization algorithm to estimate
        model parameter.
        """
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
