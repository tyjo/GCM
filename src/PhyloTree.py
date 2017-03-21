import numpy as np
import tensorflow as tf

import TransitionMatrix

class Node:

    def __init__(self, name, length=0, observation=None):
        self.name = name
        self.length = length
        self.observation = observation
        
        self.expectations = None
        self.left = None
        self.right = None

    def set_left(self, left_child):
        self.left = left_child

    def set_right(self, right_child):
        self.right = right_child


class PhyloTree:

    def __init__(self, root, tr_matrix):
        """
        root: the root node of the phylogenetic tree
        tr_matrix: transition matrix
        """
        self.root = root
        self.check_tree_(root)

        # Set root to uniform on all states

        # Set up expecation vectors for each child

        # create the transition matrix
        self.tr_matrix = tr_matrix()

    def check_tree_(self, root):
        """
        Make each node either has 2 or 0 children
        """
        if root.left == None and root.right != None or
           root.left != None and root.right == None:
           raise AssertionError("Leaf nodes must have 0 children.")

        check_tree_(root.left)
        check_tree_(root.right)

    def calculate_expectations_(self, root):
        """
        Calculates the expected state of each ancestral node.
        This isn't quite right.
        """

        if root.left == None or root.right == None:
            return

        # Given root, calculate expected left node and expected right node
        for fr in tr_matrix.states:
            for to in tr_matrix.states:
                left.expectations = tr_matrix.tr_prob(fr, left.length) * root[fr]
                right.expectations = tr_matrix.tr_prob(fr, right.length) * root[fr]

        calculate_expecations_(root.left)
        calculate_expecations_(root.right)

    def maximize_log_likelihood_(self):
        """
        Maximizes the expected complete log conditional.
        Maximization proceeds by gradient ascent.
        """
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        log_likelihood = 0 # calculate here
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = optimizer.minimize(loss)
        
        prv = sess.run([log_likelihood])[0]
        nxt = 0
        while abs(nxt - prv):
            sess.run(train)

    def simulate(self):
        """
        Given a tree, simulates states along the branches.
        """
        pass

    def estimate(self):
        """
        Runs the expectation maximization algorithm to estimate
        model parameter.
        """
        pass
