import numpy as np
import os
import tensorflow as tf
import src.TransitionMatrix

class Node:

    def __init__(self, name, length=0, observations=None):
        self.name = name
        self.length = length
        self.observations = observations
        self.left = None
        self.right = None

        # Probability of observations p(X|Node) that descended from the current
        # node. We use this in the bottom up part of the calculation for
        # expectations. Left descendants are those that are direct descendants
        # of the node. Right descendants are those observations that are not
        # directly below the node. That is, descendants of some ancestral node.
        # The root node is an exception, where left decendants and right decendants
        # direct descendents of the left and right leaves respectively.
        self.prob_left_descendants_ = None
        self.prob_right_descendants_ = None

        # Numerical conditional expectation of the node given observations and
        # current parameter values.
        self.expectations_ = None

        # Placeholders for expectations to pass to Tensorflow for maximizing 
        # the expected complete log likelihood.
        self.placeholders_ = None

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


        #############################
        ### Tensorflow parameters ###
        #############################
        self.sess = None
        # The tensorflow graph of the expected complete log likelihood, prevents recomputation
        self.log_likelihood = 0
        # Dictionary of placeholders to pass expected values to session as feed_dict
        self.expected_value_ph = {}
        # Numerical optimization routine
        self.optimizer = None
        self.train = None

        assert root.left != None, "Trees must have at least one child node"
        self.check_tree_(root)
        self.setup_(self.root)


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

        if self.is_leaf_node(root) and self.num_obs == 0:
                self.num_obs = len(root.observations)

        elif self.is_leaf_node(root) == None and self.num_obs != len(root.observations):
                raise AssertionError("Leaf nodes must all have the same number of observations")

        self.check_tree_(root.left)
        self.check_tree_(root.right)


    def setup_(self, root):
        """
        Set up tensorflow placeholders to pass to maximization routine.
        These placeholders are initialized to expectations.
        """
        if root.left == None:
            return

        root.placeholders_ = tf.placeholder(tf.float32, shape=(self.num_obs, self.num_states, 1))

        self.setup_(root.left)
        self.setup_(root.right)


    def is_leaf_node(self, node): return node.left == None


    def calculate_expectations_(self, root):
        """
        Calculates the expected state of each ancestral node.
        """
        def bottom_up(root):
            """
            Computes p(X | node) for descendents X. See Node class
            for description of left and right descendants
            """
            root.prob_left_descendants_ = np.zeros((self.num_obs, self.num_states))

            if self.is_leaf_node(root.left):
                t1 = root.left.length
                t2 = root.right.length
                if t1 not in tr_matrices: tr_matrices[t1] = self.sess.run(self.tr_matrix.tr_matrix(t1))
                if t2 not in tr_matrices: tr_matrices[t2] = self.sess.run(self.tr_matrix.tr_matrix(t2))
                m1 = tr_matrices[t1]
                m2 = tr_matrices[t2]

                for i in range(self.num_obs):
                    for fr in range(self.num_states):
                        prob_left = 0
                        prob_right = 0
                        for to in range(self.num_states):
                            if self.tr_matrix.states[to][0] == root.left.observations[i]:
                                prob_left += m1[fr][to]
                            if self.tr_matrix.states[to][0] == root.right.observations[i]:
                                prob_right += m2[fr][to]
                        root.prob_left_descendants_[i][fr] = prob_left * prob_right

            else:
                bottom_up(root.left)
                bottom_up(root.right)

                t1 = root.left.length
                t2 = root.right.length
                if t1 not in tr_matrices: tr_matrices[t1] = self.sess.run(self.tr_matrix.tr_matrix(t1))
                if t2 not in tr_matrices: tr_matrices[t2] = self.sess.run(self.tr_matrix.tr_matrix(t2))
                m1 = tr_matrices[t1]
                m2 = tr_matrices[t2]

                for i in range(self.num_obs):
                    for fr in range(self.num_states):
                        assert abs(m1[fr].sum() - 1) < 0.01, "Bad transition matrix"
                        x = root.left.prob_left_descendants_[i] * m1[fr]
                        y = root.right.prob_left_descendants_[i] * m2[fr]
                        root.prob_left_descendants_[i][fr] = x.T.dot(np.eye(self.num_states).dot(y))


        def top_down(node, prob_right_descendants, expectations):
            """
            Computes expecations of left and right children after call to bottom_up.
            """
            if self.is_leaf_node(node):
                return

            node.prob_right_descendants_ = np.zeros((self.num_obs, self.num_states))
            node.expectations_ = np.zeros((self.num_obs, self.num_states))
            for i in range(self.num_obs):
                node.prob_right_descendants_[i] = tr_matrices[node.length].dot(prob_right_descendants[i])
                node.expectations_[i] = node.prob_left_descendants_[i] * prob_right_descendants[i] * tr_matrices[node.left.length].dot(expectations[i])
                node.expectations_[i] /= node.expectations_[i].sum()
            self.expected_value_ph[node.placeholders_] = node.expectations_.reshape(self.num_obs, self.num_states, 1)

            if not self.is_leaf_node(root.left):
                top_down(node.left, node.prob_right_descendants_, node.expectations_)
                top_down(node.right, node.prob_right_descendants_, node.expectations_)
        
        # Save computed transition matrices. Dictionary from time to np.array.
        tr_matrices = {}
        bottom_up(root.left)
        bottom_up(root.right)

        t1 = root.left.length
        t2 = root.right.length
        if t1 not in tr_matrices: tr_matrices[t1] = self.sess.run(self.tr_matrix.tr_matrix(t1))
        if t2 not in tr_matrices: tr_matrices[t2] = self.sess.run(self.tr_matrix.tr_matrix(t2))
        m1 = tr_matrices[t1]
        m2 = tr_matrices[t2]

        root.prob_left_descendants_ = np.zeros((self.num_obs, self.num_states))
        root.prob_right_descendants_ = np.zeros((self.num_obs, self.num_states))
        root.expectations_ = np.zeros((self.num_obs, self.num_states))

        for i in range(self.num_obs):
            root.prob_left_descendants_[i] = m1.dot(root.left.prob_left_descendants_[i])
            root.prob_right_descendants_[i] = m2.dot(root.right.prob_left_descendants_[i])
            root.expectations_[i] = root.prob_left_descendants_[i] * root.prob_right_descendants_[i] * self.initial_distribution
            root.expectations_[i] /= root.expectations_[i].sum()
        self.expected_value_ph[root.placeholders_] = root.expectations_.reshape(self.num_obs, self.num_states, 1)

        top_down(root.left, root.prob_right_descendants_, root.expectations_)
        top_down(root.right, root.prob_left_descendants_, root.expectations_)


    def compute_complete_log_likelihood_(self, root):
        """
        Computes the graph corresponding to the complete log likelihood and
        stores it in self.log_likelihood
        """
        def compute_helper(root, expectations):
            """
            root: the node whose complete log likelihood is to be computed
            expecations: placeholders giving expected values of parent node
            """
            if self.is_leaf_node(root):
                for i in range(self.num_obs):
                    possible_states = tf.constant([1. if root.observations[i] == self.tr_matrix.states[j][0] else 0. \
                                                       for j in range(self.num_states)], dtype=tf.float32)
                    # Can't have zero entries when we take the log. We set zero entries in A
                    # to one so they become zero after applying tf.log.
                    A = tf.multiply(self.tr_matrix.tr_matrix(root.length), possible_states) \
                        + tf.constant([1. for i in range(self.num_states)]) - possible_states
                    self.log_likelihood += tf.reduce_sum(tf.matmul(tf.log(A), expectations[i]))

            else:
                for i in range(self.num_obs):
                    self.log_likelihood += tf.reduce_sum(tf.matmul(tf.log(self.tr_matrix.tr_matrix(root.length)), expectations[i]))
                    compute_helper(root.left, root.placeholders_)
                    compute_helper(root.right, root.placeholders_)

        compute_helper(root.left, root.placeholders_)
        compute_helper(root.right, root.placeholders_)


    def maximize_log_likelihood_(self):
        """
        Maximizes the expected complete log conditional.
        Maximization proceeds by gradient ascent.
        """
        prv = 0
        nxt = self.sess.run(self.log_likelihood, feed_dict=self.expected_value_ph)
        it = 1
        while abs(nxt - prv) > 0.1:
            prv = nxt
            self.sess.run(self.train, feed_dict=self.expected_value_ph)
            nxt = self.sess.run(self.log_likelihood, feed_dict=self.expected_value_ph)
            it += 1
            print("\tlog_likelihood =", nxt)


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
            row1 = self.sess.run(self.tr_matrix.tr_matrix(node.left.length))[index]
            row2 = self.sess.run(self.tr_matrix.tr_matrix(node.right.length))[index]
            
            # check matrix exponential calculation
            assert(sum(row1) - 1 < 0.01)
            assert(sum(row2) - 1 < 0.01)

            i1 = np.random.multinomial(1, row1).argmax()
            i2 = np.random.multinomial(1, row2).argmax()
            node.left.simulated_obs.append(self.tr_matrix.states[i1])
            node.right.simulated_obs.append(self.tr_matrix.states[i2])

        print("{}\t{}".format(node.left.name, node.left.simulated_obs))
        print("{}\t{}".format(node.right.name, node.right.simulated_obs))

        self.simulate_(node.left)
        self.simulate_(node.right)


    def simulate(self, size):
        """
        Given a tree, simulates states along the branches.
        """
        print("Simulating tree...")
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.root.simulated_obs = []
        for i in range(size):
            self.root.simulated_obs.append(np.random.choice(self.tr_matrix.states))

        print("{}\t{}".format(self.root.name, self.root.simulated_obs))
        if self.root.left != None:
            self.simulate_(self.root)


    def estimate(self, step_size):
        """
        Runs the expectation maximization algorithm to estimate
        model parameter.
        """
        np.seterr(under="raise")
        print("Building computation graph...")
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.compute_complete_log_likelihood_(self.root)
        self.optimizer = tf.train.GradientDescentOptimizer(step_size)
        self.train = self.optimizer.minimize(-self.log_likelihood)   
        #writer = tf.summary.FileWriter("log", graph=tf.get_default_graph())

        prv = np.zeros(4)
        nxt = np.array(self.sess.run([self.tr_matrix.tr_rate, self.tr_matrix.tv_rate,
                                      self.tr_matrix.on_rate, self.tr_matrix.off_rate]))
        print("Running EM algorithm...")
        it = 1
        while abs((nxt - prv).sum()) > 0.001:
            print("iteration: ", it)
            print("\tE step...")
            self.calculate_expectations_(self.root)
            print("\tM step...")
            self.maximize_log_likelihood_()
            prv = nxt
            nxt = np.array(self.sess.run([self.tr_matrix.tr_rate, self.tr_matrix.tv_rate,
                                      self.tr_matrix.on_rate, self.tr_matrix.off_rate]))
            #nxt = self.sess.run(self.log_likelihood, feed_dict=self.expected_value_ph)
            #parameters = np.array(self.sess.run([self.tr_matrix.tr_rate, self.tr_matrix.tv_rate,
            #                                     self.tr_matrix.on_rate, self.tr_matrix.off_rate]))
            print("\tParameter estimates: ", nxt)
            it += 1


    def print_parameters(self):
        print("tr_rate:\t", self.sess.run(self.tr_matrix.tr_rate))
        print("tv_rate:\t", self.sess.run(self.tr_matrix.tv_rate))
        print("on_rate:\t", self.sess.run(self.tr_matrix.on_rate))
        print("off_rate:\t", self.sess.run(self.tr_matrix.off_rate))
