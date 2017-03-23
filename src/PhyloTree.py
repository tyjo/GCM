import numpy as np
import tensorflow as tf
import src.TransitionMatrix

class Node:

    def __init__(self, name, length=0, observations=None):
        self.name = name
        self.length = length
        self.observations = observations
        
        self.expectations = None
        self.placeholder = None
        # expected log likelihood for observations
        self.log_likelihood = None
        self.left = None
        self.right = None


class PhyloTree:

    def __init__(self, root, tr_matrix):
        """
        root: the root node of the phylogenetic tree
        tr_matrix: transition matrix
        """
        self.root = root
        assert root.left != None, "Trees must have at least one child node"
        
        self.tr_matrix = tr_matrix
        self.num_states = len(tr_matrix.states)
        self.child_nodes = []
        self.get_child_nodes_(self.root)
        self.check_tree_(root)
        self.setup_(self.root)
        # This should be the initial distribution
        self.root.expectations = np.array([ 1. / self.num_states for st in tr_matrix.states ])

        ## Tensorflow parameters
        # store the tensorflow session
        self.sess = None
        # the tensorflow graph of the expected complete log likelihood, prevents recomputation
        self.log_likelihood = None
        # dictionary of placeholders to pass as feed_dict to session
        self.exp_placeholders = {}
        self.root.placeholder = tf.placeholder(tf.float64, self.num_states)
        self.exp_placeholders[self.root.placeholder] = self.root.expectations
        self.optimizer = tf.train.GradientDescentOptimizer(0.01)
        self.train = None


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

        self.check_tree_(root.left)
        self.check_tree_(root.right)


    def setup_(self, root):
        """
        Set up tensorflow placeholders to pass to maximization routine.
        These placeholders are initialized to expectations.
        """
        if root == None:
            return

        elif root.left != None:
            root.right.placeholder = tf.placeholder(tf.float64, shape=self.num_states)
            root.left.placeholder = tf.placeholder(tf.float64, shape=self.num_states)

        self.setup_(root.left)
        self.setup_(root.right)


    def get_child_nodes_(self, root):
        """
        Makes a list of leaf nodes.
        """
        if root.left.left == None:
            self.child_nodes.append(root.left)
            self.child_nodes.append(root.right)

        else:
            self.get_child_nodes_(root.left)
            self.get_child_nodes_(root.right)


    def calculate_expectations_(self, root):
        """
        Calculates the expected state of each ancestral node.
        """

        if root.left == None:
            return

        # Given root, calculate expected left node and expected right node
        root.left.expectations = np.zeros([len(self.tr_matrix.states)])
        root.right.expectations = np.zeros([len(self.tr_matrix.states)])
        tr1 = self.sess.run(self.tr_matrix.tr_matrix(root.left.length))
        tr2 = self.sess.run(self.tr_matrix.tr_matrix(root.right.length))

        for fr in self.tr_matrix.states:
            for to in self.tr_matrix.states:
                fr_index = self.tr_matrix.states.index(fr)
                to_index = self.tr_matrix.states.index(to)
                root.left.expectations[to_index] += \
                    tr1[fr_index, to_index] * root.expectations[fr_index]
                root.right.expectations[to_index] += \
                    tr2[fr_index, to_index] * root.expectations[fr_index]

        # Update the feed_dict to pass to Tensorflow
        self.exp_placeholders[root.left.placeholder] = root.left.expectations
        self.exp_placeholders[root.right.placeholder] = root.right.expectations

        # Check that we correctly calculated the matrix exponential
        assert(abs(root.left.expectations.sum() - 1) < 0.01)
        assert(abs(root.right.expectations.sum() - 1) < 0.01)
        self.calculate_expectations_(root.left)
        self.calculate_expectations_(root.right)


    def compute_observation_likelihoods_(self, root):
        """
        Computes the tensorflow graph of the log likelihood.
        Only need to do this once.
        """
        # The current node is a parent of an observations
        if root.left.left == None:
            root.left.log_likelihood = 0
            root.right.log_likelihood = 0
            tr1 = self.tr_matrix.tr_matrix(root.left.length)
            tr2 = self.tr_matrix.tr_matrix(root.right.length)
            for fr in self.tr_matrix.states:
                for to in self.tr_matrix.states:
                    fr_index = self.tr_matrix.states.index(fr)
                    to_index = self.tr_matrix.states.index(to)
                    for i in range(len(root.left.observations)):
                        if root.left.observations[i][0] == to[0]:
                            #print("left", fr, to, root.left.observations[i])
                            root.left.log_likelihood += tf.log(tr1[fr_index, to_index]) * \
                                                        root.placeholder[fr_index]
                        if root.right.observations[i][0] == to[0]:
                            #print("right", fr, to, root.right.observations[i])
                            root.right.log_likelihood += tf.log(tr2[fr_index, to_index]) * \
                                                         root.placeholder[fr_index]
        else:
            self.compute_observation_likelihoods_(root.left)
            self.compute_observation_likelihoods_(root.right)
            

    def maximize_log_likelihood_(self):
        """
        Maximizes the expected complete log conditional.
        Maximization proceeds by gradient ascent.
        """
        prv = 0
        nxt = self.sess.run(self.log_likelihood, feed_dict=self.exp_placeholders)
        it = 1
        while abs(nxt - prv) > 0.1:
            print("\tlog_likelihood =", nxt)
            prv = nxt
            self.sess.run(self.train, feed_dict=self.exp_placeholders)
            nxt = self.sess.run(self.log_likelihood, feed_dict=self.exp_placeholders)
            it += 1


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
            
            # check matrix exponential calculation
            assert(sum(row1 - 1) < 0.01)
            assert(sum(row2 - 1) < 0.01)

            i1 = np.random.multinomial(1, row1).argmax()
            i2 = np.random.multinomial(1, row2).argmax()
            node.left.observations.append(self.tr_matrix.states[i1])
            node.right.observations.append(self.tr_matrix.states[i2])

        print("{}\t{}".format(node.left.name, node.left.observations))
        print("{}\t{}".format(node.right.name, node.right.observations))

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

        self.root.observations = []
        for i in range(size):
            index = np.random.multinomial(1, self.root.expectations).argmax()
            self.root.observations.append(self.tr_matrix.states[index])

        print("{}\t{}".format(self.root.name, self.root.observations))
        
        if self.root.left != None:
            self.simulate_(self.root)


    def estimate(self):
        """
        Runs the expectation maximization algorithm to estimate
        model parameter.
        """

        print("Building computation graph...")
        self.compute_observation_likelihoods_(self.root)
        self.log_likelihood = 0
        for node in self.child_nodes:
            self.log_likelihood += node.log_likelihood
        self.train = self.optimizer.minimize(-self.log_likelihood)


        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        #writer = tf.summary.FileWriter("log", graph=tf.get_default_graph())

        prv = np.zeros(4)
        nxt = np.array(self.sess.run([self.tr_matrix.tr_rate, self.tr_matrix.tv_rate,
                                      self.tr_matrix.on_rate, self.tr_matrix.off_rate]))
        
        print("Running EM algorithm...")
        it = 1
        while abs((nxt - prv).sum()) > 0.1:
            print("iteration: ", it)
            print("\tE step...")
            self.calculate_expectations_(self.root)
            print("\tM step...")
            self.maximize_log_likelihood_()
            prv = nxt
            nxt = np.array(self.sess.run([self.tr_matrix.tr_rate, self.tr_matrix.tv_rate,
                                          self.tr_matrix.on_rate, self.tr_matrix.off_rate]))
            print("\tParameter estimates: ", nxt)
            it += 1


    def print_parameters(self):
        print("tr_rate:\t", self.sess.run(self.tr_matrix.tr_rate))
        print("tv_rate:\t", self.sess.run(self.tr_matrix.tv_rate))
        print("on_rate:\t", self.sess.run(self.tr_matrix.on_rate))
        print("off_rate:\t", self.sess.run(self.tr_matrix.off_rate))
