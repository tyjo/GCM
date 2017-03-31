import numpy as np
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
        # node. We use this in the bottom up part  of the calculation for 
        # expectations. Left descendants are those that are direct descendants
        # of the node. Right  descendants are those observations that are not 
        # directly below the node. The root node is an exception, where left
        # decendants and right decendants are true descendants.
        self.prob_left_descendants_ = None
        self.prob_right_descendants_ = None

        # Numerical conditional expectation of the node given observations and
        # current parameter values.
        self.expectations_ = None

        # Placeholders for expectations to pass to Tensorflow for maximizing 
        # the expected complete log likelihood.
        self.placeholders_ = None




class PhyloTree:

    def __init__(self, root, tr_matrix):
        """
        root: the root node of the phylogenetic tree
        tr_matrix: transition matrix
        """
        self.root = root
        assert root.left != None, "Trees must have at least one child node"
        
        self.tr_matrix = tr_matrix
        self.num_states = len(self.tr_matrix.states)
        self.check_tree_(root)
        self.num_obs = 0 # The length of the observed sequence in leaf nodes
        # This should be the initial distribution
        self.initial_distribution = np.array([ 1. / self.num_states for st in tr_matrix.states ])

        ## Tensorflow parameters
        self.sess = None
        # the tensorflow graph of the expected complete log likelihood, prevents recomputation
        self.log_likelihood = None
        # dictionary of placeholders to pass expected values to session as feed_dict
        self.expected_value_ph = {}
        # numerical optimization routine
        self.optimizer = tf.train.GradientDescentOptimizer(0.01)
        self.train = None

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

        self.check_tree_(root.left)
        self.check_tree_(root.right)


    def setup_(self, root):
        """
        Set up tensorflow placeholders to pass to maximization routine.
        These placeholders are initialized to expectations.
        """
        if root == None:
            return

        root.placeholders_ = tf.placeholder(tf.float64, shape=self.num_states)
        
        if self.is_leaf_node(root) and self.num_obs == 0:
            self.num_obs = len(root.observations)
            
        elif self.is_leaf_node(root) and self.num_obs == 0:
            self.num_obs = len(root.observations)

        elif self.is_leaf_node(root) == None and self.num_obs != len(root.observations):
            raise AssertionError("Leaf nodes must all have the same number of observations")

        self.setup_(root.left)
        self.setup_(root.right)


    def get_leaf_nodes_(self, root):
        """
        Makes a list of leaf nodes.
        """
        if root.left.left == None:
            self.leaf_nodes.append(root.left)
            self.leaf_nodes.append(root.right)

        else:
            self.get_leaf_nodes_(root.left)
            self.get_leaf_nodes_(root.right)


    def is_leaf_node(self, node):
        return node.left == None


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

                if t1 in tr_matrices:
                    m1 = tr_matrices[t1]
                else:
                    tr_matrices[t1] = m1 = self.sess.run(self.tr_matrix.tr_matrix(t1))

                if t1 in tr_matrices:
                    m2 = tr_matrices[t2]
                else:
                    tr_matrices[t2] = m2 = self.sess.run(self.tr_matrix.tr_matrix(t2))

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
                if t1 in tr_matrices:
                    m1 = tr_matrices[t1]
                else:
                    tr_matrices[t1] = m1 = self.sess.run(self.tr_matrix.tr_matrix(t1))

                if t1 in tr_matrices:
                    m2 = tr_matrices[t2]
                else:
                    tr_matrices[t2] = m2 = self.sess.run(self.tr_matrix.tr_matrix(t2))

                for i in range(self.num_obs):
                    #v_L = root.left.prob_left_descendants_[i]
                    #v_R = root.right.prob_left_descendants_[i]
                    for fr in range(self.num_states):
                        assert abs(m1[fr].sum() - 1) < 0.01, "Bad transition matrix"
                        x = root.left.prob_left_descendants_[i] * m1[fr]
                        y = root.right.prob_left_descendants_[i] * m2[fr]
                        root.prob_left_descendants_[i][fr] = x.dot(np.eye(self.num_states).dot(y.T))


        def top_down(node):
            """
            Computes expecations of left and right children after call to bottom_up.
            """
            # Compute expectations for left node
            if self.is_leaf_node(node.left):
                return

            assert(node.expectations_ != None)
            assert(node.left.prob_left_descendants_ != None)
            assert(node.right.prob_left_descendants_ != None)
            t1 = root.left.length
            t2 = root.right.length
            m1 = tr_matrices[t1]
            m2 = tr_matrices[t2]
            node.left.prob_right_descendants_ = np.zeros((self.num_obs, self.num_states))
            node.right.prob_right_descendants_ = np.zeros((self.num_obs, self.num_states))
            node.left.expectations_ = np.zeros((self.num_obs, self.num_states))
            node.right.expectations_ = np.zeros((self.num_obs, self.num_states))
            for i in range(self.num_obs):
                node.left.prob_right_descendants_[i] = tr_matrices[root.left.length].dot(node.prob_right_descendants_[i])
                node.left.expectations_[i] = node.left.prob_left_descendants_[i] * node.left.prob_right_descendants_[i] * tr_matrices[root.left.length].dot(node.expectations_[i])
                node.left.expectations_[i] /= node.left.expectations_[i].sum()
                self.expected_value_ph[node.left.placeholders_[i]] = node.left.expectations_[i]

                node.right.prob_right_descendants_[i] = tr_matrices[root.right.length].dot(node.prob_right_descendants_[i])
                node.right.expectations_[i] = node.right.prob_right_descendants_[i] * node.right.prob_right_descendants_[i] * tr_matrices[root.right.length].dot(node.expectations_[i])
                node.right.expectations_[i] /= node.right.expectations_[i].sum()
                self.expected_value_ph[node.right.placeholders_[i]] = node.right.expectations_[i]

            if not self.is_leaf_node(root.left):
                top_down(node.left)
                top_down(node.right)
        
        # Save computed transition matrices. Dictionary
        # from time to np.array.
        tr_matrices = {}
        bottom_up(root)
        root.prob_right_descendants_ = np.zeros((self.num_obs, self.num_states))
        t1 = root.left.length
        t2 = root.right.length
        m1 = tr_matrices[t1]
        m2 = tr_matrices[t2]
        root.left.prob_right_descendants_ = np.zeros((self.num_obs, self.num_states))
        root.right.prob_right_descendants_ = np.zeros((self.num_obs, self.num_states))
        root.expectations_ = np.zeros((self.num_obs, self.num_states))
        root.left.expectations_ = np.zeros((self.num_obs, self.num_states))
        root.right.expectations_ = np.zeros((self.num_obs, self.num_states))
        for i in range(self.num_obs):
            root.prob_right_descendants_[i] = tr_matrices[root.right.length].dot(root.right.prob_left_descendants_[i])
            root.expectations_[i] = root.prob_left_descendants_[i] * root.prob_right_descendants_[i] * self.initial_distribution
            root.expectations_[i] /= root.expectations_[i].sum()
            self.expected_value_ph[root.placeholders_[i]] = root.expectations_[i]

            root.left.prob_right_descendants_[i] = tr_matrices[root.left.length].dot(root.prob_right_descendants_[i])
            root.left.expectations_[i] = root.left.prob_left_descendants_[i] * root.left.prob_right_descendants_[i] * tr_matrices[root.left.length].dot(root.expectations_[i])
            root.left.expectations_[i] /= root.left.expectations_[i].sum()
            self.expected_value_ph[root.left.placeholders_[i]] = root.left.expectations_[i]

            root.right.prob_right_descendants_[i] = tr_matrices[root.right.length].dot(root.prob_left_descendants_[i])
            root.right.expectations_[i] = root.right.prob_right_descendants_[i] * root.right.prob_right_descendants_[i] * tr_matrices[root.right.length].dot(root.expectations_[i])
            root.right.expectations_[i] /= root.right.expectations_[i].sum()
            self.expected_value_ph[root.right.placeholders_[i]] = root.right.expectations_[i]

        top_down(root.left)
        top_down(root.right)


    def compute_complete_log_likelihood_(self, root):
        """
        Computes the graph corresponding to the complete log likelihood and
        stores it in self.log_likelihood
        """
        pass



    def maximize_log_likelihood_(self):
        """
        Maximizes the expected complete log conditional.
        Maximization proceeds by gradient ascent.
        """
        prv = 0
        nxt = self.sess.run(self.log_likelihood, feed_dict=self.expected_value_ph)
        it = 1
        while abs(nxt - prv) > 0.1:
            print("\tlog_likelihood =", nxt)
            prv = nxt
            self.sess.run(self.train, feed_dict=self.expected_value_ph)
            nxt = self.sess.run(self.log_likelihood, feed_dict=self.expected_value_ph)
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

        #self.root.observations = []
        #for i in range(size):
        #    index = np.random.multinomial(1, self.root.expectations).argmax()
        #    self.root.observations.append(self.tr_matrix.states[index])

        print("{}\t{}".format(self.root.name, self.root.observations))
        
        if self.root.left != None:
            self.simulate_(self.root)


    def estimate(self):
        """
        Runs the expectation maximization algorithm to estimate
        model parameter.
        """

        print("Building computation graph...")
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.calculate_expectations_(self.root)
        return

        #self.compute_observation_likelihoods_(self.root)
        #self.log_likelihood = 0
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
