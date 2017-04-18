import numpy as np
import os
import scipy.linalg
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
        self.initial_distribution = np.array([ 1. / self.num_states for st in tr_matrix.states ], np.float32)
        # Dictionary of node name to index
        self.index = {}
        self.setup_(self.root)
        self.check_tree_(root)
        self.num_nodes = max(self.index.values()) + 1


        #############################
        ### Tensorflow parameters ###
        #############################
        self.sess = None
        # The tensorflow graph of the expected complete log likelihood, prevents recomputation
        self.log_likelihood = 0
        # Numerical optimization routine
        self.optimizer = None
        self.gradient = None
        self.train = None

        assert root.left != None, "Trees must have at least one child node"

        # Placesholders to pass to graph
        self.observations_ph_ = tf.placeholder(tf.float32, shape=(min(self.num_obs, 200), self.num_nodes, self.num_states), name="observations")
        self.observations_ = np.zeros((self.num_obs, self.num_nodes, self.num_states), dtype=np.float32)
        self.set_observations_(self.root)
        
        self.expectations_ph_ = tf.placeholder(tf.float32, shape=(min(self.num_obs, 200), self.num_nodes, self.num_states, 1), name="expectations")
        self.expectations_ = np.zeros((self.num_obs, self.num_nodes, self.num_states, 1), dtype=np.float32)
        #self.placeholders_ = { self.observations_ph_ : self.observations_,
        #                       self.expectations_ph_ : self.expectations_}


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

        elif self.is_leaf_node(root):
            if root.observations != None and self.num_obs != len(root.observations):
                print(self.num_obs)
                print(len(root.observations))
                raise AssertionError("Leaf nodes must all have the same number of observations")

        self.check_tree_(root.left)
        self.check_tree_(root.right)


    def set_observations_(self, root):
        if root == None:
            return

        index = self.index[root.name]
        if self.is_leaf_node(root):
            for i in range(self.num_obs):
                self.observations_[i][index] = np.array([1. if root.observations[i] == self.tr_matrix.states[j][0] else 0. for j in range(self.num_states)])
                #self.observations_[i][index] = np.array([1. if root.observations[i] == self.tr_matrix.states[j] else 0. for j in range(self.num_states)])
        else:
            for i in range(self.num_obs):
                self.observations_[i][index] = np.ones(self.num_states)

        self.set_observations_(root.left)
        self.set_observations_(root.right)


    def setup_(self, root):
        """
        """
        def setup_helper(root, index):
            if root == None:
                return

            elif self.is_leaf_node(root) and root.observations != None \
                                         and self.num_obs == 0:
                self.num_obs = len(root.observations)

            self.index[root.name] = index

            setup_helper(root.left, index + 1)
            setup_helper(root.right, index + 2)

        index = 0
        self.index[root.name] = index
        setup_helper(root.left, index + 1)
        setup_helper(root.right, index + 2)


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
                        assert abs(m1[fr].sum() - 1) < 0.01, "Bad transition matrix"
                        assert abs(m2[fr].sum() - 1) < 0.01, "Bad transition matrix"
                        prob_left = 0
                        prob_right = 0
                        for to in range(self.num_states):
                            if self.tr_matrix.states[to][0] == root.left.observations[i]:
                                prob_left += m1[fr][to]
                            if self.tr_matrix.states[to][0] == root.right.observations[i]:
                                prob_right += m2[fr][to]
                            #if self.tr_matrix.states[to] == root.left.observations[i]:
                            #    prob_left += m1[fr][to]
                            #if self.tr_matrix.states[to] == root.right.observations[i]:
                            #    prob_right += m2[fr][to]
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
                        assert abs(m2[fr].sum() - 1) < 0.01, "Bad transition matrix"
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
                self.expectations_[i][self.index[node.name]] = node.expectations_[i].reshape((self.num_states, 1))

            if not self.is_leaf_node(root.left):
                top_down(node.left, node.prob_right_descendants_, node.expectations_)
                top_down(node.right, node.prob_right_descendants_, node.expectations_)
        
        # Save computed transition matrices. Dictionary from time to np.array.
        tr_matrices = {}
        root.prob_left_descendants_ = np.zeros((self.num_obs, self.num_states))
        root.prob_right_descendants_ = np.zeros((self.num_obs, self.num_states))
        root.expectations_ = np.zeros((self.num_obs, self.num_states))

        if self.is_leaf_node(root.left):
            bottom_up(root)
            for i in range(self.num_obs):
                root.expectations_[i] = root.prob_left_descendants_[i] * self.initial_distribution
                root.expectations_[i] /= root.expectations_[i].sum()
                self.expectations_[i][self.index[root.name]] = root.expectations_[i].reshape((self.num_states, 1))
            return

        bottom_up(root.left)
        bottom_up(root.right)
        t1 = root.left.length
        t2 = root.right.length
        if t1 not in tr_matrices: tr_matrices[t1] = self.sess.run(self.tr_matrix.tr_matrix(t1))
        if t2 not in tr_matrices: tr_matrices[t2] = self.sess.run(self.tr_matrix.tr_matrix(t2))
        m1 = tr_matrices[t1]
        m2 = tr_matrices[t2]

        for i in range(self.num_obs):
            root.prob_left_descendants_[i] = m1.dot(root.left.prob_left_descendants_[i])
            root.prob_right_descendants_[i] = m2.dot(root.right.prob_left_descendants_[i])
            root.expectations_[i] = root.prob_left_descendants_[i] * root.prob_right_descendants_[i] * self.initial_distribution
            root.expectations_[i] /= root.expectations_[i].sum()
            self.expectations_[i][self.index[root.name]] = root.expectations_[i]

        top_down(root.left, root.prob_right_descendants_, root.expectations_)
        top_down(root.right, root.prob_left_descendants_, root.expectations_)


    def compute_complete_log_likelihood_(self):
        """
        Computes the graph corresponding to the complete log likelihood and
        stores it in self.log_likelihood
        """
        def compute_helper(root, parent_index):
            if self.is_leaf_node(root):
                #for i in range(self.num_obs):
                for i in range(min(self.num_obs, 200)):
                    A = tf.multiply(self.tr_matrix.tr_matrix(root.length), self.observations_ph_[i][self.index[root.name]]) \
                        + tf.constant([1. for j in range(self.num_states)], tf.float32) - self.observations_ph_[i][self.index[root.name]]
                    self.log_likelihood += tf.reduce_sum(tf.matmul(tf.log(tf.transpose(A)), self.expectations_ph_[i][parent_index]))

            else:
                for i in range(min(self.num_obs, 200)):
                    self.log_likelihood[self.index[root.name]][i] += tf.reduce_sum(tf.matmul(tf.log(tf.transpose(self.tr_matrix.tr_matrix(root.length))), self.expectations_ph_[i][parent_index]))
                    compute_helper(root.left, self.index[root.name])
                    compute_helper(root.right, self.index[root.name])

        for i in range(min(self.num_obs, 200)):
            self.log_likelihood += tf.reduce_sum(self.expectations_[i][self.index[self.root.name]]*tf.log(self.initial_distribution))
        
        compute_helper(self.root.left, self.index[self.root.name])
        compute_helper(self.root.right, self.index[self.root.name])


    def maximize_log_likelihood_(self):
        """
        Minimizes the negative log complete conditional by stochastic
        gradient descent.
        """
        prv = 0
        nxt = 1
        grad = 100
        var = [self.tr_matrix.tr_rate, self.tr_matrix.tv_rate, 
               self.tr_matrix.on_rate, self.tr_matrix.off_rate]
        step_ph = tf.placeholder(tf.float32)
        update_ph = tf.placeholder(tf.float32, shape=(len(var), 2))
        update = tf.pack([tf.assign(var[i], update_ph[i][1] - step_ph*update_ph[i][0]) for i in range(len(var))])

        while scipy.linalg.norm(grad) > 10 and prv != nxt:
            step = 0.00001
            samples = [np.random.randint(0, self.num_obs) for i in range(min(self.num_obs, 200))]
            placeholders_ = { self.observations_ph_ : self.observations_[samples],
                              self.expectations_ph_ : self.expectations_[samples]}
            grad_var = self.sess.run(self.gradient, feed_dict=placeholders_)
            grad = np.array([grd[0] for grd in grad_var])
            prv = self.sess.run(self.log_likelihood, feed_dict=placeholders_)
            self.sess.run(update, {update_ph : grad_var, step_ph: step})
            nxt = self.sess.run(self.log_likelihood, feed_dict=placeholders_)
            
            # Backtracking line search
            while np.isnan(nxt) or -nxt > -prv - 0.0001*step*grad.dot(grad):
                step /= 2
                self.sess.run(update, {update_ph : grad_var, step_ph: step})
                nxt = self.sess.run(self.log_likelihood, feed_dict=placeholders_)
            
            grad_var = self.sess.run(self.gradient, feed_dict=placeholders_)
            grad = np.array([grd[0] for grd in grad_var])
            print("\tlog_likelihood =", nxt)
            print("\tgrad_var =", grad_var)
            print("\tnorm =", np.linalg.norm(grad))


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
            if self.is_leaf_node(node.left):
                node.left.simulated_obs.append(self.tr_matrix.states[i1][0])
                node.right.simulated_obs.append(self.tr_matrix.states[i2][0])
            else:
                node.left.simulated_obs.append(self.tr_matrix.states[i1])
                node.right.simulated_obs.append(self.tr_matrix.states[i2])

        if self.is_leaf_node(node.left):
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

        #print("{}\t{}".format(self.root.name, self.root.simulated_obs))
        if self.root.left != None:
            self.simulate_(self.root)


    def estimate(self):
        """
        Runs the expectation maximization algorithm to estimate
        model parameter.
        """
        np.seterr(under="raise")
        print("Building computational graph...")
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.compute_complete_log_likelihood_()
        self.optimizer = tf.train.GradientDescentOptimizer(0.001)
        #self.train = self.optimizer.minimize(-self.log_likelihood)
        self.gradient = self.optimizer.compute_gradients(-self.log_likelihood)  
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


    def set_simulated_observations(self):
        def set_simulations_helper(root):
            if self.is_leaf_node(root):
                root.observations = "".join(root.simulated_obs)
            else:
                set_simulations_helper(root.left)
                set_simulations_helper(root.right)
        
        set_simulations_helper(self.root)
        self.setup_(self.root)
        self.observations_ph_ = tf.placeholder(tf.float32, shape=(min(self.num_obs, 200), self.num_nodes, self.num_states), name="observations")
        self.observations_ = np.zeros((self.num_obs, self.num_nodes, self.num_states), dtype=np.float32)
        self.set_observations_(self.root)
        
        self.expectations_ph_ = tf.placeholder(tf.float32, shape=(min(self.num_obs, 200), self.num_nodes, self.num_states, 1), name="expectations")
        self.expectations_ = np.zeros((self.num_obs, self.num_nodes, self.num_states, 1), dtype=np.float32)


    def print_parameters(self):
        print("tr_rate:\t", self.sess.run(self.tr_matrix.tr_rate))
        print("tv_rate:\t", self.sess.run(self.tr_matrix.tv_rate))
        print("on_rate:\t", self.sess.run(self.tr_matrix.on_rate))
        print("off_rate:\t", self.sess.run(self.tr_matrix.off_rate))
