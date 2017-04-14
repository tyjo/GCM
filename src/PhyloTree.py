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
        self.initial_distribution = np.array([ 1. / self.num_states for st in tr_matrix.states ])
        self.log_likelihood = 0

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

        if self.is_leaf_node(root) and root.observations != None and self.num_obs == 0:
                self.num_obs = len(root.observations)

        elif self.is_leaf_node(root) == None and root.observations != None and self.num_obs != len(root.observations):
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
                if t1 not in tr_matrices: tr_matrices[t1] = self.tr_matrix.tr_matrix(t1)
                if t2 not in tr_matrices: tr_matrices[t2] = self.tr_matrix.tr_matrix(t2)
                m1 = tr_matrices[t1]
                m2 = tr_matrices[t2]

                for i in range(self.num_obs):
                    for fr in range(self.num_states):
                        prob_left = 0
                        prob_right = 0
                        assert abs(m1[fr].sum() - 1) < 0.01, "Bad transition matrix"
                        assert abs(m2[fr].sum() - 1) < 0.01, "Bad transition matrix"
                        for to in range(self.num_states):
                            # Be sure to change back to self.tr_matrix.states[to][0]
                            if self.tr_matrix.states[to] == root.left.observations[i]:
                                prob_left += m1[fr][to]
                            if self.tr_matrix.states[to] == root.right.observations[i]:
                                prob_right += m2[fr][to]
                            #if self.tr_matrix.states[to][0] == root.left.observations[i]:
                            #    prob_left += m1[fr][to]
                            #if self.tr_matrix.states[to][0] == root.right.observations[i]:
                            #    prob_right += m2[fr][to]
                        root.prob_left_descendants_[i][fr] = prob_left * prob_right

            else:
                bottom_up(root.left)
                bottom_up(root.right)

                t1 = root.left.length
                t2 = root.right.length
                if t1 not in tr_matrices: tr_matrices[t1] = self.tr_matrix.tr_matrix(t1)
                if t2 not in tr_matrices: tr_matrices[t2] = self.tr_matrix.tr_matrix(t2)
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
            return

        bottom_up(root.left)
        bottom_up(root.right)

        t1 = root.left.length
        t2 = root.right.length
        if t1 not in tr_matrices: tr_matrices[t1] = self.tr_matrix.tr_matrix(t1)
        if t2 not in tr_matrices: tr_matrices[t2] = self.tr_matrix.tr_matrix(t2)
        m1 = tr_matrices[t1]
        m2 = tr_matrices[t2]

        for i in range(self.num_obs):
            root.prob_left_descendants_[i] = m1.dot(root.left.prob_left_descendants_[i])
            root.prob_right_descendants_[i] = m2.dot(root.right.prob_left_descendants_[i])
            root.expectations_[i] = root.prob_left_descendants_[i] * root.prob_right_descendants_[i] * self.initial_distribution
            root.expectations_[i] /= root.expectations_[i].sum()

        top_down(root.left, root.prob_right_descendants_, root.expectations_)
        top_down(root.right, root.prob_left_descendants_, root.expectations_)


    def compute_complete_log_likelihood_(self, param):
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
                    #possible_states = tf.constant([1. if root.observations[i] == self.tr_matrix.states[j][0] else 0. \
                    #                                  for j in range(self.num_states)], dtype=tf.float32)
                    possible_states = np.array([1. if root.observations[i] == self.tr_matrix.states[j] else 0. \
                                                      for j in range(self.num_states)])
                    A = self.tr_matrix.tr_matrix(root.length) * possible_states \
                        + np.array([1. for i in range(self.num_states)]) - possible_states
                    #if np.isnan(np.log(A).sum()):
                    #    np.set_printoptions(edgeitems=16)
                    #    print(possible_states)
                    #    print(self.tr_matrix.tr_matrix(root.length))
                    return np.sum(np.matmul(np.log(A).T, expectations[i]))
            else:
                for i in range(self.num_obs):
                    log_likelihood = np.sum(np.matmul(np.log(self.tr_matrix.tr_matrix(root.length)).T, expectations[i]))
                    return log_likelihood + compute_helper(root.left, root.expectations_) \
                                          + compute_helper(root.right, root.expectations_)

        self.tr_matrix = tm.TransitionMatrix(param[0], param[1], param[2], param[3])
        log_likelihood = 0
        for i in range(self.num_obs):
            log_likelihood += (self.root.expectations_[i]*np.log(self.initial_distribution)).sum()
        log_likelihood += compute_helper(self.root.left, self.root.expectations_) \
                        + compute_helper(self.root.right, self.root.expectations_)
        self.log_likelihood = log_likelihood
        print(log_likelihood)
        return -log_likelihood


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
                    root.left.expectations_[i] = np.array([1. if root.left.observations[i] == self.tr_matrix.states[j][0] else 0. \
                                                       for j in range(self.num_states)])
                    root.right.expectations_[i] = np.array([1. if root.right.observations[i] == self.tr_matrix.states[j][0] else 0. \
                                                       for j in range(self.num_states)])
                    left = tr_matrix.tr_matrix(root.left.length) * root.left.expectations_[i]
                    right = tr_matrix.tr_matrix(root.right.length) * root.right.expectations_[i]
                    root.expectations_[i] = left.sum(axis=1)*right.sum(axis=1)

            else:
                compute_helper(root.left)
                compute_helper(root.right)
            
                ones = np.ones((self.num_states, self.num_states))
                for i in range(self.num_obs):   
                    left = tr_matrix.tr_matrix(root.left.length)*root.left.expectations_[i]
                    right = tr_matrix.tr_matrix(root.right.length)*root.right.expectations_[i]
                    root.expectations_[i] = (right.dot(ones)*left).sum(axis=1)
            
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
                                #method="nelder-mead")
                                method="cobyla",
                                options={"rhobeg":0.4},
                                constraints = [ {'type': 'ineq', 'fun': lambda x: x > 0.001},
                                                {'type': 'ineq', 'fun': lambda x: x > 0.001},
                                                {'type': 'ineq', 'fun': lambda x: x > 0.001},
                                                {'type': 'ineq', 'fun': lambda x: x > 0.001}])
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

        print("{}\t{}".format(node.left.name, "".join(obs[0] for obs in node.left.simulated_obs)))
        print("{}\t{}".format(node.right.name, "".join(obs[0] for obs in node.right.simulated_obs)))
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
        prv = np.zeros(4)
        nxt = np.array([self.tr_matrix.tr_rate, self.tr_matrix.tv_rate,
                        self.tr_matrix.on_rate, self.tr_matrix.off_rate])
        print("Running...")
        self.maximize_log_likelihood_()
        print("\tParameter estimates: ", nxt)
        

    def print_parameters(self):
        print("tr_rate:\t", self.tr_matrix.tr_rate)
        print("tv_rate:\t", self.tr_matrix.tv_rate)
        print("on_rate:\t", self.tr_matrix.on_rate)
        print("off_rate:\t", self.tr_matrix.off_rate)
