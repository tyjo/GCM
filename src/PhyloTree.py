import numpy as np
import scipy.optimize
import src.TransitionMatrix as tm

class Node:

    def __init__(self, name, length=0, observations=None):
        self.name = name
        # branch length from Node to parent
        self.length = length
        self.observations = observations
        # left and right subtrees
        self.left = None
        self.right = None

        # Vector of conditional probabilities of the subtree
        # below the node given the state of the node. Used
        # in the pruning algorithm.
        self.prob_below_ = None

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
        # The length of the observed sequence in leaf nodes. This gets set in check_tree_
        self.num_obs = 0
        # The initial distribution of the root node
        self.initial_distribution = np.array([ 1. / self.num_states for st in tr_matrix.states ])

        # If True, infers parameters based on observing both the
        # nucleotide and the switch. If false, infers parameters
        # based on nucleotide observations alone.
        self.observe_switch = False

        assert root.left != None, "Trees must have at least one child node"
        self.check_tree_(self.root)
        self.setup_(self.root)


    def check_tree_(self, root):
        """
        Make sure each node is either an internal node with 2 descendants or
        a leaf node wih no descendants. Sets self.num_obs variable.
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
        if root == None:
            return

        if self.observe_switch and self.is_leaf_node(root):
            root.prob_below_ = np.zeros((self.num_obs, self.num_states))
            for i in range(self.num_obs):
                root.prob_below_[i] = np.array([1. if root.observations[i] == self.tr_matrix.states[j] else 0. \
                                                for j in range(self.num_states)])

        if not self.observe_switch and self.is_leaf_node(root):
            root.prob_below_ = np.zeros((self.num_obs, self.num_states))
            for i in range(self.num_obs):
                root.prob_below_[i] = np.array([1. if root.observations[i][0] == self.tr_matrix.states[j][0] else 0. \
                                                for j in range(self.num_states)])

        self.setup_(root.left)
        self.setup_(root.right)


    def is_leaf_node(self, node): return node.left == None


    def compute_log_likelihood_(self, param):
        """
        Returns the log_likelihood with parameters param.
        """
        def compute_helper(root, tr_matrix):
            if self.is_leaf_node(root): return

            compute_helper(root.left, tr_matrix)
            compute_helper(root.right, tr_matrix)

            #root.prob_below_ = np.zeros((self.num_obs, self.num_states))
            left = tr_matrix.tr_matrix(root.left.length).dot(root.left.prob_below_.T)
            right = tr_matrix.tr_matrix(root.right.length).dot(root.right.prob_below_.T)
            root.prob_below_ = (left*right).T
            #for i in range(self.num_obs):
            #    left = tr_matrix.tr_matrix(root.left.length).dot(root.left.prob_below_[i])
            #    right = tr_matrix.tr_matrix(root.right.length).dot(root.right.prob_below_[i])
            #    root.prob_below_[i] = left*right

        for i in range(4):
            if param[i] < 0 or param[i] > 1.5:
                return -2147483648

        tr_matrix = tm.TransitionMatrix(param[0], param[1], param[2], param[3])
        compute_helper(self.root, tr_matrix)
        log_likelihood = np.log((self.root.prob_below_*self.initial_distribution).sum(axis=1)).sum()
        assert(log_likelihood < 0.)
        print("log likelihood      =", log_likelihood)
        print("parameter estimates =", param)
        return log_likelihood


    def maximize_log_likelihood_(self):
        """
        Minimizes the negative log likelihood using the Nelder-Mead simplex algorithm.
        """
        np.set_printoptions(edgeitems=16)
        res = scipy.optimize.minimize(lambda param: -self.compute_log_likelihood_(param), 
                                      np.array([self.tr_matrix.tr_rate,
                                                self.tr_matrix.tv_rate,
                                                self.tr_matrix.on_rate,
                                                self.tr_matrix.off_rate]),
                                      method="nelder-mead")
        # The attribute x is the vector of inferred parameter values
        return res.x


    def simulate(self, size):
        """
        Given a tree, simulates size observations along the branches.
        """
        print("Simulating tree...")
        self.root.simulated_obs = np.array( [ np.random.multinomial(1, self.initial_distribution)
                                              for i in range(size) ])

        left_transitions = self.root.simulated_obs.dot(self.tr_matrix.tr_matrix(self.root.left.length))
        right_transitions = self.root.simulated_obs.dot(self.tr_matrix.tr_matrix(self.root.right.length))
        self.root.left.simulated_obs = np.array( [ np.random.multinomial(1, left_transitions[i])
                                                   for i in range(size) ])
        self.root.right.simulated_obs = np.array([ np.random.multinomial(1, right_transitions[i])
                                                   for i in range(size) ])
        self.root.simulated_obs = None
        
        #print("{}\t{}".format(self.root.name, self.root.simulated_obs))
        self.simulate_(self.root.left)
        self.simulate_(self.root.right)


    def simulate_(self, node):
        """
        Helper function to simulate observations.
        """
        if self.is_leaf_node(node):
            node.simulated_obs = [ self.tr_matrix.states[obs.argmax()] for obs in node.simulated_obs ]
            #print("{}\t{}".format(node.name, node.simulated_obs))
            return

        #print(node.name)
        left_transitions = node.simulated_obs.dot(self.tr_matrix.tr_matrix(self.root.left.length))
        right_transitions = node.simulated_obs.dot(self.tr_matrix.tr_matrix(self.root.right.length))

        node.left.simulated_obs = np.array( [ np.random.multinomial(1, left_transitions[i])
                                                   for i in range(len(node.simulated_obs)) ])
        node.right.simulated_obs = np.array([ np.random.multinomial(1, right_transitions[i])
                                                   for i in range(len(node.simulated_obs)) ])
        node.simulated_obs = None
        
        #print("{}\t{}".format(node.left.name, node.left.simulated_obs))
        #print("{}\t{}".format(node.right.name, node.right.simulated_obs))
        self.simulate_(node.left)
        self.simulate_(node.right)


    def set_simulated_observations(self):
        """
        After simulations have been run, assigns the simulated
        observations to observations in Node. These can then be
        used to estimate the parameters of the model.
        """
        self.set_simulated_observations_(self.root)
        self.check_tree_(self.root)
        self.setup_(self.root)


    def set_simulated_observations_(self, root):
        if root == None:
            return
            
        root.observations = root.simulated_obs
        self.set_simulated_observations_(root.left)
        self.set_simulated_observations_(root.right)
    

    def estimate(self, observe_switch = False):
        """
        Runs the maximization routine to infer parameter values. If
        observe_switch is set to True, infers parameters based on both
        the switch and observeed nucleotide. If set to False, infers 
        parameter values based on the observed nucleotide alone.

        Returns the final log likelihood along with estimated parameters.
        """
        self.observe_switch = observe_switch
        np.seterr(under="raise") # raise an exception of underflow occurs
        print("Running...")

        param = self.maximize_log_likelihood_()
        lk = self.compute_log_likelihood_(param)
        return (param, lk)
        

    def print_parameters(self):
        print("tr_rate:\t", self.tr_matrix.tr_rate)
        print("tv_rate:\t", self.tr_matrix.tv_rate)
        print("on_rate:\t", self.tr_matrix.on_rate)
        print("off_rate:\t", self.tr_matrix.off_rate)
