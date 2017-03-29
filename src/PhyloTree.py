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
        self.ancestor = None
        self.likelihoods = None

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
        self.current_node = None
        # This should be the initial distribution

        self.sess = None
        # the tensorflow graph of the expected complete log likelihood, prevents recomputation
        self.log_likelihood = None
        self.train = None

    def find_node_by_name_(self,root,name):
        if (root.name == name):
            self.current_node=root
        else:
            if root.left != None:
                self.find_node_by_name_(root.left,name)
            if root.right != None:
                self.find_node_by_name_(root.right,name)
                
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
            index = np.random.multinomial(1,np.array([ 1. / self.num_states for st in self.tr_matrix.states ])).argmax()
            self.root.observations.append(self.tr_matrix.states[index])

        print("{}\t{}".format(self.root.name, self.root.observations))
        
        if self.root.left != None:
            self.simulate_(self.root)

    def calculate_likelihoods_(self, root):
        """
        Calculates the expected state of each ancestral node.
        """
        print ('root', root.name)

        # Given left and right likelihoods  calculate root likelihoode
        tmp_for_root_likelihoods=[0 for i in range(len(self.tr_matrix.states))]
        #tr1 = self.sess.run(self.tr_matrix.tr_matrix(root.left.length))
        #tr2 = self.sess.run(self.tr_matrix.tr_matrix(root.right.length))
        tr1 = self.tr_matrix.tr_matrix(root.left.length)
        tr2 = self.tr_matrix.tr_matrix(root.right.length)

        for fr in self.tr_matrix.states:
            tmp_left=0
            tmp_right=0
            fr_index = self.tr_matrix.states.index(fr)
            for to in self.tr_matrix.states:
                to_index = self.tr_matrix.states.index(to)
                #print ('to  ', to, ' tr1 ', self.sess.run(tr1[fr_index,to_index]),' exp_to ',root.left.likelihoods[to_index])
                print ('to  ', to, ' tr1 ', tr1[fr_index,to_index],' exp_to ',root.left.likelihoods[to_index])
                tmp_left  = tmp_left + tr1[fr_index, to_index] * root.left.likelihoods[to_index]
                tmp_right = tmp_right + tr2[fr_index, to_index] * root.right.likelihoods[to_index]
            #print (fr, ' tmp_left ', self.sess.run(tmp_left), ' tmp_right ',self.sess.run(tmp_right))
            #print (self.sess.run(tf.multiply(tmp_left,tmp_right)))
            tmp_for_root_likelihoods[fr_index]=tmp_left*tmp_right
            root.likelihoods=tmp_for_root_likelihoods

        
        if root == self.root:
            tmp_sum_likelihood=0
            for fr in self.tr_matrix.states:
                fr_index = self.tr_matrix.states.index(fr)
                tmp_sum_likelihood += root.likelihoods[fr_index]#*self.root.placeholder[fr_index]
            self.log_likelihood=tf.log(tmp_sum_likelihood)#*self.root.placeholder[fr_index]

        else:        
            #assert(abs(root.likelihoods.sum() - 1) < 0.01)
            self.find_node_by_name_(self.root,str(int(root.name)-1))
            self.calculate_likelihoods_(self.current_node)

    def compute_observation_likelihoods_pruning_(self,root,site_number):
        if root.left == None:
            tmp_root_likelihoods=np.zeros([len(self.tr_matrix.states)])
            for fr in self.tr_matrix.states:
                fr_index = self.tr_matrix.states.index(fr)
                if root.observations[site_number][0] == fr[0]:
                    tmp_root_likelihoods[fr_index]=1./8.
            root.likelihoods =tmp_root_likelihoods
        else:
            self.compute_observation_likelihoods_pruning_(root.left,site_number)
            self.compute_observation_likelihoods_pruning_(root.right,site_number)



    def estimate(self):

        print("Building computation graph...")

        self.current_node=self.root.right
        
        self.compute_observation_likelihoods_pruning_(self.root,0)
        self.calculate_likelihoods_(self.current_node)

        self.train=tf.train.GradientDescentOptimizer(0.01).minimize(-self.log_likelihood)
        #print ("train ", train)
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.sess.run(self.tr_matrix.tr_rate.assign(0.1))
        self.sess.run(self.tr_matrix.tv_rate.assign(0.1))
        self.sess.run(self.tr_matrix.on_rate.assign(0.1))
        self.sess.run(self.tr_matrix.off_rate.assign(0.1))
   
        prv = np.zeros(4)
        nxt = np.array(self.sess.run([self.tr_matrix.tr_rate, self.tr_matrix.tv_rate,
                                      self.tr_matrix.on_rate, self.tr_matrix.off_rate]))
        print (nxt)
        self.sess.run(self.train)
        nxt = self.sess.run(self.log_likelihood)
        print (nxt)
    def print_parameters(self):
        print("tr_rate:\t", self.sess.run(self.tr_matrix.tr_rate))
        print("tv_rate:\t", self.sess.run(self.tr_matrix.tv_rate))
        print("on_rate:\t", self.sess.run(self.tr_matrix.on_rate))
        print("off_rate:\t", self.sess.run(self.tr_matrix.off_rate))
    