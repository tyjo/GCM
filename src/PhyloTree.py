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

    def calculate_likelihoods_(self, root):
        """
        Calculates the expected state of each ancestral node.
        """
        print ('root', root.name)

        # Given left and right likelihoods  calculate root likelihoode
        tmp_for_root_likelihoods=np.zeros([len(self.tr_matrix.states)])
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

        train=tf.train.GradientDescentOptimizer(0.01).minimize(tf.reduce_sum(self.log_likelihood))
        print ("train ", train)


        prv = np.zeros(4)
        nxt = np.array(self.sess.run([self.tr_matrix.tr_rate, self.tr_matrix.tv_rate,
                                      self.tr_matrix.on_rate, self.tr_matrix.off_rate]))
        print (nxt)
    