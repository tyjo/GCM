import numpy as np
import tensorflow as tf
from scipy.special import factorial

class TransitionMatrix:

    def __init__(self, tr_rate, tv_rate, on_rate, off_rate):
        # transition rate
        self.tr_rate = tf.Variable(tr_rate, dtype=tf.float32, name="tr_rate")

        # transversion rate
        self.tv_rate = tf.Variable(tv_rate, dtype=tf.float32, name="tv_rate")

        # switch OFF/ON rate
        self.on_rate = tf.Variable(on_rate, dtype=tf.float32, name="on_rate")

        # switch ON/OFF rate
        self.off_rate = tf.Variable(off_rate, dtype=tf.float32, name="off_rate")

        # states
        self.states = ["A000", "A100", "A010", "A001", "A110", "A101", "A011", "A111",
                       "C000", "C100", "C010", "C001", "C110", "C101", "C011", "C111",
                       "G000", "G100", "G010", "G001", "G110", "G101", "G011", "G111",
                       "T000", "T100", "T010", "T001", "T110", "T101", "T011", "T111"]

        #nucleotide states
        self.nucl_states = "ACGT"

        # valid transitions
        self.transitions = {"A": "G",
                            "G": "A",
                            "C": "T",
                            "T": "C"}

        # valid switches
        self.on_switch = {"000": ["100", "010", "001"],
                          "100": ["110", "101"],
                          "010": ["110", "011"],
                          "001": ["101", "011"],
                          "110": ["111"],
                          "101": ["111"],
                          "011": ["111"],
                          "111": []}

        self.off_switch = {"000": [],
                           "100": ["000"],
                           "010": ["000"],
                           "001": ["000"],
                           "110": ["100", "010"],
                           "101": ["100", "001"],
                           "011": ["010", "001"],
                           "111": ["110", "101", "001"]}

        # store previously computed transistion matrices
        # tr_matrices[time] = e^{Qt}
        self.tr_matrices = {}

        # rate matrix
        self.Q = tf.pack([ [self.rate_matrix(s1, s2) for s2 in self.states] for s1 in self.states ], name="rate_matrix")


    def off_diagional(self, fr, to):
        """
        Returns off diagional entries (fr, to) in the rate matrix.
        """
        assert(fr != to)

        # Transition
        if fr[0] != to[0] and fr[1:] == to[1:] and self.transitions[fr[0]] == to[0]:
            if (self.nucl_states.find(to[0]) < self.nucl_states.find(fr[0])):
                index = self.nucl_states.find(to[0])
            else:
                index = self.nucl_states.find(to[0])-1
            return int(fr[1:][index])*self.tr_rate

        # Transversion
        elif fr[0] != to[0] and fr[1:] == to[1:] and self.transitions[fr[0]] != to[0]:
            if (self.nucl_states.find(to[0]) < self.nucl_states.find(fr[0])):
                index = self.nucl_states.find(to[0])
            else:
                index = self.nucl_states.find(to[0])-1
            return int(fr[1:][index])*self.tv_rate

        # OFF => ON
        elif fr[0] == to[0] and to[1:] in self.on_switch[fr[1:]]:
            return self.on_rate

        # ON => OFF
        elif fr[0] == to[0] and to[1:] in self.off_switch[fr[1:]]:
            return self.off_rate
        
        else:
            return tf.constant(0, dtype=tf.float32)

    
    def rate_matrix(self, fr, to):
        """
        Returns entry (fr, to) in rate matrix.
        """
        if fr != to:
            return self.off_diagional(fr, to)

        diag = tf.constant(0, dtype=tf.float32)
        for state in self.states:
            if state != fr:
                diag += self.off_diagional(fr, state)
        return -diag


    def matrix_power(self, M, n):
        """
        Compute matrix powers by repeated squaring and store the result.
        """
        # Find first power of 2
        if n == 1:
            return M
        elif n == 2:
            ret = tf.matmul(M, M)
            return ret
        
        ret = tf.eye(len(self.states), dtype=tf.float32, name="matrix_power_" + str(n))
        while np.log2(n) != np.floor(np.log2(n)):
            ret = tf.matmul(ret, M)
            n -= 1
        
        # Repeated squaring
        sqr = tf.matmul(M, M)
        n /= 2
        while n > 1:
            sqr = tf.matmul(sqr, sqr)
            n /= 2

        ret = tf.matmul(ret, sqr)
        return ret


    def tr_matrix(self, time):
        """
        Compute the Tensorflow graph for e^{Qt} for t = time and stores the result.
        """
        if time in self.tr_matrices:
            return self.tr_matrices[time]

        t = tf.constant(time, dtype=tf.float32)
        ret = tf.eye(len(self.states), dtype=tf.float32, name="tr_matrix_" + str(time))
        Q = self.Q / 2
        for i in range(1, 10):
            ret += self.matrix_power(Q, i)*tf.pow(t, i) / tf.constant(factorial(i), dtype=tf.float32)
        ret = self.matrix_power(ret, 2)
        self.tr_matrices[time] = ret
        return ret


    def tr_prob(self, fr, to, time):
        """
        Returns the transition probility (fr, to) in P(time)
        """
        return self.tr_matrix(time)[self.states.index(fr), self.states.index(to)]



