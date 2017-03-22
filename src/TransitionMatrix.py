import numpy as np
import tensorflow as tf
from scipy.special import factorial

class TransitionMatrix:

    def __init__(self, tr_rate, tv_rate, on_rate, off_rate):
        # transition rate
        self.tr_rate = tf.Variable(tr_rate, dtype=tf.float64)

        # transversion rate
        self.tv_rate = tf.Variable(tv_rate, dtype=tf.float64)

        # switch OFF/ON rate
        self.on_rate = tf.Variable(on_rate, dtype=tf.float64)

        # switch ON/OFF rate
        self.off_rate = tf.Variable(off_rate, dtype=tf.float64)

        # states
        self.states = ["A000", "A100", "A010", "A001", "A110", "A101", "A011", "A111",
                       "C000", "C100", "C010", "C001", "C110", "C101", "C011", "C111",
                       "G000", "G100", "G010", "G001", "G110", "G101", "G011", "G111",
                       "T000", "T100", "T010", "T001", "T110", "T101", "T011", "T111"]

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

    def off_diagional(self, fr, to):
        """
        Returns off diagional entries (fr, to) in the rate matrix.
        """
        assert(fr != to)
        # Transition
        if fr[0] != to[0] and fr[1:] == to[1:] and self.transitions[fr[0]] == to[0]:
            return self.tr_rate

        # Transversion
        elif fr[0] != to[0] and fr[1:] == to[1:] and self.transitions[fr[0]] != to[0]:
            return self.tv_rate

        # OFF => ON
        elif fr[0] == to[0] and to[1:] in self.on_switch[fr[1:]]:
            return self.on_rate

        # ON => OFF
        elif fr[0] == to[0] and to[1:] in self.off_switch[fr[1:]]:
            return self.off_rate
        
        else:
            return tf.constant(0, dtype=tf.float64)

    
    def rate_matrix(self, fr, to):
        """
        Returns entry (fr, to) in rate matrix.
        """
        if fr != to:
            return self.off_diagional(fr, to)

        diag = tf.constant(0, dtype=tf.float64)
        for state in self.states:
            if state != fr:
                diag += self.off_diagional(fr, state)
        return -diag

    def tr_matrix(self, time):
        """
        Compute the Tensorflow graph for e^{Qt} for t = time and stores the result.
        """
        t = tf.constant(time, dtype=tf.float64)
        I = tf.eye(len(self.states), dtype=tf.float64)
        Q = tf.pack([ [self.rate_matrix(s1, s2) for s2 in self.states] for s1 in self.states ])
        ret = I
        for i in range(1, 50):
            ret += self.matrix_power(Q, i)*tf.pow(t, i) / tf.constant(factorial(i), dtype=tf.float64)
        return ret

    def matrix_power(self, Q, n):
        ret = Q
        for i in range(1,n):
            ret = tf.matmul(ret, Q)
        return ret

