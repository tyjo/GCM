import numpy as np
import tensorflow as tf
from scipy.special import factorial
jhdsfjhdsf
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


    def tr_prob(self, fr, time):
        """
        Takes: a starting state and time.
        Returns: a list of probabilities of moving from state fr to any state at time t.
        """
        time = tf.constant(time, dtype=tf.float64)
        prob = [tf.constant(1, dtype=tf.float64) if fr == state else tf.constant(0, dtype=tf.float64) for state in self.states]

        # Row fr in the rate matrix Q^1 assuming entries (i,j) are the rates
        # of moving from i to j
        q_fr = []
        for j in range(len(self.states)):
            q_fr.append(self.rate_matrix(fr, self.states[j]))
            prob[j] += q_fr[j]*time

        # Number of terms to evaluate in the series expansion
        for i in range(2,5):
            # At the end of each iteration q_fr is row fr in Q^{i}
            nxt = [tf.constant(0, dtype=tf.float64) for state in self.states]
            for j in range(len(self.states)):
                for k in range(len(self.states)):
                    nxt[j] += q_fr[k]*self.rate_matrix(self.states[k], self.states[j])*time / factorial(i)
                prob[j] += nxt[j]
            q_fr = nxt[:]

        return prob

    def tr_matrix(self, time):
        """
        
        """
        pass
