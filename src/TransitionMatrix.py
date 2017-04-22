import numpy as np
import scipy.linalg

class TransitionMatrix:

    def __init__(self, tr_rate, tv_rate, on_rate, off_rate):

        # transition rate
        self.tr_rate = tr_rate

        # transversion rate
        self.tv_rate = tv_rate

        # switch OFF/ON rate
        self.on_rate = on_rate

        # switch ON/OFF rate
        self.off_rate = off_rate

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

        # rate matrix
        self.Q =  self.compute_rate_matrix()

        # stored transition matrices to prevent recomputation
        self.tr_matrices = {}

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
            return 0.

    
    def rate_matrix(self, fr, to):
        """
        Returns entry (fr, to) in rate matrix.
        """
        if fr != to:
            return self.off_diagional(fr, to)

        diag = 0.
        for state in self.states:
            if state != fr:
                diag += self.off_diagional(fr, state)
        return -diag


    def compute_rate_matrix(self):
        return np.array([ [self.rate_matrix(s1, s2) for s2 in self.states] for s1 in self.states ])


    def tr_matrix(self, time):
        """
        Compute the for e^{Qt} for t = time.
        """
        if not time in self.tr_matrices:
            self.tr_matrices[time] = scipy.linalg.expm(self.Q*time)
        return self.tr_matrices[time]





