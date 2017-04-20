# Numerically compute the transition matrix in numpy
# given constants below

import numpy as np
from scipy.special import factorial
import scipy.linalg

tr_rate = 0.1
tv_rate = 0.5
on_rate = 0.1
off_rate = 1
time = 1
num_terms = 200

states = ["A000", "A100", "A010", "A001", "A110", "A101", "A011", "A111",
          "C000", "C100", "C010", "C001", "C110", "C101", "C011", "C111",
          "G000", "G100", "G010", "G001", "G110", "G101", "G011", "G111",
          "T000", "T100", "T010", "T001", "T110", "T101", "T011", "T111"]

transitions = {"A": "G",
               "G": "A",
               "C": "T",
               "T": "C"}

on_switch = {"000": ["100", "010", "001"],
             "100": ["110", "101"],
             "010": ["110", "011"],
             "001": ["101", "011"],
             "110": ["111"],
             "101": ["111"],
             "011": ["111"],
             "111": []}

off_switch = {"000": [],
              "100": ["000"],
              "010": ["000"],
              "001": ["000"],
              "110": ["100", "010"],
              "101": ["100", "001"],
              "011": ["010", "001"],
              "111": ["110", "101", "001"]}

nucl_states = "ACGT"


def off_diagional(fr, to):
    """
    Returns off diagional entries (fr, to) in the rate matrix.
    """
    assert(fr != to)
    # Transition
    if fr[0] != to[0] and fr[1:] == to[1:] and transitions[fr[0]] == to[0]:
        if (nucl_states.find(to[0]) < nucl_states.find(fr[0])):
            index = nucl_states.find(to[0])
        else:
            index = nucl_states.find(to[0])-1
        return int(fr[1:][index])*tr_rate

    # Transversion
    elif fr[0] != to[0] and fr[1:] == to[1:] and transitions[fr[0]] != to[0]:
        if (nucl_states.find(to[0]) < nucl_states.find(fr[0])):
            index = nucl_states.find(to[0])
        else:
            index = nucl_states.find(to[0])-1
        return int(fr[1:][index])*tv_rate
    
    # OFF => ON
    elif fr[0] == to[0] and to[1:] in on_switch[fr[1:]]:
        return on_rate

    # ON => OFF
    elif fr[0] == to[0] and to[1:] in off_switch[fr[1:]]:
        return off_rate
        
    else:
        return 0

def rate_matrix(fr, to):
    """
    Returns entry (fr, to) in rate matrix.
    """
    if fr != to:
        return off_diagional(fr, to)

    diag = 0
    for state in states:
        if state != fr:
            diag += off_diagional(fr, state)
    return -diag

if __name__ == "__main__":
    I = np.identity(len(states))
    Q = np.array([ [rate_matrix(s1, s2) for s2 in states] for s1 in states ]) / 2
    ret = I
    for i in range(1, num_terms):
        ret += np.linalg.matrix_power(Q, i)*np.power(time,i) / factorial(i)
    print(np.linalg.matrix_power(ret, 2))
    print(scipy.linalg.expm(Q*time))