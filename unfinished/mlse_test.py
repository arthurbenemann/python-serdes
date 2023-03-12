# How I learned to love the trellis - Signal Processing Magazine, IEEE
# https://www.ece.ucdavis.edu/~bbaas/281/notes/Handout.viterbi.pdf

import numpy as np
from itertools import product


def mlse(signal, channel_response, traceback_length):

    # K: number of states in channel state machine
    length = len(channel_response)
    K = 2**(length-1)

    # all possible transmitted sequences (considering n,n-1,n-2)
    m = np.array(list(product([-1, 1], repeat=length))).reshape((K*2, length))

    # decoder treliss, based on ideal channel response to each message
    decoder_trellis = np.dot(m, channel_response)

    # transition matrix[previous state,decision] = previous_state
    transition_matrix = np.tile(np.arange(K), 2).reshape((K, 2))

    # initialize memory for loop
    state_metrics = np.zeros(K)
    path_mem = np.zeros((traceback_length, K), dtype=bool)
    detections = []

    # run through samples
    for sample in signal:

        # double the vector by replicating end-to-end
        previous_state_metrics = np.tile(state_metrics, 2)

        # Calculate Branch metrics
        branch_metrics = np.abs(decoder_trellis - sample)

        # Add
        metric = previous_state_metrics + branch_metrics

        # Compare
        even = metric[::2]
        odd = metric[1::2]
        decisions = odd < even

        # Select
        state_metrics = np.where(decisions, odd, even)  # select lower metric

        # Update path memmory
        path_mem = np.roll(path_mem, 1, axis=0)
        path_mem[0, :] = decisions

        # Find most likely path
        likely_state = np.argmin(state_metrics)

        for test in path_mem:
            traceback_decision = int(test[likely_state])
            likely_state = transition_matrix[likely_state, traceback_decision]

        # decide bit based on oldest state on likely path
        bit = likely_state >= K/2
        detections = np.append(detections, bit)

        # debug
        # print(lowblue(np.transpose(state_metrics)),
        #       lowblue(branch_metrics), bit)
        # print(path_mem)
        # print()
    return detections


np.set_printoptions(precision=1, suppress=True,
                    floatmode='fixed', linewidth=250)


np.random.seed(0)   # lock seed for repeatable results
o = np.where(np.random.randint(0, 2, 100) == 1, 1, -1)  # generate random bits
# o = np.array([1, 1,  1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
h = np.array([1, -1,2,1,3,2,0.1])
r2 = np.convolve(o, h)
r2 += np.random.normal(0, 1.0, len(r2))  # noise

traceback = 10
detections = mlse(r2, h, traceback)


transmitted = (o > 0).astype(int)
received = detections[traceback:].astype(int)
errors = (received != transmitted[:len(received)]).astype(int)

print(transmitted)
print(received)
print(errors, '\t errors:', np.sum(errors))
