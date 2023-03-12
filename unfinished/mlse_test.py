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
    detections = np.zeros_like(signal).astype(int)

    # run through samples
    for i in range(len(signal)):

        # Calculate Branch metrics (negative transitions followed by positive)
        branch_metrics = np.abs(decoder_trellis - signal[i])

        # Add - Compare - Select
        metric = np.tile(state_metrics, 2) + branch_metrics

        even = metric[::2]
        odd = metric[1::2]
        decisions = odd < even

        state_metrics = np.where(decisions, odd, even)  # select lower metric

        # Update path memmory
        path_mem = np.roll(path_mem, 1, axis=0)
        path_mem[0, :] = decisions

        # Traceback with path mem starting from most likely state
        likely = np.argmin(state_metrics)

        for test in path_mem:
            likely = transition_matrix[likely, int(test[likely])]

        # decide bit based on oldest state on likely path
        detections[i] = (likely >= K/2)
    return detections


np.set_printoptions(precision=1, suppress=True,
                    floatmode='fixed', linewidth=250)


np.random.seed(0)   # lock seed for repeatable results
o = np.where(np.random.randint(0, 2, 200) == 1, 1, -1)  # generate random bits
h = np.array([1, -1, 2, 1, 3, 2, 0.1, 2, 3, -5, 4])
r2 = np.convolve(o, h)
r2 += np.random.normal(0, 5, len(r2))  # noise

traceback = len(h)*3
detections = mlse(r2, h, traceback)


transmitted = (o > 0).astype(int)
received = detections[traceback:].astype(int)
errors = (received != transmitted[:len(received)]).astype(int)

# print(transmitted)
# print(received)
print(errors, '\t errors:', np.sum(errors))
