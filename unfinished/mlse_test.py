# How I learned to love the trellis - Signal Processing Magazine, IEEE
# https://www.ece.ucdavis.edu/~bbaas/281/notes/Handout.viterbi.pdf

import numpy as np
from itertools import product


def mlse(signal, channel_response, traceback_length):

    channel_length = len(channel_response)

    # Generate all possible transmitted sequences
    m = np.array(list(product([-1, 1], repeat=3))).reshape((8, 3))

    # branch metrics considering n,n-1,n-2 indexing to messages
    decoder_trellis = np.dot(m, channel_response)

    # transition matrix
    transition_matrix = np.array([[0, 1], [2, 3], [0, 1], [2, 3]])

    K = 2**(len(channel_response)-1)   # number of states

    # Γ same as the number of states = len(channel)
    state_metrics = np.zeros(K)

    # each row is a path ?
    path_mem = np.zeros((traceback_length, K), dtype=bool)

    # detected signal
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
        even_metrics = metric[::2]
        odd_metrics = metric[1::2]
        decisions = odd_metrics < even_metrics

        # Select
        state_metrics = np.where(
            decisions, odd_metrics, even_metrics)  # select lower metric

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


def lowblue(vector):
    # format the numbers in the vector with 1 decimal place and leading zeros
    formatted_vector = [f"{num:0>.1f}" for num in vector]

    # find the index of the lowest value
    min_index = np.argmin(vector)

    # create a string with ANSI escape codes to print in blue
    blue_text = "\033[34m{}\033[0m".format(formatted_vector[min_index])

    # replace the lowest value in the vector with the blue text
    formatted_vector[min_index] = blue_text

    # join the formatted numbers into a string and return it
    return "[" + " ".join(formatted_vector) + "]"


np.set_printoptions(precision=1, suppress=True,
                    floatmode='fixed', linewidth=250)

# path memmory


np.random.seed(0)   # lock seed for repeatable results
o = np.where(np.random.randint(0, 2, 100) == 1, 1, -1)  # generate random bits
# o = np.array([1, 1,  1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
h = np.array([1, 0.8, 0.2])
r2 = np.convolve(o, h)

traceback = 10
detections = mlse(r2, h, traceback)


transmitted = (o > 0).astype(int)
received = detections[traceback:].astype(int)
errors = (received != transmitted[:len(received)]).astype(int)

print(transmitted)
print(received)
print(errors, '\t errors:', np.sum(errors))
