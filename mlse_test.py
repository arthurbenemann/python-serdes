import numpy as np


def mlse(signal, channel_response, traceback_length):

    K = 2**len(channel_response)   # number of states

    branch_metrics = fun(channel_response)  # needs function, size 2 * K

    # Γ same as the number of states = len(channel)
    state_metrics = np.zeros(K)

    path_history = np.zeros((K, traceback_length))  # each row is a path ?

    detections = np.zeros_like(signal)

    # run through samples
    for n in range(signal):

        # Add Compare Select
        for j in range(state_metrics):

            # need to figure out mapping
            one = np.abs(signal[n] - state_metrics[x] + j)
            zero = np.abs(signal[n] - state_metrics[y] + j)

            if (one < zero):
                state_metrics[z] = state_metrics[x]
                path_history[z].append(1)
            else:
                state_metrics[z] = state_metrics[y]
                path_history[z].append(0)

        # select path
        detections[n] = path_history[np.argmin(state_metrics)]
        shift_path_history()

    return detections


    # path memmory
r = np.array([1., 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
h = np.array([1, 0.8, 0.5, 0.2])

mlse(r, h)
