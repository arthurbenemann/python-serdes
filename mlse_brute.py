import numpy as np

def mlse_brute(signal, channel_response, traceback_length):

    # Generate all possible transmitted sequences
    m = np.array(np.meshgrid(*[[-1, 1]]*traceback_length)
                 ).T.reshape(-1, traceback_length)

    # calculate expected received sequence
    u = np.apply_along_axis(lambda row: np.convolve(
        row, channel_response, mode='full')[:traceback_length+1], axis=1, arr=m)

    # pad receive array
    received_padded = np.pad(signal, (0, traceback_length))

    # initialize detected sequence with zeros
    detected_seq = np.zeros(len(signal))

    # perform MSLD
    for n in range(traceback_length, len(signal)+traceback_length):
        # message subset to be matched
        sub = received_padded[n-traceback_length:n+1]

        # calculate error for each possible sequence against subset
        error = result = np.sum(np.abs(u - sub), axis=1)

        # detection is the first symbol of sequence that minimizes error
        detected_seq[n-traceback_length] = m[np.argmin(error), 0]

    return detected_seq


r = np.array([1., 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
h = np.array([1, 0.8, 0.5, 0.2])
received = np.convolve(r, h)[:len(r)]
received += np.random.normal(0, .1, len(received))
L = 5

s_hat = mlse_brute(received, h, L)
print(received)
print(r)
print(s_hat)
print(np.sum(r != s_hat))
