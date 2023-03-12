# How I learned to love the trellis - Signal Processing Magazine, IEEE
# https://www.ece.ucdavis.edu/~bbaas/281/notes/Handout.viterbi.pdf

import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from tqdm import tqdm


def nrz_encode(bits):
    signal = np.where(bits == 1, 1, -1)
    return signal


def nrz_decode(signal):
    bits = (signal > 0).astype(int)
    return bits


def add_noise(signal, snr):
    noise_std = np.sqrt(1 / (2 * snr))
    noise = np.random.normal(0, noise_std, len(signal))
    noisy_signal = signal + noise
    return noisy_signal


def channel_sim(signal, channel, snr):
    noise_power = 10 ** (snr / 10)
    isi_signal = simulate_isi(signal, channel)
    received_signal = add_noise(isi_signal, noise_power)
    return received_signal


def calculate_ber(original_bits, decoded_bits):
    errors = np.sum(original_bits != decoded_bits)
    ber = errors / len(original_bits)
    return ber


def simulate_isi(signal, channel_response):
    """ISI effect of channel, with timeshift so input and output are aligned

        Args:
        signal (ndarray): Input signal to be transmitted over the channel.
        channel_response (ndarray): Impulse response of the channel. Example lossy channel used if omitted
    """

    # Find the index of the peak in the channel response.
    peak_index = np.argmax(np.abs(channel_response))

    # Pad the signal with zeros to accommodate the channel response.
    padded_signal = np.pad(
        signal, (0, len(channel_response) - 1), mode='constant')

    # Convolve the padded signal with the channel response.
    output_signal = np.convolve(padded_signal, channel_response)

    # Roll and trim the output signal to match the input signal.
    output_signal = output_signal[peak_index: len(signal)+peak_index]

    return output_signal


def ffe(signal, taps):
    return simulate_isi(signal, taps)


def dfe(signal, taps):
    tap_length = len(taps)

    feedback = np.zeros(tap_length)  # feedback buffer

    dfe_signal = np.zeros_like(signal)

    for i in range(len(signal)):

        # apply DFE equalization based on previous symbols detected
        dfe_signal[i] = signal[i] - np.dot(feedback, taps)

        # detect the current symbol
        detected_symbol = nrz_decode(dfe_signal[i])

        # update feedback buffer accordingly
        feedback = np.roll(feedback, 1)
        feedback[0] = nrz_encode(detected_symbol)

    return dfe_signal


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


def main():
    # Simulation paramenters
    N = int(1e5)  # number of bits
    steps = 10
    snr_range = np.linspace(0, 10, steps)  # range of SNR values

    ber_raw = np.zeros(len(snr_range))
    ber_isi = np.zeros(len(snr_range))
    ber_ffe = np.zeros(len(snr_range))
    ber_dfe = np.zeros(len(snr_range))
    ber_mlse = np.zeros(len(snr_range))

    # Transmit Signal
    np.random.seed(0)   # lock seed for repeatable results
    bits = np.random.randint(0, 2, N)  # generate random bits
    signal = nrz_encode(bits)  # encode the bits using NRZ
    channel = np.array([1., .9, .5, 0.2, -0.15])

    # Received signal at each SNR
    received_signal = []
    for i in tqdm(range(len(snr_range)), desc='ideal', unit_scale=N//1e3, unit='kbit'):
        received_signal.append(channel_sim(signal, channel, snr_range[i]))      
        received_signal_no_isi = channel_sim(signal, np.ones(1), snr_range[i])
        ber_raw[i] = calculate_ber(bits, nrz_decode(received_signal_no_isi))      

    for i in tqdm(range(len(snr_range)), desc='isi', unit_scale=N//1e3, unit='kbit'):
        ber_isi[i] = calculate_ber(bits, nrz_decode(received_signal[i]))

    for i in tqdm(range(len(snr_range)), desc='ffe', unit_scale=N//1e3, unit='kbit'):
        ffe_signal = ffe(received_signal[i], [1, -.5, +.05, +.08])
        ber_ffe[i] = calculate_ber(bits, nrz_decode(ffe_signal))

    for i in tqdm(range(len(snr_range)), desc='dfe', unit_scale=N//1e3, unit='kbit'):
        dfe_signal = dfe(received_signal[i], [.9, .5, 0.2, -0.15])
        ber_dfe[i] = calculate_ber(bits, nrz_decode(dfe_signal))

    for i in tqdm(range(len(snr_range)), desc='mlse', unit_scale=N//1e3, unit='kbit'):
        traceback = len(channel)*5
        mlse_detections = mlse(received_signal[i], channel, traceback)[traceback:]
        ber_mlse[i] = calculate_ber(bits[:len(mlse_detections)], mlse_detections)

    # Plotting
    plt.semilogy(snr_range, ber_raw, label='ideal', linestyle='--')
    plt.semilogy(snr_range, ber_dfe, label='dfe')
    plt.semilogy(snr_range, ber_ffe, label='ffe')
    plt.semilogy(snr_range, ber_isi, label='isi')
    plt.semilogy(snr_range, ber_mlse, label='mlse')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('SNR vs BER for NRZ link over a lossy channel')
    plt.grid(which='major',  linestyle='-')
    plt.grid(which='minor', color='dimgrey', linestyle='--')
    plt.legend()
    plt.show()
    #plt.savefig('snr_vs_ber.png')


main()
