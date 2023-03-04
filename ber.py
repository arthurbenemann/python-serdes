import numpy as np
import matplotlib.pyplot as plt
import timeit

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

def calculate_ber(original_bits, decoded_bits):
    errors = np.sum(original_bits != decoded_bits)
    ber = errors / len(original_bits)
    return ber

def main():
    N = int(1e6)  # number of bits
    snr_range = np.linspace(0, 10, 11)  # range of SNR values
    ber = np.zeros(len(snr_range))

    bits = np.random.randint(0, 2, N)  # generate random bits
    signal = nrz_encode(bits)  # encode the bits using NRZ
    for i in range(len(snr_range)):
        noisy_signal = add_noise(signal, 10 ** (snr_range[i] / 10))  # add noise to simulate lossy channel
        decoded_bits = nrz_decode(noisy_signal)  # decode the noisy signal using NRZ
        ber[i] = calculate_ber(bits, decoded_bits)  # calculate BER for current SNR


    plt.semilogy(snr_range, ber)
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('SNR vs BER for NRZ link over a lossy channel')
    plt.grid(b=True, which='major',  linestyle='-')
    plt.grid(b=True, which='minor', color='dimgrey', linestyle='--')
    plt.show()

print(timeit.timeit(main, number=20))
