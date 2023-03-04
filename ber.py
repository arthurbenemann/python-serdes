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

def simulate_isi(signal, channel_response=np.array([1.,.1,.05,0,0,-0.11])):
    """ISI effect of channel, with timeshift so input and output are aligned
        
        Args:
        signal (ndarray): Input signal to be transmitted over the channel.
        channel_response (ndarray): Impulse response of the channel. Example lossy channel used if omitted
    """
    
    # Normalize channel response
    channel_response /= np.linalg.norm(channel_response)

    # Find the index of the peak in the channel response.
    peak_index = np.argmax(np.abs(channel_response))

    # Pad the signal with zeros to accommodate the channel response.
    padded_signal = np.pad(signal, (0, len(channel_response) - 1), mode='constant')

    # Convolve the padded signal with the channel response.
    output_signal = np.convolve(padded_signal, channel_response)

    # Roll and trim the output signal to match the input signal.
    output_signal = output_signal[peak_index : len(signal)+peak_index]

    return output_signal

def main():
    N = int(1e6)  # number of bits
    snr_range = np.linspace(0, 10, 11)  # range of SNR values
    ber = np.zeros(len(snr_range))
    ber_isi = np.zeros(len(snr_range))

    np.random.seed(0) 
    bits = np.random.randint(0, 2, N)  # generate random bits
    signal = nrz_encode(bits)  # encode the bits using NRZ
    for i in range(len(snr_range)):
        isi_signal = simulate_isi(signal,np.array([1.,.1,.05,0,-0.15]))
        noisy_signal = add_noise(isi_signal, 10 ** (snr_range[i] / 10))  # add noise to simulate lossy channel
        decoded_bits = nrz_decode(noisy_signal)  # decode the noisy signal using NRZ
        ber_isi[i] = calculate_ber(bits, decoded_bits)  # calculate BER for current SNR

        noisy_signal = add_noise(signal, 10 ** (snr_range[i] / 10))  # add noise to simulate lossy channel
        decoded_bits = nrz_decode(noisy_signal)  # decode the noisy signal using NRZ
        ber[i] = calculate_ber(bits, decoded_bits)  # calculate BER for current SNR


    plt.semilogy(snr_range, ber,label='baseline')
    plt.semilogy(snr_range, ber_isi, label='isi')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('SNR vs BER for NRZ link over a lossy channel')
    plt.grid(b=True, which='major',  linestyle='-')
    plt.grid(b=True, which='minor', color='dimgrey', linestyle='--')
    plt.legend()
    plt.show()

print(timeit.timeit(main, number=1))
