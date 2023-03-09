import numpy as np
import matplotlib.pyplot as plt
import timeit
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

def ffe(signal, taps):
    return simulate_isi(signal,taps)

def dfe(signal, taps):
    tap_length = len(taps)

    feedback = np.zeros(tap_length)  # feedback buffer

    dfe_signal = np.zeros_like(signal)

    for i in range(len(signal)):

        # apply DFE equalization based on previous symbols detected
        dfe_signal[i]= signal[i] - np.dot(feedback,taps)

        # detect the current symbol  
        detected_symbol = nrz_decode(dfe_signal[i])
        
        # update feedback buffer accordingly
        feedback = np.roll(feedback, 1)
        feedback[0] = nrz_encode(detected_symbol)

    return dfe_signal

def main():
    # Simulation paramenters
    N = int(1e6)  # number of bits
    snr_range = np.linspace(0, 10, 6)  # range of SNR values

    ber_raw = np.zeros(len(snr_range))
    ber_isi = np.zeros(len(snr_range))
    ber_ffe = np.zeros(len(snr_range))
    ber_dfe = np.zeros(len(snr_range))

    # Transmit Signal
    np.random.seed(0)   # lock seed for repeatable results 
    bits = np.random.randint(0, 2, N)  # generate random bits
    signal = nrz_encode(bits)  # encode the bits using NRZ


    # SNR simulations
    for i in tqdm(range(len(snr_range))):

        # Channel simultation
        noise_power = 10 ** (snr_range[i] / 10)
        noisy_signal = add_noise(signal, noise_power)                       
        isi_signal = simulate_isi(signal,np.array([1.,.5,.2,0,-0.15]))
        noisy_isi_signal = add_noise(isi_signal,noise_power)

        # Equalization, 
        ffe_signal = ffe(noisy_isi_signal, [1,-.5,+.05,+.08])
        dfe_signal = dfe(noisy_isi_signal,[.5,.2,0,-.15])

        # Detectors and BER calculation
        ber_raw[i] = calculate_ber(bits, nrz_decode(noisy_signal) )  # calculate BER for current SNR
        ber_isi[i] = calculate_ber(bits, nrz_decode(noisy_isi_signal)  )  # calculate BER for current SNR
        ber_ffe[i] = calculate_ber(bits, nrz_decode(ffe_signal))  # calculate BER for current SNR
        ber_dfe[i] = calculate_ber(bits, nrz_decode(dfe_signal))  # calculate BER for current SNR

    # Plotting
    plt.semilogy(snr_range, ber_raw, label='raw')
    plt.semilogy(snr_range, ber_dfe, label='dfe-4tap')
    plt.semilogy(snr_range, ber_ffe, label='ffe-4tap')
    plt.semilogy(snr_range, ber_isi, label='isi')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('SNR vs BER for NRZ link over a lossy channel')
    plt.grid(which='major',  linestyle='-')
    plt.grid(which='minor', color='dimgrey', linestyle='--')
    plt.legend()
    plt.show()

print(timeit.timeit(main, number=1))
