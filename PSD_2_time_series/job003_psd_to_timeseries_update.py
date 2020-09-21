import numpy as np
import pandas as pd
import pickle as pkl
from utils import *
from plotting_functions import *

# Load the measured spectrum in a data frame
path_to_data = '/home/natalia/quatantine_phd/From_Themis_and_Phillipe/measured_noise_spectrum/recieved_1Sep2020/'
file_name = 'coast4EX-0dBm.csv'
df = pd.read_csv(path_to_data + file_name)
print('Dictionary keys {}'.format(df.keys()))

# Convert the SSB, dBc/Hz, to DSB, rad^2/Hz
Sxx_load = ssb_2_dsb(np.array(df['Phase Noise (dBc/Hz)'])) # PSD in rad^2/Hz
freq_load = np.array(df['Offset Frequency (Hz)'])  # equally spaced in logarithmic scale

# keep the values up to 43.45e3 kHz
freq_load = freq_load[:213]
Sxx_load = Sxx_load[:213] # only for positive frequencies
plot_measured_NoiseSpectrums(Sxx_load, freq_load, 3, 'PN', 'C', savefig=False)

# Linear interpolation of the frequency array, such as the values are equally spaced linearly
N_half, frev = 10001, 43.45e3 # number of points
t = np.linspace(0, N_half/43.45e3, N_half)  # time in seconds
delta_t = t[1]-t[0]
freq_pos = np.linspace(0, N_half/t[-1], N_half) # just for the interpolation and the delta f
print('max freq = {}'.format(freq_pos[-10]))
delta_f = freq_pos[1]-freq_pos[0]
print('sampling Delta t = {} s'.format(delta_t))
print('sampling frequency Delta f = {} s'.format(delta_f))
Sxx = np.interp(freq_pos, freq_load, Sxx_load)

plt.figure()
plt.plot(freq_load/1000, Sxx_load, c='C0', label='original psd')
plt.plot(freq_pos/1000, Sxx, '--', c='C1', label='linear interpolant')
plt.yscale('log')
plt.legend()

# Consider the spectrum symmetric and create the negative frequencies and amplitudes
Sxx_neg = np.flip(Sxx[1:])
freq_neg = np.flip(-freq_pos[1:])
# the two following arrays have the format of the output of the fft
Sxx_total = np.concatenate((Sxx, Sxx_neg), axis=0)
freq_total = np.concatenate((freq_pos, freq_neg), axis=0)

plt.figure()
plt.plot(freq_pos/1000, Sxx, '--', c='C0', label='linear interpolant')
plt.plot(np.fft.fftshift(freq_total)/1000, np.fft.fftshift(Sxx_total), '--', c='C1', label='2 sided spectrum')
plt.yscale('log')
plt.legend(loc=1)

# Go back to time series
N = len(freq_total)
fft_amplitude = np.sqrt(Sxx_total*delta_f*(N**2))

# Create random angles
phi = 2*np.pi*np.random.uniform(0, 1, int(len(freq_total)))
phi[0] = 0
# Create the new FFT
fft_2 = fft_amplitude*np.exp(1j*phi)

# Hermitian, to obtain real signal as a result
fft_temp = (np.empty_like(fft_2))
fft_temp[0] = 0
fft_temp[1:N//2+1] = fft_2[1:N//2+1]
fft_temp[N//2+1:] = np.conj(np.flip(fft_2[1:N//2+1]))

signal_new = np.fft.ifft(fft_temp)  # time series, noise kicks
t2 = np.linspace(0, len(signal_new)/43.45e3, len(signal_new))

plt.figure()
plt.plot(t2, np.real(signal_new), label='real part')
plt.plot(t2, np.imag(signal_new), label='imaginary part')  # imag expected 0
plt.title('Generated time series')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (a.u.)')
plt.legend(loc=1)

print('generated signal mean = {}'.format(np.mean(np.real(signal_new)))) # should be zero
print('generated signal std = {}'.format(np.std(np.real(signal_new))))

# sanity check, psd
psd_new, freq_new = compute_PSD(signal_new, N, frev)
plt.figure()
plt.plot(freq_pos/1000, Sxx, '--', c='C0', label='linear interpolant')
plt.plot(np.fft.fftshift(freq_new)/1e3, np.fft.fftshift(psd_new), '--', c='C1', label ='reconstructed')

plt.xlim(0, 50)
plt.yscale('log')
plt.ylim(1e-13, 1e-8)
plt.show()


quit()


signal_new = np.fft.ifft(fft_temp) # time series, noise kicks
t2 = np.linspace(0, len(signal_new)/43.45e3, len(signal_new))

plt.figure()
plt.plot(t2, np.real(signal_new), label='real part')
plt.plot(t2, np.imag(signal_new), label='imaginary part')  # imag expected 0
plt.title('Generated time series')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (a.u.)')
plt.legend()
plt.show()


# sanity check, psd
psd_new, freq_new = compute_PSD(signal_new, len(signal_new), frev)
plt.figure()
plt.plot(freq/1000, Sxx, '--', c='C0', label='linear interpolant')
plt.plot(np.fft.fftshift(freq_new)/1e3, 2*np.fft.fftshift(psd_new), '--', c='C1', label ='reconstructed')
plt.xlim(0, 50)
plt.yscale('log')
plt.ylim(1e-13, 1e-9)
plt.show()

saveflag = False
if saveflag:
    with open('PN_realNoise_v2.pkl', 'wb') as f2:
        pkl.dump(np.real(signal_new), f2)
    f2.close()
