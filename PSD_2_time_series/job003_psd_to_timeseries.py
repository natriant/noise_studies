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
freq_load = freq_load[:213] # 254 the simulations so far
Sxx_load = Sxx_load[:213]

print(freq_load[-1])
#plot_measured_NoiseSpectrums(Sxx_load, freq_load, 3, 'PN', 'C', savefig=False)

# Linear interpolation of the frequency array, such as the values are equally spaced linearly
N, frev = 10001, 43.45e3 # number of points, 1002 such as the length of freq_pos is odd
t = np.linspace(0, N/43.45e3, N)  # time in seconds
delta_t = t[1]-t[0] # this should be fixed
#freq = np.linspace(0, N/t[-1], N) # [0, 2frev]
freq = np.fft.fftfreq(N, delta_t)  # positive and negative frequencies
delta_f = freq[1]-freq[0]
print('Delta t = {} Hz'.format(delta_t))
print('Delta f = {} Hz'.format(delta_f))


# for the linear interpolation, we need only the positive frequencies
#freq_pos = np.fft.fftshift(freq[:N//2])  # not sure if i should include the zero term or not
freq_pos = freq[0:N//2]
print('positive frequencies ={}'.format(len(freq_pos)))
Sxx = np.interp(freq_pos, freq_load, Sxx_load)

plt.figure()
plt.plot(freq_load/1000, Sxx_load, c='C0', label='original psd')
#plt.plot(np.fft.fftshift(freq_pos)/1000, np.fft.fftshift(Sxx), '--', c='C1', label='linear interpolant')
plt.plot(freq_pos/1000, Sxx, '--', c='C1', label='linear interpolant')
plt.yscale('log')
plt.legend()
plt.show()

# Go back to time series
fft_amplitude = np.sqrt(Sxx*delta_f*(N**2))

# Create random angles
phi = 2*np.pi*np.random.uniform(0, 1, int(len(freq_pos)))

# Create the new FFT
fft_2 = fft_amplitude*np.exp(1j*phi) # note, this does contain the zero frequency term. a[1:N//2]
plt.figure()
plt.plot(np.imag(fft_2), '.')
plt.ylabel('angles (rad)')
plt.tight_layout()

# Hermitian, to obtain real signal as a result
fft_temp = np.concatenate((np.empty_like(fft_2), np.empty_like(fft_2)), axis=0)
fft_temp = np.concatenate((fft_temp, np.empty_like(fft_2[1:2])), axis=0)
fft_temp[0] = 0   # 0, the zero frequency term. only real
fft_temp[1:(N//2+1)] = fft_2[0:(N//2)] # from 1 to 5000
fft_temp[N//2+1:] = np.conj(np.flip(fft_2[0:(N//2)]))  # from 5001 to 10001


signal_new = np.fft.ifft(fft_temp) # time series, noise kicks

plt.figure()
plt.plot(t, np.real(signal_new), '.', label='real part')
plt.plot(t, np.imag(signal_new), '.', label='imaginary part')  # imag expected 0
plt.title('Generated time series')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (a.u.)')
plt.tight_layout()
plt.legend()


print('generated signal mean = {}'.format(np.mean(np.real(signal_new)))) # should be zero
print('generated signal std = {}'.format(np.std(np.real(signal_new))))


# Sanity check, compute back psd
psd_new, freq_new = compute_PSD(signal_new, N, frev)


plt.figure()
plt.plot(freq_pos/1000, Sxx, '-', lw=5, c='C0', label='linear interpolant')
plt.plot(np.fft.fftshift(freq_new)/1e3, np.fft.fftshift(psd_new), '--', c='C1', label ='reconstructed')
plt.xlim(0, 30)
plt.yscale('log')
plt.ylim(1e-13, 1e-9)
plt.legend(loc=1)
plt.show()

saveflag = False
if saveflag:
    with open('PN_realNoise_v3.pkl', 'wb') as f2:
        pkl.dump(np.real(signal_new), f2)
    f2.close()
