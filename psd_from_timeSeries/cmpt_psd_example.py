import numpy as np
import matplotlib.pyplot as plt

def compute_PSD_whiteNoise(N, frev, muNoise, stdNoise):
    t = np.linspace(0, N / frev, N)  # time in seconds
    delta_t = t[1] - t[0]
    freq = np.fft.fftfreq(N, delta_t)
    delta_f = freq[1] - freq[0]
    print('sampling Delta t = {} s'.format(delta_t))
    print('frequency resolution, Delta f = {} s'.format(delta_f))

    #### Iterate over 1000 times to increase the precision of the FFT
    fft_list = []
    for i in range(1000):
        y_noise = np.random.normal(muNoise, stdNoise, N)
        fft = np.fft.fft(y_noise)
        fft_list.append(fft)
    mean_dft = np.mean(np.abs(fft_list)**2, axis=0)
    psd = mean_dft/(delta_f*(N**2))

    return psd, freq