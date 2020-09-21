import numpy as np


def compute_PSD_whiteNoise(N, frev, muNoise, stdNoise):
    t = np.linspace(0, N / frev, N)  # time in seconds
    delta_t = t[1] - t[0]
    freq = np.fft.fftfreq(N, delta_t)
    delta_f = freq[1] - freq[0]
    print('sampling Delta t = {} s'.format(delta_t))
    print('sampling frequency Delta f = {} s'.format(delta_f))

    #### Iterate over 1000 times to increase the precision of the FFT
    fft_list = []
    for i in range(1000):
        y_noise = np.random.normal(muNoise, stdNoise, N)
        fft = np.fft.fft(y_noise)
        fft_list.append(fft)
    mean_dft = np.mean(np.abs(fft_list)**2, axis=0)
    psd = mean_dft/(delta_f*(N**2))

    return psd, freq


def compute_PSD(signal, N, frev):
    t = np.linspace(0, N / frev, N)  # time in seconds
    delta_t = t[1] - t[0]
    freq = np.fft.fftfreq(N, delta_t)
    delta_f = freq[1] - freq[0]
    print('sampling Delta t = {} s'.format(delta_t))
    print('sampling frequency Delta f = {} s'.format(delta_f))
    fft = np.fft.fft(signal)
    psd = np.abs(fft)**2/(delta_f*(N**2))

    return psd, freq


def ssb_2_dsb(L):
    """
    Convert the single sideband, SBB, measurement of the noise, L(f) in dBc/Hz, to the double
    sideband density, DSB,  S(f) in rad^2/Hz. According to the IEEE standards:
    S(f) = 2*10^(L(f)/10) in rad^2/Hz, where L(f) in dBc/Hz.
    IMPORTANT: In the definitions here, both L(f) and S(f) are one sided, ie only positive frequencies.
    """

    S = 2*10**(L/10)
    return S
