import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

saveflag = False
N = 10001  # number of sampling points/turns

signal = np.random.normal(0, 1e-8, N)
print(np.std(signal))

# save the original sequence of noise kicks
if saveflag:
    with open('PN_original_kicks.pkl', 'wb') as f:
        pkl.dump(signal, f)
    f.close()

t = np.linspace(0, N/43.45e3, N)  # time in seconds
Dt = t[1]-t[0]

plt.figure()
plt.plot(t, signal, '.')
plt.xlabel('time (s)')
plt.ylabel('Amplitude (a.u.)')

# FFT of the white noise signal
fft = np.fft.fft(signal)
freq = np.fft.fftfreq(N, Dt)

plt.figure()
plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.abs(fft)), '.-')
plt.xlim(-50, 50)
plt.yscale('log')
plt.xlabel('Hz')
plt.ylabel('Amplitude (a.u.)')

plt.figure()
plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.angle(fft)), '.')
plt.plot(freq, np.angle(fft), '.')
plt.xlabel('Hz')
plt.ylabel('angles (rad)')

phi = 2*np.pi*np.random.uniform(0,1, int(len(freq)))
phi[0]=0

fft_2 = np.abs(fft)*np.exp(1j*phi)
plt.figure()
plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.abs(fft)), '.-',c='C0')
plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.abs(fft_2)), '.-',c='C1')
plt.xlim(-50, 50)
plt.yscale('log')
plt.xlabel('Hz')
plt.ylabel('Amplitude (a.u.)')

# Hermitian, to obtain real signal as a result
fft_temp = np.empty_like(fft_2)
fft_temp[0] = 0
fft_temp[1:N//2] = fft_2[1:N//2]
fft_temp[N//2+2:] = np.conj(np.flip(fft_2[1:N//2]))

signal_new = np.fft.ifft(fft_temp)

# keep only the real part, the imaginary is zero anyway, e-25
if saveflag:
    with open('PN_reconstructed_kicks.pkl', 'wb') as f2:
        pkl.dump(np.real(signal_new), f2)
    f2.close()


plt.figure()
plt.plot(t, signal, '.', c='C0', label='original')
plt.plot(t, np.real(signal_new), '.',  c='C1', label='reconstructed, Re')
plt.plot(t, np.imag(signal_new), '.', c='C2', label='reconstructed, Imag')
plt.xlabel('time')
plt.legend()

print(np.imag(signal_new))

plt.show(block=False)
input()


