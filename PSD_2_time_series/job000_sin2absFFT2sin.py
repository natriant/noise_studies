import numpy as np
import matplotlib.pyplot as plt

T = 0.5 # s
N = 1000001 # 10001
f = 20 # Hz

t = np.linspace(0, T, N)
Dt = t[1]-t[0]

signal = np.sin(2*np.pi*f*t)
print(np.std(signal))
plt.plot(t, signal, '.')
plt.xlabel('time (s)')
plt.ylabel('Amplitude (a.u.)')

fft = np.fft.fft(signal)
freq = np.fft.fftfreq(N, Dt)


plt.figure()
plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.abs(fft)), '.-')
plt.xlim(-50, 50)
plt.yscale('log')
plt.xlabel('Hz')
plt.ylabel('Amplitude (a.u.)')

plt.figure()
#plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.angle(fft)), '.')
plt.plot(freq, np.angle(fft), '.')
plt.xlabel('Hz')
plt.ylabel('angles (rad)')

phi = 2*np.pi*np.random.uniform(0,1, int(len(freq)))
phi[0]=np.angle(fft[0])

fft_2 = np.abs(fft)*np.exp(1j*phi)
plt.figure()
plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.abs(fft)), '.-',c='C0')
plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.abs(fft_2)), '.-',c='C1')
plt.xlim(-50, 50)
plt.yscale('log')
plt.xlabel('Hz')
plt.ylabel('Amplitude (a.u.)')

fft_temp = np.empty_like(fft_2)
fft_temp[0] = 0
fft_temp[1:N//2] = fft_2[1:N//2]
fft_temp[N//2+2:] = np.conj(np.flip(fft_2[1:N//2]))

signal_new = np.fft.ifft(fft_temp)
plt.figure()
plt.plot(t, signal, c='C0')
plt.plot(t, np.real(signal_new), c='C1')
#plt.plot(t, np.imag(signal_new), c='C1')
plt.xlabel('time')



plt.show(block=False)
input()


