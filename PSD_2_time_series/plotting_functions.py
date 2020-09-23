import matplotlib.pyplot as plt
from utils import *

params = {'legend.fontsize': 16,
          'axes.labelsize': 17,
          'xtick.labelsize': 17,
          'ytick.labelsize': 17}

plt.rc('text', usetex=False)
plt.rcParams.update(params)


def plot_measured_NoiseSpectrums_SSB(psd, freq, n_coast, noise_type, part, savefig=False, target_value=False, target=0, cmpt_psd = False, at_freq =0.0, step=500 ):
    fig, ax1 = plt.subplots(figsize=(8, 6))

    if noise_type == 'PN':
        ax1.plot(freq, psd, c='mediumblue')
        ax1.set_ylabel('Phase Noise (dBc/Hz)')
    else:
        ax1.plot(freq, psd, c='mediumblue')
        ax1.set_ylabel('Amplitude Noise (dBc/Hz)')

    ax1.set_xscale('log')
    #ax1.set_yscale('log')
    ax1.set_xlim(1e3, 1.1e6)
    ax1.set_ylim(-145, 0)
    ax1.set_xlabel('Offset Frequency (Hz)')

    if target_value:
        ax1.axhline(target, c='r', linestyle='dashed', label='Target value: {} dBc/Hz'.format(target))
    ax1.axvline(10e3, c='grey', linestyle='dashed', label='10 kHz')
    ax1.axvline(0.18 * 43.45e3, c='g', linestyle='dashdot',
                label='Betatron frequency, ' + r'$f_b$=' + '{:.3}kHz'.format(0.18 * 43.45))

    my_mean, my_std = find_average_psd_around_freq(freq, psd, at_freq, step)
    ax1.text(3e4, -60, 'PSD at ' + r'$f_b = {:.2f} \pm {:.2f}$'.format(my_mean, my_std), fontsize=16)

    ax1.legend()
    if noise_type == 'PN':
        ax1.set_title('Phase noise: MD5, Coast {}, Part {}'.format(n_coast, part), fontsize=17)
    else:
        ax1.set_title('Amplitude noise: MD5, Coast {}, Part {}'.format(n_coast, part), fontsize=17)
    plt.tight_layout()

    if savefig:
        plt.savefig('Measured_spectrum_MD5_coast_{}_noexc_{}_Part{}.png'.format(n_coast, noise_type, part))
    else:
        plt.show()