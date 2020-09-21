import matplotlib.pyplot as plt

params = {'legend.fontsize': 16,
          'axes.labelsize': 17,
          'xtick.labelsize': 17,
          'ytick.labelsize': 17}

plt.rc('text', usetex=False)
plt.rcParams.update(params)


def plot_measured_NoiseSpectrums(psd, freq, n_coast, noise_type, part, savefig=False):
    fig, ax1 = plt.subplots(figsize=(8, 6))

    if noise_type == 'PN':
        ax1.plot(freq, psd, c='mediumblue')
        ax1.set_ylabel('Phase Noise (dBc/Hz)')
    else:
        ax1.plot(freq, psd, c='mediumblue')
        ax1.set_ylabel('Amplitude Noise (dBc/Hz)')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(1e3, 1.1e6)
    ax1.set_xlabel('Offset Frequency (Hz)')

    ax1.axvline(10e3, c='grey', linestyle='dashed', label='10 kHz')
    ax1.axvline(0.18 * 43.45e3, c='g', linestyle='dashdot',
                label='Betatron frequency, ' + r'$f_b$=' + '{:.3}kHz'.format(0.18 * 43.45))

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