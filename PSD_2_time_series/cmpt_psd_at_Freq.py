from utils import *
import numpy as np
import pandas as pd
from plotting_functions import *
from utils import *

# Load the measured spectrum in a data frame
path_to_data = '/home/natalia/quatantine_phd/From_Themis_and_Phillipe/measured_noise_spectrum/recieved_1Sep2020/'
file_name = 'coast4EX-0dBm.csv'
df = pd.read_csv(path_to_data + file_name)
print('Dictionary keys {}'.format(df.keys()))

# Create arrays with the PSD and frequency information
Sxx_load = np.array(df['Phase Noise (dBc/Hz)']) # PSD in dBc/Hz
freq_load = np.array(df['Offset Frequency (Hz)'])  # equally spaced in logarithmic scale

# Find the average psd value over the frequency you want, fb, over a wanted frequency range, step
fb = 0.24*43.45e3 # Hz
print('Looking for PSD at {}Hz'.format(fb))
step = 500  # Hz +-

my_mean, my_std = find_average_psd_around_freq(freq_load, Sxx_load, fb, step)
print('The PSD value around {} Hz over a range of {} Hz has mean {} dBc/Hz and std {}'.format(fb, step, my_mean, my_std))

print('PSD at {}Hz is {} rad^2Hz'.format(fb, ssb_2_dsb(my_mean)))