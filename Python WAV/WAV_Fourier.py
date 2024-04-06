import scipy.io.wavfile as wavfile
import scipy
import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal
from scipy.signal import butter, lfilter

ORDER = 3
def twos_complement_hex(number, bit_length):
    if number < 0:
        complement = (1 << bit_length) + number
    else:
        complement = number
    return format(complement, '#06x')

############################# Create a bandpass filter function #############################
def bandpass_filter (
        data, 
        lowfreq, 
        hifreq, 
        fs, 
        order = 5):
    max_freq = fs/2
    low = lowfreq / max_freq
    high = hifreq / max_freq
    b, a = butter(order, [low, high], btype ='bandpass')
    filtered = lfilter(b, a, data)
    return filtered

############################# Create a 8-band Equilizer #############################
def equilizer_8band (
        data, 
        fs, 
        gain1 = 0, 
        gain2 = 0, 
        gain3 = 0, 
        gain4 = 0, 
        gain5 = 0, 
        gain6 = 0, 
        gain7 = 0, 
        gain8 = 0 ):
    # sub_bass    = bandpass_filter(data, 20, 49, fs, order = 3) * 10 ** (gain1/20)
    # bass        = bandpass_filter(data, 50, 199, fs, order = 3) * 10 ** (gain2/20)
    # upper_bass  = bandpass_filter(data, 200, 799, fs, order = 3) * 10 ** (gain3/20)
    # mid_range   = bandpass_filter(data, 800, 1999, fs, order = 3) * 10 ** (gain4/20)
    # upper_mid   = bandpass_filter(data, 2000, 3999, fs, order = 3) * 10 ** (gain5/20)
    # presence    = bandpass_filter(data, 4000, 6999, fs, order = 3) * 10 ** (gain6/20)
    # mid_treble  = bandpass_filter(data, 7000, 11999, fs, order = 3) * 10 ** (gain7/20)
    # air         = bandpass_filter(data, 12000, 20000, fs, order = 3) * 10 ** (gain8/20)
    # eq_data = sub_bass + bass + upper_bass + mid_range + upper_mid + presence + mid_treble + air

    band1 = bandpass_filter(data, 20, 49, fs, order = ORDER) * 10 ** (gain1/20)
    band2 = bandpass_filter(data, 50, 199, fs, order = ORDER) * 10 ** (gain2/20)
    band3 = bandpass_filter(data, 200, 499, fs, order = ORDER) * 10 ** (gain3/20)
    band4 = bandpass_filter(data, 500, 999, fs, order = ORDER) * 10 ** (gain4/20)
    band5 = bandpass_filter(data, 1000, 1499, fs, order = ORDER) * 10 ** (gain5/20)
    band6 = bandpass_filter(data, 1500, 2499, fs, order = ORDER) * 10 ** (gain6/20)
    band7 = bandpass_filter(data, 2500, 4199, fs, order = ORDER) * 10 ** (gain7/20)
    band8 = bandpass_filter(data, 4299, 7999, fs, order = ORDER) * 10 ** (gain8/20)
    eq_data = band1 + band2 + band3 + band4 + band5 + band6 + band7 + band8
    return eq_data


############################# Import a test audio (*.wav) #############################
s_rate, data = wavfile.read("Example 16bit.wav") 
print(s_rate)
# Fourier 
FFT = abs(scipy.fft.fft(data))
freqs = scipy.fft.fftfreq(len(FFT), (1.0/s_rate))

# Equilizing
equalized = equilizer_8band(data, s_rate, 0, 0, 0, 0, 0, 0, 0, 0)

# Tap value calculating
tap1 = np.array([twos_complement_hex(x, 16) for x in np.rint(signal.firwin(ORDER + 1, [20, 49], pass_zero='bandpass', fs = s_rate )* 32768).astype(int)])
tap2 = np.array([twos_complement_hex(x, 16) for x in np.rint(signal.firwin(ORDER + 1, [50, 199], pass_zero='bandpass', fs = s_rate )* 32768).astype(int)])
tap3 = np.array([twos_complement_hex(x, 16) for x in np.rint(signal.firwin(ORDER + 1, [200, 499], pass_zero='bandpass', fs = s_rate )* 32768).astype(int)])
tap4 = np.array([twos_complement_hex(x, 16) for x in np.rint(signal.firwin(ORDER + 1, [500, 999], pass_zero='bandpass', fs = s_rate )* 32768).astype(int)])
tap5 = np.array([twos_complement_hex(x, 16) for x in np.rint(signal.firwin(ORDER + 1, [1000, 1499], pass_zero='bandpass', fs = s_rate )* 32768).astype(int)])
tap6 = np.array([twos_complement_hex(x, 16) for x in np.rint(signal.firwin(ORDER + 1, [1500, 2499], pass_zero='bandpass', fs = s_rate )* 32768).astype(int)])
tap7 = np.array([twos_complement_hex(x, 16) for x in np.rint(signal.firwin(ORDER + 1, [2500, 4199], pass_zero='bandpass', fs = s_rate )* 32768).astype(int)])
tap8 = np.array([twos_complement_hex(x, 16) for x in np.rint(signal.firwin(ORDER + 1, [4299, 7999], pass_zero='bandpass', fs = s_rate )* 32768).astype(int)])
np.savetxt('Tap.txt', [tap1, tap2, tap3, tap4, tap5, tap6, tap7, tap8], fmt = '%s', newline = '\n')


# Export music as txt file
hex_data = np.array([twos_complement_hex(x, 16) for x in data])
np.savetxt('Original data.txt', hex_data, fmt = '%s')
np.savetxt('Equalized data.txt',data)

# Fourier 
EQ_FFT = abs(scipy.fft.fft(equalized))
EQ_freqs = scipy.fft.fftfreq(len(EQ_FFT), (1.0/s_rate))

# Draw a comparision
plot1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
plot1.set(xlabel='Frequency (Hz)', ylabel='Amplitude') 
plot1.set_title('Original')  

plot2 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
plot2.set(xlabel='Frequency (Hz)', ylabel='Amplitude') 
plot2.set_title('Equalized') 

plot3 = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=2)  


plot1.plot(freqs[range(len(FFT)//2)], FFT[range(len(FFT)//2)])
plot2.plot(EQ_freqs[range(len(EQ_FFT)//2)], EQ_FFT[range(len(EQ_FFT)//2)])
plot3.plot(freqs[range(len(FFT)//2)], FFT[range(len(FFT)//2)], color='r', label='Original')
plot3.plot(EQ_freqs[range(len(EQ_FFT)//2)], EQ_FFT[range(len(EQ_FFT)//2)], color='g', label='Equalized')
plot3.legend()
                                                       
plt.show()

