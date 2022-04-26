#########################################################################
#																		#
# Tristan Denning														#
# ECE 351 - 51															#
# Final Project															#	
# 26APR2022    												     		#
#																		#
#########################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches    
import scipy.signal as sig
import scipy.fftpack
import pandas as pd

df = pd.read_csv('NoisySignal.csv')

t = df['0'].values
sensor_sig = df['1'].values

plt.figure(figsize = (10, 7))
plt.plot(t , sensor_sig )
plt.grid()
plt.title('Noisy Input Signal')
plt.ylabel('Amplitude [V]')
plt.xlabel('Time [s]')
plt.show()


#--------- Part 1 ------------------------------------------------------------#

def make_stem ( ax ,x ,y , color ='k', style ='solid', label ='',
               linewidths =2.5 ,** kwargs ) :
    ax.axhline(x [0] , x[-1] ,0 , color ='r')
    ax.vlines(x , 0, y, color = color , linestyles = style, label=label,
              linewidths = linewidths)
    ax.set_ylim([1.05*y.min(), 1.05*y.max()])
    
def FT(x, fs):                                      #FFT from Lab 9  
    N = len(x)
    X_fft = scipy.fftpack.fft(x)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft)
    
    freq = np.arange(-N/2, N/2)*fs/N
    
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    
    for i in range(len(X_phi)):
        if abs(X_mag[i]) < 0.01:
            X_phi[i] = 0
            
    return freq, X_mag, X_phi




fs = 1e6
freq, X_mag, X_phi = FT(sensor_sig, fs)

print('length of freq is: ', len(freq))
print('length of X_mag is: ',len(X_mag))
print('length of X_phi is: ', len(X_phi))

#---- Full Range Noise--------------------------------------------------------#
fig, ax = plt . subplots ( figsize =(10 , 7) )
make_stem( ax , freq , X_mag )
plt.title('Full Range FFT (Unfiltered)')
plt.ylabel ('Noise Magnitude|X(f)|')
plt.xlabel ('f[Hz]')
plt.grid()
plt.show()



#---- Full Range Noise Zoomed in ---------------------------------------------#
fig, ax = plt . subplots ( figsize =(10 , 7) )
make_stem( ax , freq , X_mag )
plt.title('Full Range (zoomed in) FFT (Unfiltered)')
plt.ylabel ('Noise Magnitude|X(f)|')
plt.xlabel ('f[Hz]')
plt.grid()
plt.xlim(0, 450000)
plt.show()





#---- Noise below desired Range ----------------------------------------------#
fig,ax = plt.subplots(figsize =(10 , 7) )
make_stem(ax , freq , X_mag )
plt.title('Low Frequency FFT (Unfiltered)')
plt.ylabel('Noise Magnitude |X(f)|')
plt.xlabel('f[Hz]')
plt.grid()
plt.xlim(0, 1800)
plt.show()



#---- Signal within desired Range -- Don't filter ----------------------------#

fig , ax = plt.subplots(figsize =(10 , 7) )
make_stem(ax , freq, X_mag )
plt.title('[1800-2000 Hz] FFT (Unfiltered)')
plt.ylabel('Noise Magnitude|X(f)|')
plt.xlabel('f[Hz]')
plt.grid()
plt.xlim(1800, 2000)
plt.show()


#--- NOISE ABOVE 2000 Hz------------------------------------------------------#
fig, ax = plt.subplots ( figsize =(10 , 7) )
make_stem(ax , freq , X_mag )
plt.title('High Frequency FFT (Unfiltered)')
plt.ylabel('Noise Magnitude|X(f)|')
plt.xlabel('f[Hz]')
plt.grid()
plt.xlim(2000, 100000)
plt.show()

#----- Part 2 and 3 ----------------------------------------------------------#
C = 8.81745e-8
L = 0.0795775
R = 500

num = [R/L, 0]
den = [1, R/L, 1/(L*C)]

steps = 1
f = np.arange(1, 1e6 +steps, steps)
w = f*2*np.pi

numz, denz = scipy.signal.bilinear(num, den, fs)
y = scipy.signal.lfilter(numz, denz, sensor_sig)

plt.figure ( figsize = (10 , 7) )
plt.subplot (1 , 1 , 1)
plt.plot (t , y)
plt.grid ()
plt.ylabel ('Amplitude [V]')
plt.xlabel ('t')
plt.title('Filtered Signal')

import control as con

sys = con.TransferFunction ( num , den )
plt.figure ( figsize = (10 , 7) )
_ = con.bode ( sys , w , dB = True , Hz = True , deg = True , Plot = True )

plt.figure ( figsize = (10 , 7) )
_ = con.bode ( sys , np.arange(1, 1800 +steps, steps)*2*np.pi , dB = True , Hz = True , deg = True , Plot = True )

plt.figure ( figsize = (10 , 7) )
_ = con.bode ( sys , np.arange(1800, 2000 +steps, steps)*2*np.pi , dB = True , Hz = True , deg = True , Plot = True )

plt.figure ( figsize = (10 , 7) )
_ = con.bode ( sys , np.arange(2000, 1e6 +steps, steps)*2*np.pi , dB = True , Hz = True , deg = True , Plot = True )


freqFiltered, X_magFiltered, X_phiFiltered = FT(y, fs)



fig, ax = plt.subplots(figsize=(10,7))
make_stem(ax, freqFiltered, X_magFiltered)
plt.title('Full Range FFT (Filtered)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

fig, ax = plt . subplots ( figsize =(10 , 7) )
make_stem( ax , freqFiltered , X_magFiltered )
plt.title('Full Range (zoomed in) FFT (Unfiltered)')
plt.ylabel ('Noise Magnitude|X(f)|')
plt.xlabel ('f[Hz]')
plt.grid()
plt.xlim(0, 450000)
plt.show()

fig, ax = plt.subplots(figsize=(10,7))
ax.set_xlim([0, 1780])
make_stem(ax, freqFiltered, X_magFiltered)
plt.title('Low Frequency FFT (Filtered)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()



fig, ax = plt.subplots(figsize=(10,7))
ax.set_xlim([1800, 2000])
make_stem(ax, freqFiltered, X_magFiltered)
plt.title('[1800-2000 Hz] FFT (Filtered)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize=(10,7))
ax.set_xlim([2010, 100000])
make_stem(ax, freqFiltered, X_magFiltered)
plt.title('High Frequency FFT (Filtered)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()