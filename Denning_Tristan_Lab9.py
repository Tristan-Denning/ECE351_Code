#########################################################################
#																		#
# Tristan Denning														#
# ECE 351 - 51															#
# Lab 9																	#	
# 22MAR2022													 		    #
#																		#
#########################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal 
import scipy.fftpack


#---------- Time Array Declaration -------------------------------------------#
steps = 1e-2                                      # RESOLUTION       
t = np.arange(0, 2 , steps )              # INTERVAL


#---------- Function Definitions ---------------------------------------------#

def FT(x, fs):
    N = len(x)
    X_fft = scipy.fftpack.fft(x)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft)
    
    freq = np.arange(-N/2, N/2)*fs/N
    
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    return freq, X_mag, X_phi

#--------- Plots -------------------------------------------------------------#
#--------- Task 1 ------------------------------------------------------------#
x = np.cos(2*np.pi*t)
fs = 100
freq, X_mag, X_phi = FT(x, fs)

plt.figure ( figsize = (10 , 8) )
plt.subplot (3 , 1 , 1)
plt.plot (t , x)
plt.grid ()
plt.ylabel ('k = 1')
plt.xlabel ('t')
plt.title('Task 1: cos(2*\u03C0*t)', fontsize = 13)

plt.subplot (3 , 2 , 3)
plt.stem(freq, X_mag)
plt.grid ()
plt.ylabel ('|H(f)|')

plt.subplot (3 , 2 , 5)
plt.stem (freq, X_phi)
plt.grid ()
plt.ylabel (r'$\angle$X(f)')
plt.xlabel ('f[Hz]')

plt.subplot (3 , 2 , 4)
plt.stem(freq, X_mag)
plt.grid ()
plt.xlim(-2, 2)

plt.subplot (3 , 2 , 6)
plt.stem (freq, X_phi)
plt.grid ()
plt.xlim(-2, 2)
plt.xlabel ('f[Hz]')
plt.show ()

#--------- Task 2 ------------------------------------------------------------#
x = 5*np.sin(2*np.pi*t)

freq, X_mag, X_phi = FT(x, fs)

plt.figure ( figsize = (10 , 8) )
plt.subplot (3 , 1 , 1)
plt.plot (t , x)
plt.grid ()
plt.ylabel ('k = 1')
plt.xlabel ('t')
plt.title('Task 2: 5sin(2*\u03C0*t)', fontsize = 13)

plt.subplot (3 , 2 , 3)
plt.stem(freq, X_mag)
plt.grid ()
plt.ylabel ('|H(f)|')

plt.subplot (3 , 2 , 5)
plt.stem (freq, X_phi)
plt.grid ()
plt.ylabel (r'$\angle$X(f)')
plt.xlabel ('f[Hz]')

plt.subplot (3 , 2 , 4)
plt.stem(freq, X_mag)
plt.grid ()
plt.xlim(-2, 2)

plt.subplot (3 , 2 , 6)
plt.stem (freq, X_phi)
plt.grid ()
plt.xlim(-2, 2)
plt.xlabel ('f[Hz]')
plt.show ()

#--------- Task 3 ------------------------------------------------------------#
x = 2*np.cos((4*np.pi*t)-2) + (np.sin((12*np.pi*t)+3))**2

freq, X_mag, X_phi = FT(x, fs)

plt.figure ( figsize = (10 , 8) )
plt.subplot (3 , 1 , 1)
plt.plot (t , x)
plt.grid ()
plt.ylabel ('k = 1')
plt.xlabel ('t')
plt.title("Task 3: 2cos(2*\u03C0*2t-2) + $\mathregular{sin^{2}}$(2*\u03C0*6t)", 
          fontsize = 13)

plt.subplot (3 , 2 , 3)
plt.stem(freq, X_mag)
plt.grid ()
plt.ylabel ('|H(f)|')

plt.subplot (3 , 2 , 5)
plt.stem (freq, X_phi)
plt.grid ()
plt.ylabel (r'$\angle$X(f)')
plt.xlabel ('f[Hz]')

plt.subplot (3 , 2 , 4)
plt.stem(freq, X_mag)
plt.grid ()
plt.xlim(-2, 2)

plt.subplot (3 , 2 , 6)
plt.stem (freq, X_phi)
plt.grid ()
plt.xlim(-2, 2)
plt.xlabel ('f[Hz]')
plt.show ()

#---------- Task 4 -----------------------------------------------------------#

def FT2(x, fs):
    N = len(x)
    X_fft = scipy.fftpack.fft(x)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft)
    freq = np.arange(-N/2, N/2)*fs/N
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    
    for i in range(len(X_phi)):
        if abs(X_mag[i]) < 1e-10:
            X_phi[i] = 0
    return freq, X_mag, X_phi

x1 = np.cos(2*np.pi*t)
freq1, X_mag1, X_phi1 = FT2(x1, fs)

plt.figure ( figsize = (10 , 7) )
plt.subplot (3 , 1 , 1)
plt.plot (t , x1)
plt.grid ()
plt.ylabel ('k = 1')
plt.xlabel ('t')
plt.title('Task 4.1')

plt.subplot (3 , 2 , 3)
plt.stem(freq1, X_mag1)
plt.grid ()
plt.ylabel ('|H(f)|')

plt.subplot (3 , 2 , 5)
plt.stem (freq1, X_phi1)
plt.grid ()
plt.ylabel (r'$\angle$X(f)')
plt.xlabel ('f[Hz]')

plt.subplot (3 , 2 , 4)
plt.stem(freq1, X_mag1)
plt.grid ()
plt.xlim(-2, 2)

plt.subplot (3 , 2 , 6)
plt.stem (freq1, X_phi1)
plt.grid ()
plt.xlim(-2, 2)
plt.xlabel ('f[Hz]')
plt.show ()

x2 = 5*np.sin(2*np.pi*t)
freq2, X_mag2, X_phi2 = FT2(x2, fs)

plt.figure ( figsize = (10 , 7) )
plt.subplot (3 , 1 , 1)
plt.plot (t , x2)
plt.grid ()
plt.ylabel ('k = 1')
plt.xlabel ('t')
plt.title('Task 4.2')

plt.subplot (3 , 2 , 3)
plt.stem(freq2, X_mag2)
plt.grid ()
plt.ylabel ('|H(f)|')

plt.subplot (3 , 2 , 5)
plt.stem (freq2, X_phi2)
plt.grid ()
plt.ylabel (r'$\angle$X(f)')
plt.xlabel ('f[Hz]')

plt.subplot (3 , 2 , 4)
plt.stem(freq2, X_mag2)
plt.grid ()
plt.xlim(-2, 2)

plt.subplot (3 , 2 , 6)
plt.stem (freq2, X_phi2)
plt.grid ()
plt.xlim(-2, 2)
plt.xlabel ('f[Hz]')
plt.show ()


x3 = 2*np.cos((4*np.pi*t)-2) + (np.sin((12*np.pi*t)+3))**2
freq3, X_mag3, X_phi3 = FT2(x3, fs)

plt.figure ( figsize = (10 , 7) )
plt.subplot (3 , 1 , 1)
plt.plot (t , x3)
plt.grid ()
plt.ylabel ('k = 1')
plt.xlabel ('t')
plt.title('Task 4.3')

plt.subplot (3 , 2 , 3)
plt.stem(freq3, X_mag3)
plt.grid ()
plt.ylabel ('|H(f)|')

plt.subplot (3 , 2 , 5)
plt.stem (freq3, X_phi3)
plt.grid ()
plt.ylabel (r'$\angle$X(f)')
plt.xlabel ('f[Hz]')

plt.subplot (3 , 2 , 4)
plt.stem(freq3, X_mag3)
plt.grid ()
plt.xlim(-15, 15)

plt.subplot (3 , 2 , 6)
plt.stem (freq3, X_phi3)
plt.grid ()
plt.xlim(-15, 15)
plt.xlabel ('f[Hz]')
plt.show ()


#---------- Task 5 -----------------------------------------------------------#

T = 8
t = np.arange(0, 16, steps)
w0 = 2*np.pi/T

def bk(k):
    y = (2/(k*np.pi))*(1-(-1)**(k))
    return y

def x(N, T):
    w = 2*np.pi/T
    
    result = 0
    for i in np.arange(1, N+1):
        intermediate = bk(i)*np.sin(i*w*t)
        result += intermediate
    return result
   

x3 = x(15, T)
freq3, X_mag3, X_phi3 = FT2(x3, fs)

plt.figure ( figsize = (10 , 7) )
plt.subplot (3 , 1 , 1)
plt.plot (t , x3)
plt.grid ()
plt.ylabel ('k = 1')
plt.xlabel ('t')
plt.title('Task 5')

plt.subplot (3 , 2 , 3)
plt.stem(freq3, X_mag3)
plt.grid ()
plt.ylabel ('|H(f)|')

plt.subplot (3 , 2 , 5)
plt.stem (freq3, X_phi3)
plt.grid ()
plt.ylabel (r'$\angle$X(f)')
plt.xlabel ('f[Hz]')

plt.subplot (3 , 2 , 4)
plt.stem(freq3, X_mag3)
plt.grid ()
plt.xlim(-3, 3)

plt.subplot (3 , 2 , 6)
plt.stem (freq3, X_phi3)
plt.grid ()
plt.xlim(-3, 3)
plt.xlabel ('f[Hz]')
plt.show ()