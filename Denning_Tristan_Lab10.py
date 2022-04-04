#########################################################################
#																		#
# Tristan Denning														#
# ECE 351 - 51															#
# Lab 10																#	
# 29MAR2022	    														#
#																		#
#########################################################################
import numpy as np
import control as con
import matplotlib.pyplot as plt
import scipy.signal as sig

R = 1000
L = 27e-3
C = 100e-9

steps = 1
w = np.arange(1e3, 1e6 +steps, steps)

#-------- Prelab-Derived Equations -------------------------------------------#

def mag(R, C, L, w):
    y = (w/(R*C))/np.sqrt(w**4+(1/(L*C))**2+(1/(R*C)**2-2/(L*C))*w**2)
    y = 20*np.log(y)
    return y

def ang(R, C, L, w):
    
    x = np.pi/2 - np.arctan((w/(R*C))/((1/(L*C))-w**2))
    x = np.degrees(x)
    for i in range(len(w)):  
        if x[i] > 90:
            x[i] = x[i]-180
    return x

H = mag(R, C, L, w)
Ha = ang(R, C, L, w)

num = [0, 1/(R*C), 0]
den = [1, 1/(R*C), 1/(L*C)]

#-------- Part 1 Task 1 Plots ------------------------------------------------#

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.semilogx(w, H)
plt.grid()
plt.ylabel(r'|H($\omega$)|')
plt.title('Task 1 Plot')

plt.subplot(2, 1, 2)
plt.semilogx(w, Ha)
plt.grid()
plt.ylabel(r'$\angle$H(w)')
plt.xlabel(r'$\omega$ [rad/s]', fontsize = 12)


#-------- Part 1 Task 2 ------------------------------------------------------#

sys = sig.TransferFunction(num, den)

w, mag, phase = sig.bode(sys, w)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.semilogx(w, mag)
plt.grid()
plt.ylabel(r'|H($\omega$)|')
plt.title('Task 2 Plot')

plt.subplot(2, 1, 2)
plt.semilogx(w, phase)
plt.grid()
plt.ylabel(r'$\angle$H(w)')
plt.xlabel(r'$\omega$ [rad/s]', fontsize = 12)

#------- Part 1 Task 3 -------------------------------------------------------#

sys = con.TransferFunction(num, den)
plt.figure(figsize = (10,7))
_ = con.bode(sys , w , dB = True , Hz = True , deg = True , Plot = True,)

#-------- Part 2 Task 1 ------------------------------------------------------#

fs = 1000000
steps = 1/fs
t = np.arange(0, .01 + steps, steps)

x = np.cos(2*np.pi*100*t) + np.cos(2*np.pi*3024*t) + np.sin(2*np.pi*50000*t)

plt.figure(figsize = (10, 7))
plt.subplot(1 , 1 , 1)
plt.plot(t, x)
plt.grid()
plt.ylabel('x(t)', fontsize = 12)
plt.xlabel('t', fontsize = 12)
plt.title('Part 2 Task 1')

#-------- Part 2 Task 2 ------------------------------------------------------#

Znum, Dnum = sig.bilinear(num, den, fs )
filtered = sig.lfilter(Znum, Dnum, x)

plt.figure(figsize = (10, 7))
plt.subplot(1 , 1 , 1)
plt.plot(t, filtered)
plt.grid()
plt.ylabel('x(t)', fontsize = 12)
plt.xlabel('t', fontsize = 12)
plt.title('Part 2 Task 4')