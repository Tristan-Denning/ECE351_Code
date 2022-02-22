#########################################################################
#																		#
# Tristan Denning														#
# ECE 351 - 51															#
# Lab 5																	#	
# 22FEB2022																#
#																		#
#########################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

steps = 1e-8                                 # RESOLUTION       
t = np.arange(0, 1.2e-3 + steps , steps )              # INTERVAL
#-------Ramp and Step Functions ----------------------------------------------#    
def u(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if (t[i] < 0):
            y[i] = 0
        else:
            y[i] = 1
    return y

#-----------Given Component Values -------------------------------------------#
R = 1000
L = 27e-3
C = 100e-9

#-----------Scipy Generated Impulse Response ---------------------------------#
num = [0, 1, 0]
den = [R*C, 1, R/L]

tout, yout = sig.impulse((num, den), T = t)

#----------- Hand-Calculated Impulse Response --------------------------------#

def h(t):
    
    y = 1.03556e4*np.exp(-5000*t)*np.sin(18584*t + 1.83363)*u(t) # IN RAD
    return y

h1 = h(t)

#------------ Plots of Hand-Calculated and Scipy Impulse ---------------------#

# plt.figure ( figsize = (11 , 8) )
# plt.subplot(2, 1, 1)
# plt.plot (t, h1)
# plt.grid()
# plt.ylabel('h(t)', fontsize = 18)
# plt.title('Hand-Calculated Impulse Response', fontsize = 18)


# plt.subplot (2 , 1 , 2)
# plt.plot(tout, yout )
# plt.grid()
# plt.ylabel ('h(t)', fontsize = 18)
# plt.title('Scipy.signal Impulse Response', fontsize = 18)
# plt.xlabel('t', fontsize = 18)

#---------- Step Response ----------------------------------------------------#

tout, yout = sig.step((num, den), T = t)
plt.figure ( figsize = (11 , 8) )
plt.plot(tout, yout )
plt.grid()
plt.ylabel ('h(t)', fontsize = 18)
plt.title('Scipy.signal Step Response', fontsize = 18)
plt.xlabel('t', fontsize = 18)