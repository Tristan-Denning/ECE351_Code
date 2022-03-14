#########################################################################
#																		#
# Tristan Denning														#
# ECE 351 - 51															#
# Lab 8																	#	
# 15MAR2022															#
#																		#
#########################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#---------- Time Array Declaration -------------------------------------------#
steps = 1e-2                                      # RESOLUTION       
t = np.arange(0, 20 + steps , steps )              # INTERVAL



#---------- Function Definitions ---------------------------------------------#
def ak(k):
    y = 0
    
    return y

def bk(k):
    y = (2/(k*np.pi))*(1-(-1)**(k))
    # y = 2/((k)*np.pi)*(1-np.cos((k)*np.pi))
    return y

def fourier(N, T):
    w = 2*np.pi/T
    
    result = 0
    for i in np.arange(1, N+1):
        intermediate = bk(i)*np.sin(i*w*t)
        result += intermediate
    return result

#--------- Task 1 Printout ---------------------------------------------------#
print(u'a\u2080 = ', ak(0))
print(u'a\u2081 = ', ak(1))

print( u'\nb\u2081 = ', bk(1))
print(u'b\u2082 = ', bk(2))
print(u'b\u2083 = ', bk(3))

#--------- Task 2 Plots n = 1, 3, 15 -----------------------------------------#
plt.figure ( figsize = (10 , 8 ) )
plt.subplot (3 , 1 , 1)
plt.plot(t, fourier(1, 8))
plt.grid()
plt.ylabel('N=1', fontsize = 12)
plt.title('Fourier Series Approximation', fontsize = 12)

plt.subplot (3 , 1 , 2)
plt.plot(t, fourier(3, 8))
plt.grid()
plt.ylabel('N = 3', fontsize = 12)

plt.subplot (3 , 1 , 3)
plt.plot(t, fourier(15, 8))
plt.grid()
plt.ylabel('N = 15', fontsize = 12)
plt.xlabel('t', fontsize  = 12)

#--------- Task 2 Plots n = 1, 3, 15 -----------------------------------------#

plt.figure ( figsize = (10 , 8 ) )
plt.subplot (3 , 1 , 1)
plt.plot(t, fourier(50, 8))
plt.grid()
plt.ylabel('N = 50', fontsize = 12)
plt.title('Fourier Series Approximation', fontsize = 12)

plt.subplot (3 , 1 , 2)
plt.plot(t, fourier(150, 8))
plt.grid()
plt.ylabel('N = 150', fontsize = 12)

plt.subplot (3 , 1 , 3)
plt.plot(t, fourier(1500, 8))
plt.grid()
plt.ylabel('N = 1500', fontsize = 12)
plt.xlabel('t', fontsize  = 12)
