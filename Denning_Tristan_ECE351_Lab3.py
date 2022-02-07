#########################################################################
#																		#
# Tristan Denning														#
# ECE 351 - 51															#
# Lab 3																	#	
# 15FEB2022																#
#																		#
#########################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

steps = 1e-2                                     # RESOLUTION       
t = np.arange(0, 20 + steps, steps)              # INTERVAL
l = np.arange(0, 20 + steps, steps)              # Dummy Interval

#-------Ramp and Step Functions ---------#

def r(t):   
    y = np.zeros(t.shape)
    
    for i in range(len(t)): #Runs the loop for each increment of t
        if (t[i] > 0):      
            y[i] = t[i]
        else:
            y[i] = 0
        #y[i] = (t[i] if t[i] >=0 else 0)#y = cos(t) for t on [0,10] # This 
        #line is an alternative definition
    
    return y
    
def u(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if (t[i] > 0):
            y[i] = 1
        else:
            y[i] = 0
    return y

#--------- User Defined Functions ----------#

def f1(t):
    y = np.zeros(t.shape)
    
    y = u(t-2) - u(t-9)
    
    return y


def f2(t):
    y = np.zeros(t.shape)
    
    y = np.exp(-t)*u(t)
    
    return y

def f3(t):
    y = np.zeros(t.shape)
    
    y = r(t - 2)*(u(t - 2) - u(t - 3)) + r(4 - t)*(u(t - 3) - u(t - 4))
    
    return y


def conv(f1, f2):
    Nf1 = len(f1)
    Nf2 = len(f2) 
    f1Extended = np.append(f1, np.zeros((1, Nf2-1)))
    f2Extended = np.append(f2, np.zeros((1, Nf1-1)))
    result = np.zeros(f1Extended.shape)
    
    for i in range(Nf2 + Nf1 - 2):
        result[i] = 0
        for j in range (Nf1):
            if (i - j + 1 > 0):
                try: 
                    result[i] += f1Extended[j]*f2Extended[i-j+1]
                except:
                    print(i,j)
    return result


#---------- Calling the functions ------------#

f1 = f1(t)

f2 = f2(t)

f3 = f3(t)

#------------- Plot Convolution for f_, f_------------------------#

conv = conv(f1, f3)

convolution = scipy.signal.convolve(f1, f3)

plt.figure ( figsize = (16 , 14) )

plt.subplot(4, 1, 1)
plt.plot(t, f1)
plt.grid()
plt.ylabel('f1(t)')
plt.title('f1(t)')

plt.subplot (4 , 1 , 2)
plt.plot(t , f3 )
plt.grid()
plt.ylabel ('f3(t)')
plt.title('f3(t)')

plt.subplot(4, 1, 3)
plt.plot(t, convolution[range(len(t))]) 
plt.grid()
plt.ylabel('f1*f3')
plt.title('scipy.signal.convolve')

plt.subplot (4 , 1 , 4)
plt.plot(t, conv[range(len(t))] )
plt.grid()
plt.ylabel ('f1*f3')
plt.title('My convolve function')
plt.xlabel ('t')

