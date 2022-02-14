#########################################################################
#																		#
# Tristan Denning														#
# ECE 351 - 51															#
# Lab 4																	#	
# 15FEB2022																#
#																		#
#########################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

steps = 1e-2                                     # RESOLUTION       
t = np.arange(-10, 10 + steps, steps)              # INTERVAL

#-------Ramp and Step Functions ---------#

def r(t):   
    y = np.zeros(t.shape)
    
    for i in range(len(t)): #Runs the loop for each increment of t
        if (t[i] > 0):      
            y[i] = t[i]
        else:
            y[i] = 0
    
    return y
    
def u(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if (t[i] > 0):
            y[i] = 1
        else:
            y[i] = 0
    return y

#----------------- User-Defined Functions ------------------------------------#
def h1(t):
    y = np.zeros(t.shape)
    y = np.exp(-2*t)*(u(t) - u(t-3))
    return y

def h2(t):
    y = np.zeros(t.shape)
    y = u(t-2) - u(t-6)
    return y

def h3(t):
    w = 0.25*2*np.pi
    y = np.zeros(t.shape)
    y = np.cos(w*t)*u(t)
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


#------- Call the Functions --------------------------------------------------#
h1 = h1(t)
h2 = h2(t)
h3 = h3(t)
u = u(t)

#-------- Plot User-Defined Functions ----------------------------------------#
# plt.figure ( figsize = (16 , 14) )

# plt.subplot(3, 1, 1)
# plt.plot(t, h1)
# plt.grid()
# plt.ylabel('h1(t)', fontsize = 18)
# plt.title('e$^{-2t}$*(u(t) - u(t-3)', fontsize = 18)

# plt.subplot (3 , 1 , 2)
# plt.plot(t , h2 )
# plt.grid()
# plt.ylabel ('h2(t)', fontsize = 18)
# plt.title('u(t-2) - u(t-6)', fontsize = 18)

# plt.subplot(3, 1, 3)
# plt.plot(t, h3) 
# plt.grid()
# plt.ylabel('h3(t)', fontsize = 18)
# plt.title('h3(t) at 0.25 Hz', fontsize = 18)
# plt.xlabel('t', fontsize = 18)

#--------- Step Response Code ------------------------------------------------#
# The step response is the convolution of the given function and the unit step 
# function.

# UR1 = scipy.signal.convolve(u, h1)
# UR2 = scipy.signal.convolve(u, h2)
# UR3 = scipy.signal.convolve(u, h3)

# UR1 = conv(u, h1)*steps
# UR2 = conv(u, h2)*steps
# UR3 = conv(u, h3)*steps



#-------- Python Generated Step Responses ------------------------------------#
# t = np.arange(-10, 10 + steps, steps)
# NN = len(t)
# tExtended = np.arange(2 * t[0] , 2 * t[NN-1] + steps, steps)

# plt.figure ( figsize = (16 , 14) )


# plt.subplot(3, 1, 1)
# plt.plot(tExtended, UR1)
# plt.grid()
# plt.ylabel('h1(t)*u(t)', fontsize = 18)
# plt.title('Step Response of h1(t)', fontsize = 18)

# plt.subplot (3 , 1 , 2)
# plt.plot(tExtended, UR2)
# plt.grid()
# plt.ylabel ('h2(t)*u(t)', fontsize = 18)
# plt.title('Step Response of h2(t)', fontsize = 18)

# plt.subplot(3, 1, 3)
# plt.plot(tExtended, UR3) 
# plt.grid()
# plt.ylabel('h3(t)*u(t)', fontsize = 18)
# plt.title('Step Response of h3(t)', fontsize = 18)
# plt.xlabel('t', fontsize = 18)

#-------- Hand Calculated Step Responses -------------------------------------#
t = np.arange(-20, 20 + steps, steps) 

def u(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if (t[i] > 0):
            y[i] = 1
        else:
            y[i] = 0
    return y

def c1(t):
    
    y = 0.5*(((-np.exp(-2*t)+1)*u(t)-((-np.exp(-2*(t-3)))+1)*u(t-3)))
    
    return y


def c2(t):
    
    y = t*u(t-2)- 2*u(t-2) - (t*u(t-6)- 6*u(t-6))
    
    return y

def c3(t):
    w = 0.25*2*np.pi
    
    
    y = (1/w)*np.sin(w*t)*u(t)
    return y

c1 = c1(t)
c2 = c2(t)
c3 = c3(t)

plt.figure ( figsize = (16 , 14) )
plt.subplot(3, 1, 1)
plt.plot(t, c1)
plt.grid()
plt.ylabel('h1(t)*u(t)', fontsize = 18)
plt.title('Step Response of h1(t) (Hand-Calculated)', fontsize = 18)

plt.subplot (3 , 1 , 2)
plt.plot(t, c2)
plt.grid()
plt.ylabel ('h2(t)*u(t)', fontsize = 18)
plt.title('Step Response of h2(t) (Hand-Calculated)', fontsize = 18)

plt.subplot(3, 1, 3)
plt.plot(t, c3) 
plt.grid()
plt.ylabel('h3(t)*u(t)', fontsize = 18)
plt.title('Step Response of h3(t) (Hand-Calculated)', fontsize = 18)
plt.xlabel('t', fontsize = 18)