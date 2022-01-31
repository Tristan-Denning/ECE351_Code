#########################################################################
#																		#
# Tristan Denning														#
# ECE 351 - 51															#
# Lab 1																	#	
# 01FEB2022																#
#																		#
#########################################################################

import numpy as np
import matplotlib.pyplot as plt

steps = 1e-3                                            
t = np.arange(-2, 14 + steps, steps)

print('number of elements len((t) = ', len(t), '\n First Element: t[0] =', t[0],
    '\nLast Element: t[len(t) - 1]', t[len(t) - 1])
# --------- Function Definitions------------#
def CosFunc(t):   
    y = np.zeros(t.shape)
    for i in range(len(t)): #Runs the loop for each increment of t
        y[i] = np.cos(t[i]) #y = cos(t) for t on [0,10]
    return y

def ramp(t):   
    y = np.zeros(t.shape)
    
    for i in range(len(t)): #Runs the loop for each increment of t
        if (t[i] > 0):      
            y[i] = t[i]
        else:
            y[i] = 0
        #y[i] = (t[i] if t[i] >=0 else 0)#y = cos(t) for t on [0,10] # This 
        #line is an alternative definition
    
    return y
    
def unitstep(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if (t[i] > 0):
            y[i] = 1
        else:
            y[i] = 0
    return y

def y(t):
    u = np.zeros(t.shape)
    
    u = ramp(t) - ramp(t-3) + 5*unitstep(t-3) - 2*unitstep(t-6) - 2*ramp(t-6)
    return u


#-----------Calling functions to be used in plots -----------#

t = np.arange(-5, 10 + steps, steps)    #Sets the interval. May need to adjust 
#                                        for certain shifting and scaling.
r = ramp(t)
u = unitstep(t)
y = y(t)            # Time shifting and Scaling operations can directly be 
#                     implemented in line 61. i.e: t-2 t/2 and 2*t.
#                     The corresponding graph will update accordingly.

#-----------Derivative of y(t) -----------#
dt = np.diff(t)
dy = np.diff(y)/dt


#-----------Plots -----------#


plt.figure ( figsize = (15 , 10) )
plt.subplot (3 , 1 , 3)
plt.ylim((-2,10))
#plt.plot(t, y)
plt.plot(t[range(len(dy))] , dy )
plt.grid ()
plt.ylabel ('y(t),dy/dt')
plt.xlabel ('t')
plt.title ('dy(t)/dt')
#plt.show () 

plt.subplot(3, 1, 2)
plt.plot(t, y) 
plt.grid()
plt.ylabel('y(t)')
plt.title('User Defined Function')

plt.subplot(3, 1, 1)
plt.plot(t, u) 
plt.plot(t, r)
plt.grid()
plt.ylabel('r(t) and u(t)')
plt.title('Unit Step Function and Ramp Function')

