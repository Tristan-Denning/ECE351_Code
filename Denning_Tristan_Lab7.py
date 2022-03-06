#########################################################################
#																		#
# Tristan Denning														#
# ECE 351 - 51															#
# Lab 6																	#	
# 29FEB2022																#
#																		#
#########################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#---------- Time Array Declaration -------------------------------------------#
steps = 1e-2                                      # RESOLUTION       
t = np.arange(0, 2 + steps , steps )              # INTERVAL

#----------Function Definitions ----------------------------------------------#
def u(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if (t[i] < 0):
            y[i] = 0
        else:
            y[i] = 1
    return y



#---------- Part 1 Task 2 Array Definitions  ---------------------------------#

numG = [1, 9]
denG = [1, -2, -40, -64]

numA = [1, 4]
denA = [1, 4, 3]

numB = [1, 26, 168]
denB = [1]
    
zG, pG, kG = sig.tf2zpk(numG, denG)
print('\n')
print('Zeroes for G(s) = ', zG)
print('Poles for G(s) = ', pG)


zA, pA, kA = sig.tf2zpk(numA, denA)
print('\n')
print('Zeroes for A(s) = ', zA)
print('Poles for A(s) = ', pA)


zB, pB, kB = sig.tf2zpk(numB, denB)
print('\n')
print('Zeroes for B(s) = ', zB)
print('Poles for B(s) = ', pB)



#---------- Part 1 Task 5 ----------------------------------------------------#

numH = sig.convolve([1, 4], [1,9])
denH = sig.convolve([1, 4, 3],[2, 41, 500, 2995, 6878, 4344])
# print (numH)
# print (denH)

numHopen = sig.convolve(numA, numG)
denHopen = sig.convolve(denA, denG)

to, yo = sig.step((numHopen, denHopen), T=t)


plt.figure ( figsize = (9 , 6) )
plt.subplot (2 , 1 , 1)
plt.plot (to, yo)
plt.grid ()
plt.ylabel ('y(t)' , fontsize  = 12)
plt.title ('Open-Loop Step Response')

#---------- Part 2 Task 2 ----------------------------------------------------#
t = np.arange(0, 10+ steps , steps )  

numF = sig.convolve(numA, numG)
num2 = sig.convolve(numB, numG)
num1 = sig.convolve(denA, denG)
num2 = sig.convolve(num2, denA)
print('\n')

print(numF)
print(num1)
print(num2)
denF = num1 + num2
print(denF)

z2, p2, k2 = sig.tf2zpk(numF, denF)

print('\n')

print('Zeroes for Closed Loop H(s) = ', z2)
print('Poles for Closed Loop H(s)  = ', p2)


#----------- Part2 Task 4 ----------------------------------------------------#
 
to, yo = sig.step((numF, denF), T=t)

plt.figure ( figsize = (10 , 7) )
plt.plot (to , yo)
plt.grid ()
plt.ylabel ('y(t)' , fontsize  = 12)
plt.xlabel ('t')
plt.title ('Closed-Loop Step Response', fontsize = 12)
