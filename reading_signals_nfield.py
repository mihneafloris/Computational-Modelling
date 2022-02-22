import numpy as np
import scipy as sp
import scipy.interpolate
import matplotlib.pyplot as plt
from scipy.signal import welch


# Read the near field signals from the file
datax=np.genfromtxt("c:\\users\\shiva\\OneDrive\\Desktop\\Project\\signals_nfield.txt",comments="#")

t1=datax[:,0]               # the first column is time


# Read the far field signals from the file
dataf=np.genfromtxt("c:\\users\\shiva\\OneDrive\\Desktop\\Project\\farfield_microphones\\setpoint_07\\fwh.mic_"+str(20)+"_long.txt",comments="#")

xx= [i[0] for i in dataf]  # time
xx=xx[37504:59488]         # make the the time interval identical to the near field one

sl=1/(xx[18]-xx[17])

yy= [i[1] for i in dataf]  # pressure
yy=yy[37504:59488]         # make the the pressure interval identical to the near field one



for n in range(10,11):      # loop from the first to the last near field probe

    # data
    y = datax[:,n]             # read the pressure of the near field of each probe ( the columns in the text file)
    dy= [i-101325 for i in y]  # subtract the reference pressure



#---Interpolate the data using a linear spline to "new_length" samples for the near field

    new_length = 21984                                                # number of data points
    new_x = np.linspace(t1.min(), t1.max(), new_length)               # calculate new x values
    new_y = sp.interpolate.interp1d(t1, dy, kind='linear')(new_x)     # calculate new y values


    f,cxy=sp.signal.coherence( new_y,yy,fs=sl,nperseg=4098)                         # calculate the coherence squared and frequency

plt.plot(t1, dy)
plt.plot(xx,yy)
plt.plot(new_x,new_y,color='y')

print(max(new_y),max(dy))
plt.show()




#--------------------------- NOT IMPORTANT (plot results) ---------------------------------------

    #plt.cohere([yyy-101325 for yyy in new_y],yy)
    #plt.show()

print(max(cxy))
plt.plot(f,cxy)
plt.xlabel('f[HZ]', fontsize=16)
plt.ylabel('cxy', fontsize=16)
plt.show()

'''

    # Plot the results
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(x, y, 'bo-')
    plt.title('Using 1D Linear Spline Interpolation')

    plt.subplot(2,1,2)
    plt.plot(new_x, new_y, 'ro-')

    plt.show()
'''

