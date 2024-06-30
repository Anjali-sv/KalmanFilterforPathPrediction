import numpy as np
import matplotlib.pyplot as plt
from math import *

# Read data from the file
with open("kalmann.txt" ,'r') as file:

    lines = file.readlines()
# State variables
line1=(lines[0].split(','))
state = np.array([[line1[0]],
                   [line1[1]],
                  [0],
                   [0]],dtype=float)  # [x, y, vx, vy]
'''print(state)'''
#find variance of x and y
xmean,ymean,n,xvar,yvar=0,0,0,0,0
vx_mean,vy_mean,vx_var,vy_var=0,0,0,0
for line in lines: #to find mean of x,y,vx,vy
    line=line.split(',')
    xmean+=eval(line[0])
    ymean+=eval(line[1])
    n+=1
    if n!=1:
      vx_mean+=eval(line[2])
      vy_mean += eval(line[3])

m=0
for line in lines: #to find variance of x,y,vx,vy
    line = line.split(',')
    xvar+=(eval(line[0])-xmean)**2
    yvar+=(eval(line[1])-ymean)**2
    m+=1
    if m!=1:
      vx_var += (eval(line[2]) - vx_mean) ** 2
      vy_var += (eval(line[3]) - vy_mean) ** 2

xvar=xvar/n
yvar=yvar/n
vx_var=vx_var/(n-1)
vy_var=vy_var/(n-1)

dt=3.65

# State transition matrix
A = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# Control matrix
'''B = np.array([[0.5*dt, 0],
              [0, 0.5*dt],
              [1*dt, 0],
              [0, 1*dt]])'''

# Measurement matrix
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

# Measurement noise covariance
R = np.array([[xvar, 0],
              [0, yvar]])
# Process noise covariance
Q = np.array([[xvar, 0,sqrt(xvar*vx_var), 0],
              [0, yvar,0,sqrt(yvar*vy_var) ],
              [sqrt(xvar*vx_var), 0, vx_var, 0],
              [0,sqrt(yvar*vy_var) , 0, vy_var]])


# Error covariance matrix
P = np.eye(4)

def kalman_filter(state, P, z):
    # Predict
    state = np.dot(A, state)
    P = np.dot(np.dot(A, P), A.T) + Q

    # Update
    del_z = z - np.dot(H, state)
    S = np.dot(np.dot(H, P), H.T) + R
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
    state = state + np.dot(K, del_z)
    P = np.dot((np.eye(4) - np.dot(K, H)), P)
    print('updated=\n',state,"\nerror covariance=\n",P,'\n****************************************************************************************************')
    return state, P
k=0
for line in lines[:]:
    line=line.split(',')
    if k==0:
        state=np.array([[float(line[0])],
               [float(line[1])],
               [0],
               [0]])
        k+=1
        estimated_coords =np.array([[state[0,0], state[1,0]]])
        gps_coords=np.array([[float(line[0]),float(line[1])]])
    else:
        state=np.array([[float(line[0])],
               [float(line[1])],
               [float(line[2])],
               [float(line[3])]])
    # Set the measurement vector
    z = np.array([[float(line[0])], [float(line[1])]])

    gps_coords = np.append(gps_coords, [[float(line[0]),float(line[1])]], axis=0)
    # Apply the Kalman filter
    state,P = kalman_filter(state, P, z)
    estimated_coords=np.append(estimated_coords,[[state[0,0],state[1,0]]],axis=0)

plt.plot(estimated_coords[:,0], estimated_coords[:,1], 'r-', label='Estimated Measurements',linewidth=0.5)
plt.plot(gps_coords[:,0], gps_coords[:,1], 'y-', label='GPS Measurements',linewidth=0.5)
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.show()