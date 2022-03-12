import matplotlib.pyplot as pl
import numpy as np

r1_log = np.loadtxt("./experiment3/logRP1_automatico.txt", delimiter=',', skiprows = 2)
r2_log = np.loadtxt("./experiment3/logRP2_automatico.txt", delimiter=',', skiprows = 2)

tf = int(r1_log[-1][0] / 1000)
dt = 0.02
dt_inv = 1.0/dt # Sampling frequency in sec^-1

log_time = np.linspace(0, tf, tf*dt_inv)
log_vel = np.zeros((np.size(log_time) , 3))

# Conversion matrix for robot's velocities

R = 3.35 # Wheel's radius cm
L = 12.4 # Nu sep que es cm
Mc = np.array([[R/L, -R/L],[R/2.0, R/2.0]])

for i in range(np.size(log_time)):
    # Robots' velocities
    r1_wR = 0.0*r1_log[i][1]
    r1_wL = 0.0*r1_log[i][2]
    r2_wR = r2_log[i][1]
    r2_wL = r2_log[i][2]
    r3_wR = 0.0
    r3_wL = 0.0

    r1_vel = Mc.dot(np.array([r1_wR, r1_wL]))
    r2_vel = Mc.dot(np.array([r2_wR, r2_wL]))
    r3_vel = Mc.dot(np.array([r3_wR, r3_wL]))

    time = r2_log[i][0]

    log_vel[i,1:3] = r2_vel
    log_vel[i,0] = time

np.savetxt("./experiment3/logRP2_automatico_vel_proc.txt", log_vel, delimiter=',')
