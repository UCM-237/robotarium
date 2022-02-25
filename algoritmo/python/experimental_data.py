import matplotlib.pyplot as pl
import numpy as np
from numpy import linalg as la
from scipy.linalg import block_diag

#r1_log = np.loadtxt("./experiment/1_logoRobot1.txt", delimiter=',', skiprows = 2)
#r2_log = np.loadtxt("./experiment/1_logoRobot2.txt", delimiter=',', skiprows = 2)

#r1_log = np.loadtxt("./experiment2/logRP1_manual.txt", delimiter=',', skiprows = 2)
#r2_log = np.loadtxt("./experiment2/logRP2_manual.txt", delimiter=',', skiprows = 2)

r1_log = np.loadtxt("./experiment3/logRP1_automatico.txt", delimiter=',', skiprows = 2)
r2_log = np.loadtxt("./experiment3/logRP2_automatico.txt", delimiter=',', skiprows = 2)


def Rot(theta):
    # Rotation matrix
    return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

def compute_alphas_from_p(p):
    # Eq. 1
    alpha = np.zeros(3) # a312 a123 a231

    p1 = p[0:2]
    p2 = p[2:4]
    p3 = p[4:6]
    l12 = la.norm(p2-p1)
    l23 = la.norm(p3-p2)
    l31 = la.norm(p1-p3)

    b12 = (p2 - p1) / l12
    b21 = -b12
    b23 = (p3 - p2) / l23
    b32 = -b23
    b31 = (p1 - p3) / l31
    b13 = -b31

    Rot90 = Rot(np.pi/2)
    b12p = Rot90.dot(b12)
    b21p = Rot90.dot(b21)
    b23p = Rot90.dot(b23)
    b32p = Rot90.dot(b32)
    b31p = Rot90.dot(b31)
    b13p = Rot90.dot(b13)

    if (b12.dot(b13p) > 0):
        alpha[0] = np.arccos(b12.dot(b13))
    else:
        alpha[0] = 2*np.pi - np.arccos(b12.dot(b13))

    if (b23.dot(b21p) > 0):
        alpha[1] = np.arccos(b23.dot(b21))
    else:
        alpha[1] = 2*np.pi - np.arccos(b23.dot(b21))

    if (b31.dot(b32p) > 0):
        alpha[2] = np.arccos(b31.dot(b32))
    else:
        alpha[2] = 2*np.pi - np.arccos(b31.dot(b32))

    return alpha

def compute_alpha_kij_from_arucos(pij, pik):
    lij = la.norm(pij[0:2])
    lji = lij
    lik = la.norm(pik[0:2])
    lkk = lik

    bij = pij[0:2] / lij
    bji = -bij
    bik = pik[0:2] / lik
    bki = -bik

    Rot90 = Rot(np.pi/2)
    bijp = Rot90.dot(bij)
    bjip = Rot90.dot(bji)
    bikp = Rot90.dot(bik)
    bkip = Rot90.dot(bki)

    if (bij.dot(bikp) > 0):
        alpha = np.arccos(bij.dot(bik)) # ajik
    else:
        alpha = 2*np.pi - np.arccos(bij.dot(bik))

    return alpha

def compute_alpha_dot(p, v):
    # Analytic calculation of alpha_dot from alpha and velocities
    alpha_dot = np.zeros(3) # a312 a123 a231
    alpha = compute_alphas_from_p(p)

    p1 = p[0:2]
    p2 = p[2:4]
    p3 = p[4:6]
    v1 = v[0:2]
    v2 = v[2:4]
    v3 = v[4:6]

    l12 = la.norm(p2-p1)
    l21 = l12
    l23 = la.norm(p3-p2)
    l32 = l23
    l31 = la.norm(p1-p3)
    l13 = l31

    b12 = (p2 - p1) / l12
    b21 = -b12
    b23 = (p3 - p2) / l23
    b32 = -b23
    b31 = (p1 - p3) / l31
    b13 = -b31

    Pb12 = np.eye(2) - np.outer(b12,b12)
    Pb13 = np.eye(2) - np.outer(b13,b13)
    Pb21 = np.eye(2) - np.outer(b21,b21)
    Pb23 = np.eye(2) - np.outer(b23,b23)
    Pb31 = np.eye(2) - np.outer(b31,b31)
    Pb32 = np.eye(2) - np.outer(b32,b32)

    alpha312_dot = -((Pb13.dot(v3-v1)).T.dot(b12)/l13  + b13.dot(Pb12).dot(v2-v1)/l12)/np.sin(alpha[0])
    alpha123_dot = -((Pb21.dot(v1-v2)).T.dot(b23)/l21  + b21.dot(Pb23).dot(v3-v2)/l23)/np.sin(alpha[1])
    alpha231_dot = -((Pb32.dot(v2-v3)).T.dot(b31)/l32  + b32.dot(Pb31).dot(v1-v3)/l31)/np.sin(alpha[2])

    alpha_dot[0] = alpha312_dot
    alpha_dot[1] = alpha123_dot
    alpha_dot[2] = alpha231_dot

    return alpha_dot

def M1t123(alpha):
    # Eq 9
    # We assume that alpha = [a312 a123 a231]
    M = np.zeros((4,4))
    M[0:2,0:2] = -np.sin(alpha[1])*(Rot(alpha[0]).T)
    M[0:2,2:4] =  np.sin(alpha[2])*np.eye(2)
    M[2:4,0:2] =  np.sin(alpha[1])*np.eye(2)
    M[2:4,2:4] =  np.sin(alpha[0])*(Rot(alpha[2]).T) - np.sin(alpha[1])*np.eye(2)

    return M

def M2t123(alpha, alpha_dot):
    # Eq 13
    # We assume that alpha = [a312 a123 a231], and the same for alpha_dot
    M = np.zeros((4,4))
    M[0:2,0:2] = -alpha_dot[2]*np.cos(alpha[2])*np.eye(2)
    M[0:2,2:4] = -alpha_dot[1]*np.cos(alpha[1])*Rot(alpha[0]).T -alpha_dot[0]*np.sin(alpha[1])*Rot(alpha[0]+np.pi/2).T
    M[2:4,0:2] =  alpha_dot[1]*np.cos(alpha[1])*np.eye(2) -alpha_dot[0]*np.cos(alpha[0])*Rot(alpha[2]).T -alpha_dot[2]*np.sin(alpha[0])*Rot(alpha[2]+np.pi/2).T
    M[2:4,2:4] =  alpha_dot[1]*np.cos(alpha[1])*np.eye(2)

    return M


# Arbitrary initial positions
p = np.random.rand(6)*50 - 100 # p1, p2, p3
alpha = compute_alphas_from_p(p) # a312 a123 a231

# Check Lemma 1
p1 = p[0:2]
p2 = p[2:4]
p3 = p[4:6]
p21p31 = np.zeros(4)
p21p31[0:2] = p1-p2
p21p31[2:4] = p1-p3
M1 = M1t123(alpha)
M1.dot(p21p31) # Almost zero :)

# Check Example 1 (in the previous version of the paper, the example is wrong :O )
alpha_ex = np.array([5.83, 5.64, 4.23])
alpha_dot_ex = np.array([-1.2, 0.8, 0.4])
M2 = M2t123(alpha_ex, alpha_dot_ex)

# Check Theorem 1 & Theorem 2
# First, we need to move the robots, lets do the isosceles triangle experiment

tf = int(r1_log[-1][0] / 1000)
dt = 0.02
dt_inv = 1.0/dt # Sampling frequency in sec^-1

log_time = np.linspace(0, tf, tf*dt_inv)
log_p = np.zeros((6, np.size(log_time)))
log_v = np.zeros((6, np.size(log_time)))
log_alpha = np.zeros((3, np.size(log_time)))
log_alpha_dot = np.zeros((3, np.size(log_time)))

log_error_th1 = np.zeros((4, np.size(log_time)))
log_error_th2 = np.zeros((4, np.size(log_time)))

p13p21_hat = np.zeros(4) # For the estimator in Theorem 2
k = 5 # estimator gain kc in Eq. 21

# Conversion matrix for robot's velocities

R = 3.35 # Wheel's radius cm
L = 12.4 # Nu sep que es cm
Mc = np.array([[R/L, -R/L],[R/2.0, R/2.0]])

beg_time = 0
end_time = 1000

for i in range(np.size(log_time)):

    if(i < beg_time*50):
        continue


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

    w = r1_vel[0]
    W = np.array([[0, -w],[w, 0]])
    Wblock = block_diag(W, W)

    # All the velocities are measured in body axes, there is no y component, our robots do not slide
    v = np.zeros(6)
    v[0:2] = np.array([0, r1_vel[1]])
    v[2:4] = np.array([0, r2_vel[1]])
    v[4:6] = np.array([0, r3_vel[1]])

    # Measurements
    # Relative velocities are easy in this case because only robot 2 is moving
    v21 = v[0:2] - v[2:4]
    v31 = v[0:2] - v[4:6]
    v13 = -v31
    v23 = v[4:6] - v[2:4]

    alpha_measurements = 0
    alpha_dot_measurements = 0

    if(np.abs(r1_log[i][4]) < 9000  and np.abs(r1_log[i][8]) < 9000 and np.abs(r2_log[i][4]) < 9000  and np.abs(r2_log[i][8]) < 9000): # if four markers were detected
        if r1_log[i][3] == 2:
            p12 = np.array([r1_log[i][4],-r1_log[i][6],-r1_log[i][5]]) * 100.0
            p13 = np.array([r1_log[i][8],-r1_log[i][10],-r1_log[i][9]]) * 100.0
        else:
            p13 = np.array([r1_log[i][4],-r1_log[i][6],-r1_log[i][5]]) * 100.0
            p12 = np.array([r1_log[i][8],-r1_log[i][10],-r1_log[i][9]]) * 100.0

        if r2_log[i][3] == 1:
            p21 = np.array([r2_log[i][4],r2_log[i][6],r2_log[i][5]]) * 100.0
            p23 = np.array([r2_log[i][8],r2_log[i][10],r2_log[i][9]]) * 100.0
        else:
            p23 = np.array([r2_log[i][4],r2_log[i][6],r2_log[i][5]]) * 100.0
            p21 = np.array([r2_log[i][8],r2_log[i][10],r2_log[i][9]]) *100.0

        # The algoritm is programmed for a312 a123 a231
        alpha_123 = compute_alpha_kij_from_arucos(p23, p21)
        alpha_312 = compute_alpha_kij_from_arucos(p12, p13)

        #if(alpha_123 > np.pi):
        #    alpha_123 = 2*np.pi - alpha_123

        #if(alpha_312 > np.pi):
        #    alpha_312 = 2*np.pi - alpha_312

        if(alpha_123 > np.pi):
            alpha_231 = 5*np.pi - alpha_123 - alpha_312
        else:
            alpha_231 = np.pi - alpha_123 - alpha_312

        print(int(log_time[i]*1000 + 100), alpha_312*180/np.pi, alpha_123*180/np.pi, alpha_231*180/np.pi)
        alpha_measurements = 1
        alpha = np.zeros(3)
        alpha[0] = alpha_312
        alpha[1] = alpha_123
        alpha[2] = alpha_231

        # Relative positions from Aruco measurements to compare
        p13p21 = np.zeros(4)
        p13p21[0:2] = p13[0:2]
        p13p21[2:4] = p21[0:2]

    # alpha_dot numerical, with the experimental data, we need to use this one
    if ((i > 1) and not(np.isnan(log_alpha[0, i-1])) and alpha_measurements):
        alpha_dot = (alpha - log_alpha[:, i-1]) / dt
        alpha_dot_measurements = 1
        print("Alpha dot: ", alpha_dot*180/np.pi)

    # Theorem 1, Eq 16
    if (alpha_dot_measurements == 1):
        M2 = M2t123(alpha, alpha_dot) # alpha = [a312 a123 a231]
        aux = np.zeros(4) # We will use it for Theorem 2 too.
        aux[0:2] = np.sin(alpha[1])*Rot(alpha[0]).T.dot(v21) - np.sin(alpha[2])*v31
        aux[2:4] = np.sin(alpha[0])*Rot(alpha[2]).T.dot(v13) - np.sin(alpha[1])*v23
        if(la.cond(M2) < 1e6):
            p13p21estTh1 = la.inv(M2).dot(aux)
            print("Direct measurement vel: ", v21, v31, v13, v23)
            print("Direct measurement: ", p13p21estTh1[0:2], p13[0:2], p13p21estTh1[2:4], p21[0:2])

    # Theorem 2, Eq 21
    v13v21 = np.zeros(4)
    v13v21[0:2] = v13
    v13v21[2:4] = v21

    if(alpha_dot_measurements == 1):
        p13p21_hat_dot = v13v21 - k*M2.T.dot(M2).dot(p13p21_hat) + k*M2.T.dot(aux) + Wblock.dot(p13p21_hat)
        p13p21_hat = p13p21_hat + p13p21_hat_dot * dt
        print("Estimator: ", p13p21_hat[0:2], p13[0:2], p13p21_hat[2:4], p21[0:2])
    else:
        p13p21_hat_dot = v13v21 + Wblock.dot(p13p21_hat)
        p13p21_hat = p13p21_hat + p13p21_hat_dot * dt

    # Logs

    log_v[:,i] = v
    if(alpha_measurements == 0):
        log_alpha[:, i] = np.NaN*np.ones(3)
    else:
        log_alpha[:, i] = alpha
    if(alpha_dot_measurements == 0):
        log_alpha_dot[:, i] = np.NaN*np.ones(3)
        log_error_th1[:,i] = np.NaN*np.ones(4)
        log_error_th2[:,i] = np.NaN*np.ones(4)
    else:
        log_alpha_dot[:, i] = alpha_dot
        log_error_th1[:,i] = p13p21 - p13p21estTh1
        log_error_th2[:,i] = p13p21 - p13p21_hat

    if(i > end_time*50):
        break

# Postprocessing

# Positions of the agents
#fig, axis = pl.subplots(1,1)
#axis.plot(log_p[0,:], log_p[1,:], 'r')
#axis.plot(log_p[2,:], log_p[3,:], 'og')
#axis.plot(log_p[4,:], log_p[5,:], 'ob')
#axis.set_xlabel("X [cm]")
#axis.set_ylabel("Y [cm]")

# Error signals Theorem 1
#fig, axis = pl.subplots(4,1, sharex=True)
#axis[0].set_title("Results from Theorem 1")
#axis[0].plot(log_time[:], log_error_th1[0,:])
#axis[1].plot(log_time[:], log_error_th1[1,:])
#axis[2].plot(log_time[:], log_error_th1[2,:])
#axis[3].plot(log_time[:], log_error_th1[3,:])
#axis[0].set_ylabel("$\hat p_{{13}_x} -  p_{{13}_x}$ [cm]")
#axis[1].set_ylabel("$\hat p_{{13}_y} -  p_{{13}_y}$ [cm]")
#axis[2].set_ylabel("$\hat p_{{21}_x} -  p_{{21}_x}$ [cm]")
#axis[3].set_ylabel("$\hat p_{{21}_y} -  p_{{21}_y}$ [cm]")
#axis[3].set_xlabel("Time [s]")
#for ax in axis:
#    ax.grid()

# Error signals Theorem 2
#fig, axis = pl.subplots(4,1, sharex=True)
#axis[0].set_title("Results from Theorem 2")
#axis[0].plot(log_time[:], log_error_th2[0,:])
#axis[1].plot(log_time[:], log_error_th2[1,:])
#axis[2].plot(log_time[:], log_error_th2[2,:])
#axis[3].plot(log_time[:], log_error_th2[3,:])
#axis[0].set_ylabel("$\hat p_{{13}_x} -  p_{{13}_x}$ [cm]")
#axis[1].set_ylabel("$\hat p_{{13}_y} -  p_{{13}_y}$ [cm]")
#axis[2].set_ylabel("$\hat p_{{21}_x} -  p_{{21}_x}$ [cm]")
#axis[3].set_ylabel("$\hat p_{{21}_y} -  p_{{21}_y}$ [cm]")
#axis[3].set_xlabel("Time [s]")
#for ax in axis:
#    ax.grid()

# Velocity signals
fig, axis = pl.subplots(4,1, sharex=True)
axis[0].set_title("Relative velocities")
axis[0].plot(log_time[:], log_v[0,:])
axis[1].plot(log_time[:], log_v[1,:])
axis[2].plot(log_time[:], log_v[2,:])
axis[3].plot(log_time[:], log_v[3,:])
axis[3].set_xlabel("Time [s]")
for ax in axis:
    ax.grid()

# Interior angles signals
fig, axis = pl.subplots(3,1, sharex=True)
axis[0].set_title("Interior angles")
axis[0].plot(log_time[:], log_alpha[0,:]*180/np.pi, '-')
axis[1].plot(log_time[:], log_alpha[1,:]*180/np.pi, '-')
axis[2].plot(log_time[:], log_alpha[2,:]*180/np.pi, '-')
axis[0].set_ylabel("$\\alpha_{321} [degrees]$")
axis[1].set_ylabel("$\\alpha_{123} [degrees]$")
axis[2].set_ylabel("$\\alpha_{231} [degrees]$")
axis[2].set_xlabel("Time [s]")
for ax in axis:
    ax.grid()

# Interior angle velocities signals
fig, axis = pl.subplots(3,1, sharex=True)
axis[0].set_title("Interior angle velocities")
axis[0].plot(log_time[:], log_alpha_dot[0,:]*180/np.pi, '-')
axis[1].plot(log_time[:], log_alpha_dot[1,:]*180/np.pi, '-')
axis[2].plot(log_time[:], log_alpha_dot[2,:]*180/np.pi, '-')
axis[0].set_ylabel("$\\alpha_{321} [degrees/sec]$")
axis[1].set_ylabel("$\\alpha_{123} [degrees/sec]$")
axis[2].set_ylabel("$\\alpha_{231} [degrees/sec]$")
axis[2].set_xlabel("Time [s]")
for ax in axis:
    ax.grid()


pl.show()
