import matplotlib.pyplot as pl
import numpy as np
from numpy import linalg as la

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

    alpha312_dot = -((Pb13.dot(v3-v1)).T.dot(b12)/l13  + b13.dot(Pb12).dot(v2-v1)/l13)/np.sin(alpha[0])
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

tf = 30
dt_inv = 100 # Sampling frequency in sec^-1
dt = 0.01

log_time = np.linspace(0, tf, tf*dt_inv)
log_p = np.zeros((6, np.size(log_time)))
log_v = np.zeros((6, np.size(log_time)))
log_alpha = np.zeros((3, np.size(log_time)))
log_alpha_dot = np.zeros((3, np.size(log_time)))

log_error_th1 = np.zeros((4, np.size(log_time)))
log_error_th2 = np.zeros((4, np.size(log_time)))

# Initial conditions
p1 = np.array([50,100])  # cm
p2 = np.array([0, 0])
p3 = np.array([100, 0])

p = np.zeros(6)
p[0:2] = p1
p[2:4] = p2
p[4:6] = p3

p13p21_hat = np.zeros(4) # For the estimator in Theorem 2
k = 5 # estimator gain kc in Eq. 21

for i in range(np.size(log_time)):

    # Velocity of robot 1
    v1y = 10*2*np.pi*np.cos(2*np.pi*log_time[i])

    v = np.zeros(6)
    v[0:2] = np.array([0, v1y])
    v[2:4] = np.array([-2*v1y,0])
    v[4:6] = np.array([0,0])

    # Measurements
    v21 = v[0:2] - v[2:4]
    v31 = v[0:2] - v[4:6]
    v13 = -v31
    v23 = v[4:6] - v[2:4]
    alpha = compute_alphas_from_p(p)

    # alpha_dot numerical, with the experimental data, we need to use this one
    if i > 1:
        num_alpha_dot = (alpha - log_alpha[:, i-1]) / dt
    else:
        num_alpha_dot = np.zeros(3)

    alpha_dot = compute_alpha_dot(p,v)

    # Check that numerical and analytic are similar
    # print(alpha_dot - num_alpha_dot)

    # Theorem 1, Eq 16
    M2 = M2t123(alpha, alpha_dot) # alpha = [a312 a123 a231]
    aux = np.zeros(4) # We will use it for Theorem 2 too.
    aux[0:2] = np.sin(alpha[1])*Rot(alpha[0]).T.dot(v21) - np.sin(alpha[2])*v31
    aux[2:4] = np.sin(alpha[0])*Rot(alpha[2]).T.dot(v13) - np.sin(alpha[1])*v23
    p13p21estTh1 = la.inv(M2).dot(aux)

    p13p21 = np.zeros(4) # So we can compare the estimation with the simulation
    p13p21[0:2] = p[4:6]-p[0:2]
    p13p21[2:4] = p[0:2]-p[2:4]

    # Theorem 2, Eq 21
    v13v21 = np.zeros(4)
    v13v21[0:2] = v13
    v13v21[2:4] = v21
    p13p21_hat_dot = v13v21 - k*M2.T.dot(M2).dot(p13p21_hat) + k*M2.T.dot(aux)

    p13p21_hat = p13p21_hat + p13p21_hat_dot * dt


    # Logs
    log_error_th1[:,i] = p13p21 - p13p21estTh1
    log_error_th2[:,i] = p13p21 - p13p21_hat

    log_p[:,i] = p
    log_v[:,i] = v
    log_alpha[:, i] = alpha
    log_alpha_dot[:, i] = alpha_dot

    # Simulated world
    p = p + v *dt


# Postprocessing

# Positions of the agents
fig, axis = pl.subplots(1,1)
axis.plot(log_p[0,:], log_p[1,:], 'r')
axis.plot(log_p[2,:], log_p[3,:], 'og')
axis.plot(log_p[4,:], log_p[5,:], 'ob')
axis.set_xlabel("X [cm]")
axis.set_ylabel("Y [cm]")

# Error signals Theorem 1
fig, axis = pl.subplots(4,1, sharex=True)
axis[0].set_title("Results from Theorem 1")
axis[0].plot(log_time[:], log_error_th1[0,:])
axis[1].plot(log_time[:], log_error_th1[1,:])
axis[2].plot(log_time[:], log_error_th1[2,:])
axis[3].plot(log_time[:], log_error_th1[3,:])
axis[0].set_ylabel("$\hat p_{{13}_x} -  p_{{13}_x}$ [cm]")
axis[1].set_ylabel("$\hat p_{{13}_y} -  p_{{13}_y}$ [cm]")
axis[2].set_ylabel("$\hat p_{{21}_x} -  p_{{21}_x}$ [cm]")
axis[3].set_ylabel("$\hat p_{{21}_y} -  p_{{21}_y}$ [cm]")
axis[3].set_xlabel("Time [s]")
for ax in axis:
    ax.grid()

# Error signals Theorem 2
fig, axis = pl.subplots(4,1, sharex=True)
axis[0].set_title("Results from Theorem 2")
axis[0].plot(log_time[:], log_error_th2[0,:])
axis[1].plot(log_time[:], log_error_th2[1,:])
axis[2].plot(log_time[:], log_error_th2[2,:])
axis[3].plot(log_time[:], log_error_th2[3,:])
axis[0].set_ylabel("$\hat p_{{13}_x} -  p_{{13}_x}$ [cm]")
axis[1].set_ylabel("$\hat p_{{13}_y} -  p_{{13}_y}$ [cm]")
axis[2].set_ylabel("$\hat p_{{21}_x} -  p_{{21}_x}$ [cm]")
axis[3].set_ylabel("$\hat p_{{21}_y} -  p_{{21}_y}$ [cm]")
axis[3].set_xlabel("Time [s]")
for ax in axis:
    ax.grid()

pl.show()