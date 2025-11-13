import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm

# Mercury constants
M_M = 3.3011e23  # Mass of Mercury
R_M = 2439.7e3  # [m] Radius of Mercury
mu_M = 2.20321e13  # [m^3 * s^-2] Grav. Parameter of Mercury

# OTher
d = 4.60012e10  # [m] distance from sun to mercury
e_M = 0.20563 # eccentricity mercury
a_SM = 57909227e3 # [m] semi major axis
h = 750e3 # [m]
c = 3e8 # m/s

# Sun constants
M_S = 1.9885e30  # Mass of the Sun
mu_S = 1.32712440019e20  # [m^3 * s^-2] Grav. Parameter of the Sun
R_S = 6957e5 # m

# --- SRP Constants ---
c = 299792458  # [m/s] speed of light
I0 = 1361.0    # W/m^2 at 1 AU
AU = 1.495978707e11  # m

# Solar Arrays (SA) - flat panels
A_SA = 10.6       # m^2 per array
q_SA = 0.6        # reflectivity

# Cylinder body
C_R = 1.3 # reflection coefficient
radius_cyl = 0.9   # m
height_cyl = 2.0   # m

# Center of pressure for combined SRP (CoM --> CoP)
d_cp = np.array([0.0, 0.0, 0.1])   # meters, single offset for torque computation

# --- Functions ---
def solar_irradiance(r_sat_from_sun):
    """Compute solar irradiance at spacecraft position (W/m^2)."""
    r = np.linalg.norm(r_sat_from_sun)
    return I0 * (AU / r)**2

def srp_force_and_torque(r_sun_to_sat, r_mercury_to_sat, R_body_to_inertial):
    
    S = solar_irradiance(r_sun_to_sat)
    
    R_s_hat = r_sun_to_sat / np.linalg.norm(r_sun_to_sat)
    R_m_hat = r_mercury_to_sat / np.linalg.norm(r_mercury_to_sat)   
    
    A_s = 2*A_SA + 2*radius_cyl*height_cyl*(1-np.abs(R_s_hat@R_m_hat))
    F_srp = S/c*C_R*A_s*R_s_hat
    F_srp_b = R_body_to_inertial.T @ F_srp

    d_cp1 = np.transpose(R_body_to_inertial) @ d_cp
    M_srp = np.cross(d_cp1, F_srp_b)
    
    return F_srp, M_srp   

#spacecraft characteristics
I_xx = 2281.3
I_yy = I_xx
I_zz = 1634.2

m = 500

# Define initial conditions
alpha = 90 # Orbital inclination relative to the Sun (0 degrees means parallel to Mercury-Sun plane, 90)
v_init = np.sqrt(mu_M / (h + R_M)) # orbit velocity at given altitude
p_orbit = 2 * np.pi * np.sqrt( ((R_M + h)**3) / mu_M ) # period of orbit at given altitude

# To simulate yaw rotation of the satellite
euler_dot0 = 2*np.pi/p_orbit

R_e = np.array([0.0, 0.0, float(R_M + h)])     # position in world frame
V_e = np.array([v_init*np.sin(np.radians(alpha)), v_init*np.cos(np.radians(alpha)), 0]) # velocity in world frame

# Filter the velocities because sin(90) does not give 0
if np.abs(V_e[1]) < 0.1:
    V_e[1] = 0
elif np.abs(V_e[0]) < 0.1:
    V_e[0] = 0

xB = [1,0,0]
yB = [0,-1,0]
zB = [0,0,-1]

R_world_to_body = np.column_stack((xB, yB, zB))
r = R.from_matrix(R_world_to_body)

# Initial euler angles at starting position
euler0 = r.as_euler('zyx', degrees=False)  # [psi, theta, phi] 
euler_nom = euler0

euler = r.as_euler('zyx', degrees=False)
omega = np.zeros(3)

# simulation settings
t=0
t_tot = p_orbit
dt = 0.5

I = np.array([[I_xx, 0 , 0],
              [0, I_yy, 0],
              [0, 0, I_zz]])

def rot_matrix(psi, theta, phi):

    cpsi, spsi = np.cos(psi), np.sin(psi)
    cth, sth = np.cos(theta), np.sin(theta)
    cphi, sphi = np.cos(phi), np.sin(phi)

    Rz = np.array([[cpsi, -spsi, 0], # yaw axis rotation
                   [spsi,  cpsi, 0],
                   [0,        0, 1]])
    
    Ry = np.array([[cth,   0, sth], # pitch axis rotation
                    [0, 1   ,   0],
                    [-sth, 0, cth]])
    
    Rx = np.array([[1, 0   ,     0], # roll axis rotation
                   [0, cphi, -sphi],
                   [0, sphi,  cphi]])
    
    R = Rz @ Ry @ Rx # computing rotation matrix

    return R

def body_to_euler_rates(theta, phi, omega):
    
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    tth = np.tan(theta)

    T = np.array([
        [1, sphi*tth,   cphi*tth],
        [0, cphi,      -sphi],
        [0, sphi/cth,  cphi/cth]
    ])
    return T @ omega

def dynamics(euler, omega, F_b, M_b):
    psi = euler[0]
    theta = euler[1]
    phi = euler[2] 
    
    A_e = F_b / m

    omega_dot = np.linalg.inv(I) @ (M_b - np.cross(omega, (I @ omega)))
    
    euler_dot = body_to_euler_rates(theta, phi, omega)
    return A_e, omega_dot, euler_dot

def plot_sphere(ax, center, radius, color, name=None):

    """Draws a 3D sphere at a given position."""
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=0.7)
    if name:
        ax.text(center[0], center[1], center[2] + radius * 1.2, name,
                color=color, fontsize=10, ha='center')

def gravity_force(r_sat,m,mu):
    r = np.array(r_sat, dtype = float)
    r_norm = np.linalg.norm(r)
    F_grav = -mu * m * r / r_norm**3

    return F_grav

def gravity_gradient_torque(Fg, inertia_tensor, m, r_norm, r_body_to_inertial):
    F_norm = np.linalg.norm(Fg)
    if F_norm == 0:
        return np.zeros(3)

    r_hat_body = -Fg / F_norm  
    r_hat_body = np.transpose(r_body_to_inertial) @ r_hat_body # direction from planet to satellite in body frame
    Tg = 3 * (F_norm / (m * r_norm)) * np.cross(r_hat_body, inertia_tensor @ r_hat_body)
    return Tg

if __name__ == "__main__":
    rw_max_rpm = 3000 # rpm
    I_rw_x = I_rw_y = 15
    I_rw_z = 50

    first_phase_timestamp = 0
    second_phase_timestamp = 0 
    third_phase_timestamp = 0

    L_sat_t = []
    second_phase = False
    
    L_sat = np.zeros(3)
    omega2 = np.zeros(3)

    M_rw = np.zeros(3) # moment of reaction wheels
    rw_current_rpm = np.zeros(3)

    # Plots    
    #####################################################################
    
    # 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Planets
    plot_sphere(ax, [0, 0, 0], R_M, 'Red', name='Mercury')

    # Initial point and line
    point, = ax.plot([], [], [], 'ro', markersize=8)  # the moving point
    trace, = ax.plot([], [], [], 'b-', lw=1)          # the trajectory (trace)

    # Set axis limits
    ax.set_xlim(-4e6, 4e6)
    ax.set_ylim(-4e6, 4e6)
    ax.set_zlim(-4e6, 4e6)
    ax.set_box_aspect([1,1,1])  # X:Y:Z ratio

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    #####################################################################
    xdata, ydata, zdata = [], [], []
    roll, pitch, yaw = [], [], []
    timestamps = []
    roll_moment, pitch_moment, yaw_moment = [],[],[]
    psi_t, theta_t, phi_t = [],[],[]
    
    while t < t_tot:
        
        # Magnitude of the Mercury-satellite vector
        R_abs = np.linalg.norm(R_e)

        r_sun_to_mercury = [0,-d,0]     # Define vector from Sun to Mercury
        r_sun_to_sat = r_sun_to_mercury + R_e   # Define vector from Sun to satellite

        # Calculate gravitational force due to Mercury and Sun
        Fg_mercury = gravity_force(R_e, m, mu_M)
        Fg_sun = gravity_force(r_sun_to_sat, m, mu_S)
        
        # Euler angles
        psi, theta, phi = euler

        # Vector from body to inertial frame
        R_body_to_inertial= rot_matrix(psi, theta, phi)
        
        # SRP force and moment
        F_srp, M_srp = srp_force_and_torque(r_sun_to_sat, r_sun_to_sat - r_sun_to_mercury, R_body_to_inertial)
        
        # Total force on satellite
        F_t = Fg_mercury

        # Calculate gravity gradient torques of Mercury and Sun
        M_bm = gravity_gradient_torque(Fg_mercury, I, m, R_abs, R_body_to_inertial)
        M_bs = gravity_gradient_torque(Fg_sun, I, m, np.linalg.norm(r_sun_to_sat), R_body_to_inertial)
        
        # Total torque on satellite
        M_bt1 = M_srp
        M_bt = np.zeros(3)

        if second_phase:
            M_bt = M_bt1

        #print(np.linalg.norm(M_bt1))
        a = np.array([M_bt1[0] / I_xx, M_bt1[1] / I_yy, M_bt1[2] / I_zz])
        #print(M_bt1[0])
        omega2 += a*dt

        L_sat = np.array([omega2[0]*I_xx, omega2[1]*I_yy, omega2[2]*I_zz])
        L_sat_t.append(L_sat)
        #print(np.linalg.norm(L_sat))
        
        if L_sat[0] > 14.9*0.9 or L_sat[1] > 14.9*0.9 or L_sat[2] > 14.9*0.9:
            first_phase_timestamp = t
            second_phase = True

        if second_phase:
            if euler[1] > np.radians(0.09) or euler[2] > np.radians(0.09):
                second_phase_timestamp = t

            
        A_e, omega_dot, euler_dot = dynamics(euler, omega, F_t, M_bt)
        euler_dot += [euler_dot0,0,0] # Add yaw velocity
        
        # Vertlet integration
        v_half = V_e + 0.5 * A_e * dt
        R_e = R_e + v_half * dt
        A_e2, omega_dot2, euler_dot2 = dynamics(euler, omega, gravity_force(R_e, m, mu_M), M_bt)
        V_e = v_half + 0.5 * A_e2 * dt

        # Integration for angular velocity and acceleration
        euler += euler_dot * dt
        omega += omega_dot * dt

        # Data for the 3D plot
        xdata.append(R_e[0])
        ydata.append(R_e[1])
        zdata.append(R_e[2])

        # Add angular velocities to plot
        roll.append(euler_dot[2])
        pitch.append(euler_dot[1])
        yaw.append(euler_dot[0] - euler_dot0)

        # Add euler angles to plot
        psi_t.append(psi)
        theta_t.append(theta)
        phi_t.append(phi)

        # Do something with the 3D plot
        point.set_data([R_e[0]], [R_e[1]])
        point.set_3d_properties([R_e[2]])

        # Draw trace on 3D plot
        trace.set_data(xdata, ydata)
        trace.set_3d_properties(zdata)
        
        # Add new timestamp
        timestamps.append(t)
        
        # Add torques to plot
        roll_moment.append(M_srp[0])
        pitch_moment.append(M_srp[1])
        yaw_moment.append(M_srp[2])

        # Increase timestamp
        t += dt


"""print("First: " + first_phase_timestamp)
print("Second: " + second_phase_timestamp)
"""
# Plot of torques
#####################################################################
fig2 = plt.figure()

plt.plot(timestamps, roll_moment , 'r-', label='Roll Moment')
plt.plot(timestamps, pitch_moment, 'g-', label='Pitch Moment')
plt.plot(timestamps, yaw_moment, 'b-', label='Yaw Moment')

plt.title('Torques')
plt.xlabel('Time [s]')
plt.ylabel('Torque [Nm]')
plt.grid(True)
plt.legend() 

# Plot of angular velocities
#####################################################################

fig3 = plt.figure()

plt.plot(timestamps, roll , 'r-', label='Roll')
plt.plot(timestamps, pitch, 'g-', label='Pitch')
plt.plot(timestamps, yaw, 'b-', label='Yaw')

plt.title('Angular Velocities')
plt.xlabel('Time [s]')
plt.ylabel('Angular Velocity [rad/s]')
plt.grid(True)
plt.legend() 

# Plot of euler angles
#####################################################################
fig4 = plt.figure()

plt.plot(timestamps,  psi_t, 'r-', label='Psi')
plt.plot(timestamps, theta_t, 'g-', label='Theta')
plt.plot(timestamps, phi_t, 'b-', label='Phi')

plt.title('Change in euler angles over one orbital period')
plt.xlabel('Time [s]')
plt.ylabel('Euler Angle [rad]')
plt.grid(True)
plt.legend() 

# Plot of angular momentum
#####################################################################
"""fig5 = plt.figure()

plt.plot(timestamps, L_sat_t, 'r-', label='Angular momentum')

plt.title('Angular momentum')
plt.xlabel('Time [s]')
plt.ylabel('Euler Angle [rad]')
plt.grid(True)
plt.legend()"""

# Show plots
plt.show()