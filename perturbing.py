import numpy as np
import pylab as pl
import scipy.special.lambertw as lambertw
import scipy.optimize as opt
import matplotlib.pyplot as plt

pl.rcParams["figure.figsize"] = (4, 6)
#pl.rcParams["text.usetex"] = True

#constants
G = 6.67408e-11 # gravitational constant in SI
m_H = 1.67e-27 # mass of hydrogen in kg
mu_i = 0.5 # mean molecular weight ionized
mu_n = 1. # mean molecular weight neutral
k = 1.38e-23 # Boltzmann constant in SI

#code units
unit_length_in_si = 1.2e13
r_min = 0.2 * unit_length_in_si
r_max = 2. * unit_length_in_si
unit_mass_in_si = 2.479e31
unit_density_in_si = unit_mass_in_si / (unit_length_in_si**3)
unit_time_in_si = np.sqrt(unit_length_in_si**3 / (unit_mass_in_si * G))
unit_velocity_in_si = unit_length_in_si / unit_time_in_si
unit_pressure_in_si = unit_mass_in_si / (unit_length_in_si * unit_time_in_si**2)


#input
T_n = 500 # temperature of the neutral region
T_i = 10000 # temperature of the ionized region
rho_infty = 1.e-20*np.exp(-1.5)  # density at infinity
M_star = 1.5*unit_mass_in_si #stellar mass
R_IF1 = 0.35*unit_length_in_si# ionization front radius
R_IF2 = 0.30*unit_length_in_si# ionization front radius
R_IF3 = 0.325*unit_length_in_si# ionization front radius
R_IF4 = 0.375*unit_length_in_si# ionization front radius
print(rho_infty, "density at infinity")

#sound speed calculations
def sound_speed(T,mu):
  return np.sqrt(k*T/(mu*m_H))

c_1 = sound_speed(T_n,mu_n)
#c_2 = sound_speed(T_i,mu_i)
jump = 50.
c_2 = np.sqrt(jump)*c_1

print(c_1, "neutral sound speed")

# Bondi radii
def bondi_radius(G,M,a):
  return G*M/(2*a*a)

bondi_n = bondi_radius(G, M_star, c_1)
bondi_i = bondi_radius(G, M_star, c_2)


print("check params")
print("neutral bondi", bondi_n)
print("neutral sound", c_1)
print("pressure contrast", jump)
print("Ionization radius 1", R_IF1)
print("Ionization radius 2", R_IF2)

# Lambert W function argument

def lambertarg(x):
  return -np.exp(3. + 4. * np.log(x) - 4. * x)

#Real Lambert W function 
#def u2cs2(x):
#  return np.where(x <= 1., -lambertw(lambertarg(x), 0),
#                  -lambertw(lambertarg(x), -1)).real
def u2cs2(x):
  return np.where(x <= 1.,0,-lambertw(lambertarg(x), -1)).real

#velocity profile
def vel(c,bondi,r):
  return -c* np.sqrt(u2cs2(bondi/r))

#denstiy profile
#def dens(c,bondi,r)
#  return -c* np.sqrt(u2cs2(bondi/r))

#find neutral values at the IF

v_11 = vel(c_1,bondi_n,R_IF1)
v_12 = vel(c_1,bondi_n,R_IF2)
v_13 = vel(c_1,bondi_n,R_IF3)
v_14 = vel(c_1,bondi_n,R_IF4)
#print(v_1, "neutral speed at IF")

def rho1(rho_infty, v_1,c_1, bondi_n, R_IF):
  return rho_infty*np.exp((-v_1*v_1)/(2*c_1*c_1)+ 2*bondi_n/R_IF)

#print(rho_1, "density at i-front")

#solution for the Gamma function
def gamma_factor(u, a1, a2):
  return ((a1*a1 + u*u) - np.sqrt((a1*a1 + u*u)**2 - 4*a2*a2*u*u))/(2*a2*a2)

#Find ionized values at the IF

def vel2(v_1, c_1, c_2):
  return v_1/gamma_factor(v_1,c_1,c_2)

def rho2(rho_infty, v_1,c_1,c_2, bondi_n, R_IF):
  return rho1(rho_infty, v_1,c_1, bondi_n, R_IF)*gamma_factor(v_1,c_1,c_2)

#create array of r values
r = np.linspace(r_min, r_max, 1001)
#r = 0.5 * (r[1:] + r[:-1])
r1 = np.linspace(r_min, r_max/2, 1001)
r2 = np.linspace(r_max/2, r_max, 1001)

print("lambert", lambertw(-0.35,0), -np.exp(-1))


def arg_w_vel(r, R_IF, v_2, bondi_i):
  return np.log(R_IF**4*v_2**2/(r**4*c_2**2)) - 4*bondi_i/r - v_2**2/c_2**2 + 4*bondi_i/R_IF

def lamb(r, R_IF, v_2, bondi_i):
  return lambertw(-np.exp(arg_w_vel(r, R_IF, v_2, bondi_i)),-1).real

def vel(r, R_IF, v_1, bondi_i):
   return c_2*np.sqrt(-lamb(r, R_IF, vel2(v_1,c_1,c_2), bondi_i))

#test vel at R_IF should be v_2

#v2 = vel(R_IF, R_IF, v_2, bondi_i)
#print("vel at if", v_2, v2)


def bondi_u(r, c_1, c_2, R_IF, bondi_i, v_1):
  return np.where(r <= R_IF, -vel(r, R_IF, v_1, bondi_i),
                  -c_1 * np.sqrt(u2cs2(bondi_n / r)))


def bondi_rho(r, rho_infty, v_1, R_IF, c_1, c_2, bondi_i, bondi_n):
  return np.where(r < R_IF, \
                  rho2(rho_infty, v_1,c_1,c_2, bondi_n, R_IF)*vel2(v_1,c_1,c_2)*R_IF**2/(bondi_u(r, c_1, c_2, R_IF, bondi_i, v_1)*r**2),\
                  rho2(rho_infty, v_1,c_1,c_2, bondi_n, R_IF)*vel2(v_1,c_1,c_2)*R_IF**2/(bondi_u(r, c_1, c_2, R_IF, bondi_i, v_1)*r**2))

def bondi_P(r, rho_infty, v_1, R_IF, c_1, c_2, bondi_i, bondi_n):
  return np.where(r < R_IF, c_2**2 *bondi_rho(r, rho_infty,v_1, R_IF, c_1, c_2, bondi_i, bondi_n),
                    c_1**2 *bondi_rho(r, rho_infty, v_1, R_IF, c_1, c_2, bondi_i, bondi_n)) 

#gaussian

def gaussian(x,x0,sigma):
  return (1/np.sqrt(2*np.pi*sigma*sigma))*np.exp(-np.power((x - x0)/sigma, 2.)/2.)

#plot

fig, ax = pl.subplots(4, 1, sharex = True)


#velocity

ax[0].plot(r, bondi_u(r, c_1, c_2, R_IF1, bondi_i, v_11),'r--')
#ax[0].plot(r, bondi_u(r, c_1, c_2, R_IF2, bondi_i, v_12),'g--')
#ax[0].plot(r, bondi_u(r, c_1, c_2, R_IF3, bondi_i, v_13),'b--')
#ax[0].plot(r, bondi_u(r, c_1, c_2, R_IF4, bondi_i, v_14),'y--')
#density

ax[1].plot(r, bondi_rho(r, rho_infty, v_11, R_IF1, c_1, c_2, bondi_i, bondi_n),'r--')
#ax[1].plot(r, bondi_rho(r, rho_infty, v_12, R_IF2, c_1, c_2, bondi_i, bondi_n),'g--')
#ax[1].plot(r, bondi_rho(r, rho_infty, v_13, R_IF3, c_1, c_2, bondi_i, bondi_n),'b--')
#ax[1].plot(r, bondi_rho(r, rho_infty, v_14, R_IF4, c_1, c_2, bondi_i, bondi_n),'y--')
#pressure

ax[2].plot(r, bondi_P(r, rho_infty, v_11, R_IF1, c_1, c_2, bondi_i, bondi_n),'r--', label = "Bondi Analytic")
#ax[2].plot(r, bondi_P(r, rho_infty, v_12, R_IF2, c_1, c_2, bondi_i, bondi_n),'g--')
#ax[2].plot(r, bondi_P(r, rho_infty, v_13, R_IF3, c_1, c_2, bondi_i, bondi_n),'b--')
#ax[2].plot(r, bondi_P(r, rho_infty, v_14, R_IF4, c_1, c_2, bondi_i, bondi_n),'y--')
#ax[2].legend(loc = "best")

ax[2].set_xlabel("radius")
ax[1].set_ylabel("density")
ax[0].set_ylabel("velocity")
#ax[2].set_ylabel("pressure")

x = np.linspace(0, 10, 1001)

#pertubation
ax[3].plot(x,1000*gaussian(x,5,0.1))

#steady accreation
#ax[3].plot(r1, r1*r1*bondi_rho(r1, rho_infty, v_11, R_IF1, c_1, c_2, bondi_i, bondi_n)*bondi_u(r1, c_1, c_2, R_IF1, bondi_i, v_11),'r--')
#ax[3].plot(r1, r1*r1*bondi_rho(r1, rho_infty, v_12, R_IF2, c_1, c_2, bondi_i, bondi_n)*bondi_u(r1, c_1, c_2, R_IF2, bondi_i, v_12),'g-')
#ax[3].plot(r2, r2*r2*bondi_rho(r2, rho_infty, v_13, R_IF3, c_1, c_2, bondi_i, bondi_n)*bondi_u(r2, c_1, c_2, R_IF3, bondi_i, v_13),'b--')
#ax[3].plot(r2, r2*r2*bondi_rho(r2, rho_infty, v_14, R_IF4, c_1, c_2, bondi_i, bondi_n)*bondi_u(r2, c_1, c_2, R_IF4, bondi_i, v_14),'y-')





pl.tight_layout()
pl.savefig("profiles_perturbed.png")

