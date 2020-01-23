import os
from multiprocessing import Process, Pool
import numpy as np
from tqdm import tqdm
import definition as harmonic
import poolscripts
import observable as observable
import hub_lats as hub
import harmonic as har_spec
from scipy.integrate import ode
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# This contains the stuff needed to calculate some expectations. Generally contains stuff
# that applies operators to the wave function
import evolve as evolve

# Contains lots of important functions.
import definition as definition
# Sets up the lattice for the system
import hub_lats as hub

# These also contain various important observable calculators
import harmonic as har_spec

# input units: THz (field), eV (t, U), MV/cm (peak amplitude, F0), Angstroms (systems[0]tice cst, a)
# they're then converted to t-normalised atomic units. bc='pbc' for periodic and 'abc' for antiperiodic


neighbour = []
neighbour_check = []
energy = []
doublon_energy = []
phi_original = []
phi_reconstruct = [0., 0.]
boundary_1 = []
boundary_2 = []
two_body = []
two_body_old=[]
error=[]
D=[]
X=[]
singlon=[]
systems=[]
number=3
nelec = (number, number)
nx = 6
ny = 0
t = 0.52
# t=1.91
# t=1
U = 1*t
delta = 0.1
cycles = 3
field= 32.9
# field=32.9*0.5
F0=10
a=4
#
# def info(title):
#     print(title)
#     print('module name:', __name__)
#     # only available on Unix
#     if hasattr(os, 'getppid'):
#         print('parent process:', os.getppid())
#     print('process id:', os.getpid())
#

def f(name):
    # info('function f')
    # print('hello', name)
    return name



pool = Pool(processes=4)

# x=pool.map(f,['bob','alice'])

# print(x[0])
# print(x[1])


libN=2
U_start=0.5
U_end=2*t
sections=(cycles-1)/libN
U_list=np.linspace(U_start,U_end,libN)
time = cycles



def setup(U_input):
    system=harmonic.hhg(cycles=cycles,delta=delta,libN=libN,field=field, nup=number, ndown=number, nx=nx, ny=0, U=U_input, t=t, F0=F0, a=a, bc='pbc')
    return system
systems=[]
for U in U_list:
    systems.append(setup(U))

N = int(cycles/(systems[0].freq*delta))+1

"""returns systems with both their last current and . First index is process number, second nu mber is either """
systems=pool.map(poolscripts.pump,systems)

#
# print("normalisation")
# print(np.dot(x[1][0],np.conj(x[1][0])))
# print(np.dot(x[0][0],np.conj(x[0][0])))
# print("currents")
# print(x[0][1])
# print(x[1][1])
print(systems[0].last_J)
print(systems[0].last_psi)

"""Now we track one of the systems, evolve it, then evolve the rest with the phi we obtain."""

def evolve_track(system,k):
    phi_original=system.phi
    inittime=1/system.freq +k*sections/system.freq
    def J_func(current_time):
        J=system.last_J*np.exp(-3*(current_time-inittime))
        return J

    h= hub.create_1e_ham(system,True)
    r = ode(evolve.integrate_f_track_J).set_integrator('zvode', method='bdf')
    r.set_initial_value(system.last_psi, inittime).set_f_params(system,h,J_func)
    branch = 0
    delta=system.delta
    while r.successful() and r.t < 1/system.freq+(k+1)*sections/system.freq:
        r.integrate(r.t + delta)
        psi_temp = r.y
        newtime = r.t
        # add to expectations
        neighbour.append(har_spec.nearest_neighbour_new(system, h, psi_temp))
        # two_body.append(har_spec.two_body_old(system, psi_temp))
        # tracking current
        phi_original.append(evolve.phi_J_track(system, newtime, J_func, neighbour[-1], psi_temp))
        harmonic.progress(N, int(newtime / delta))
    return phi_original

print(len(systems))
k=0
for k in tqdm(range(0,libN)):
    print('k=%s' % k)
    phi_original= evolve_track(systems[0],k)
    for j in range(0,len(systems)):
        if j>k:
            systems[j].phi=phi_original
            systems[j].iter+=1
    if len(systems)>1:
        systems=pool.map(poolscripts.evolve_others,systems[1:])
    print('number of systems= %s' % len(systems))
plt.plot(systems[0].phi)



plt.show()





#     prop=systems[k]
#     r = ode(evolve.integrate_f_discrim).set_integrator('zvode', method='bdf')
#     r.set_initial_value(psi_temp, 0).set_f_params(systems[0],h,phi_func)
#     branch = 0
#     delta=delta
#     while r.successful() and r.t < k*time/(libN*systems[0].freq):
#         oldpsi=psi_temp
#         r.integrate(r.t + delta)
#         psi_temp = r.y
#         newtime = r.t
#         # add to expectations
#
#         # double occupancy fails for anything other than half filling.
#         # D.append(evolve.DHP(prop,psi))
#         harmonic.progress(N, int(newtime / delta))
#         # psierror=evolve.f(systems[0],evolve.ham1(systems[0],h,newtime,time),oldpsi)
#         # diff = (psi_temp - oldpsi) / delta
#         # newerror = np.linalg.norm(diff + 1j * psierror)
#         # error.append(newerror)
#         neighbour.append(har_spec.nearest_neighbour_new(systems[0], h, psi_temp))
#         # neighbour_check.append(har_spec.nearest_neighbour(systems[0], psi_temp))
#         # X.append(observable.overlap(systems[0], psi_temp)[1])
#         phi_original.append(phi_func(newtime))
#         J_field.append(har_spec.J_expectation_track(systems[0], h, psi_temp, phi_original[-1]))
#         two_body.append(har_spec.two_body_old(systems[0], psi_temp))
#         D.append(observable.DHP(systems[0], psi_temp))
#
#
# last=J_field[-1]
# lastime=newtime
# def J_func(current_time):
#     J=last*np.exp(-2*(current_time-lastime))
#     return J
#
#
# r = ode(evolve.integrate_f_track_J).set_integrator('zvode', method='bdf')
# r.set_initial_value(psi_temp, newtime).set_f_params(systems[0],h,J_func)
# branch = 0
# delta=delta
# while r.successful() and r.t < time/(systems[0].freq):
#     oldpsi=psi_temp
#     r.integrate(r.t + delta)
#     psi_temp = r.y
#     newtime = r.t
#     # add to expectations
#     neighbour.append(har_spec.nearest_neighbour_new(systems[0], h, psi_temp))
#     two_body.append(har_spec.two_body_old(systems[0], psi_temp))
#
#     # tracking current
#     phi_original.append(evolve.phi_J_track(systems[0], newtime, J_func, neighbour[-1], psi_temp))
#
#     # tracking D
#     # phi_original.append(evolve.phi_D_track(lat,newtime,D_func,two_body[-1],psi_temp))
#
#     J_field.append(har_spec.J_expectation_track(systems[0], h, psi_temp, phi_original[-1]))
#     # double occupancy fails for anything other than half filling.
#     # D.append(evolve.DHP(prop,psi))
#     harmonic.progress(N, int(newtime / delta))

# np.save('./data/original/Jfield'+parameternames,J_field)
# np.save('./data/original/phi'+parameternames,phi_original)
# np.save('./data/original/discriminatorfield', phi_original)
# np.save('./data/original/phirecon'+parameternames,phi_reconstruct)
# np.save('./data/original/neighbour'+parameternames,neighbour)
# np.save('./data/original/energy'+parameternames,energy)
# np.save('./data/original/doublonenergy'+parameternames,doublon_energy)
# np.save('./data/original/singlonenergy'+parameternames,singlon)



# np.save('./data/original/position'+parameternames,X)



# plot_observables(systems[0], delta=0.02, time=5., K=.1)
# spectra(systems[0], initial=None, delta=delta, time=cycles, method='welch', min_spec=7, max_harm=50, gabor='fL')
