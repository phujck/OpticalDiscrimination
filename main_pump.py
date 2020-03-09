import os

libN = 2
threads = libN
print("threads =%s" % threads)
os.environ["OMP_NUM_THREADS"] = "%s" % threads
# pool = Pool(processes=cpu_count())
import time
from multiprocessing import get_context

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode

import definition as harmonic
# This contains the stuff needed to calculate some expectations. Generally contains stuff
# that applies operators to the wave function
import evolve as evolve
# These also contain various important observable calculators
import harmonic as har_spec
# Contains lots of important functions.
# Sets up the lattice for the system
import hub_lats as hub
import poolscripts

# input units: THz (field), eV (t, U), MV/cm (peak amplitude, F0), Angstroms (systems[0]tice cst, a)
# they're then converted to t-normalised atomic units. bc='pbc' for periodic and 'abc' for antiperiodic

if __name__ == '__main__':
    neighbour = []
    neighbour_check = []
    energy = []
    doublon_energy = []
    phi_original = []
    phi_reconstruct = [0., 0.]
    boundary_1 = []
    boundary_2 = []
    two_body = []
    two_body_old = []
    error = []
    D = []
    X = []
    singlon = []
    systems = []
    number = 3
    nelec = (number, number)
    nx = 6
    ny = 0
    t = 0.52
    # t=1.91
    # t=1
    U = 1 * t
    delta = 0.02
    cycles = libN + 1
    field = 32.9
    # field=32.9*2
    F0 = 10
    a = 4

    U_start = 1 * t
    U_end = 1.5 * t
    sections = (cycles - 1) / libN
    U_list = np.linspace(U_start, U_end, libN)


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


    # x=pool.map(f,['bob','alice'])

    # print(x[0])
    # print(x[1])

    pool = get_context("spawn").Pool(processes=threads)
    parameternames = '-%s-nsites-%s-cycles--%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-libN-%s-U_start-%s-U_end' % (
        nx, cycles, t, number, delta, field, F0, libN, U_start, U_end)


    def setup(U_input):
        system = harmonic.hhg(cycles=cycles, delta=delta, libN=libN, field=field, nup=number, ndown=number, nx=nx, ny=0,
                              U=U_input, t=t, F0=F0, a=a, bc='pbc')
        return system


    systems = []
    for U in U_list:
        systems.append(setup(U))

    N = int(cycles / (systems[0].freq * delta))
    N_sec = int(1 / (systems[0].freq * delta))
    secN = int(N / (libN + 1))

    """returns systems with both their last current and . First index is process number, second number is either """
    parallel_start = time.time()

    """Adaptive timestep method"""
    systems = pool.map(poolscripts.long_pump, systems)
    print("pump done")

    parallel_end = time.time()
    phi_original = systems[0].phi
    plt.plot(phi_original)
    plt.show()
    for k in range(0, libN):
        plt.plot(systems[k].last_J, '*-')
    plt.show()
    for k in range(0, libN):
        plt.plot(systems[k].neighbourexpec, '*-')
    plt.show()
    for k in range(0, libN):
        ne = systems[k].neighbourexpec
        mag = np.abs(ne)
        theta = np.angle(ne)
        plt.plot(systems[k].last_J, '*-')
        plt.plot(-2 * systems[k].a * systems[k].t * mag * np.sin(phi_original - theta))
    plt.show()
    for system in systems:
        parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-libN-%s-U_start-%s-U_end' % (
            nx, cycles, system.U, t, number, delta, field, F0, libN, U_start, U_end)
        np.save('./data/discriminate/Jfieldpump' + parameternames, system.last_J)

    print("Time for using multiprocessing pool: %s seconds" % (parallel_end - parallel_start))

    parameternames = '-%s-nsites-%s-cycles-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-libN-%s-U_start-%s-U_end' % (
        nx, cycles, t, number, delta, field, F0, libN, U_start, U_end)
    np.save('./data/original/pumpfield' + parameternames, phi_original)

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
