import os

libN = 3
threads = libN
print("threads =%s" % threads)
os.environ["OMP_NUM_THREADS"] = "%s" % threads
# pool = Pool(processes=cpu_count())
import numpy as np

import definition as harmonic
import matplotlib.pyplot as plt
from multiprocessing import get_context
import poolscripts

# NOTE: time is inputted and plotted in terms of cycles, but the actual propagation happens in 'normal' time


# input units: THz (field), eV (t, U), MV/cm (peak amplitude, F0), Angstroms (lattice cst, a)
# they're then converted to t-normalised atomic units. bc='pbc' for periodic and 'abc' for antiperiodic

if __name__ == '__main__':
    neighbour = []
    phi_original = []
    phi_reconstruct = [0., 0.]
    boundary_1 = []
    boundary_2 = []
    two_body = []
    two_body_old = []
    error = []
    J_field = []
    D_cutfreq = []
    Jalt = []
    systems = []

    number = 3
    nelec = (number, number)
    nx = 6
    ny = 0
    t = 0.52
    # t=1.91
    # t=1
    U = 1 * t
    cutoff = 40
    delta = 0.005
    cycles = libN + 1
    field = 32.9
    # field=32.9*2
    F0 = 10
    a = 4
    ascale = 1
    U_start = 0 * t
    U_end = 1 * t
    sections = (cycles - 1) / libN
    U_list = np.linspace(U_start, U_end, libN)
    print(U_list)

    pool = get_context("spawn").Pool(processes=threads)

    parameternames = '-%s-nsites-%s-cycles-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-libN-%s-U_start-%s-U_end.npy' % (
        nx, cycles, t, number, delta, field, F0, libN, U_start, U_end)
    phi_discrim = np.load('./data/original/discriminatorfield' + parameternames)

    # for j in range(1,len(phi_discrim)):
    #     if phi_discrim[j]-phi_discrim[j-1] > 1.8*np.pi:
    #         phi_discrim[j:] = phi_discrim[j:]-2*np.pi
    #     elif phi_discrim[j]-phi_discrim[j-1] < -1.8*np.pi:
    #         phi_discrim[j:] = phi_discrim[j:]+2*np.pi
    plt.plot(phi_discrim)
    plt.show()


    def setup(U_input):
        system = harmonic.hhg(cycles=cycles, delta=delta, libN=libN, field=field, nup=number, ndown=number, nx=nx, ny=0,
                              U=U_input, t=t, F0=F0, a=a, bc='pbc', phi=phi_discrim)
        return system


    systems = []
    for U in U_list:
        print("U value is %.2f" % U)
        systems.append(setup(U))

    systems = pool.map(poolscripts.evolve_discrimate, systems)

    for j in range(len(systems)):
        plt.plot(systems[j].last_J)
    plt.show()

    plt.plot(phi_discrim)
    plt.plot(systems[0].phi)
    plt.show()

    for system in systems:
        parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-libN-%s-U_start-%s-U_end' % (
            nx, cycles, system.U, t, number, delta, field, F0, libN, U_start, U_end)
        np.save('./data/discriminate/Jfield_discrim' + parameternames, system.last_J)
