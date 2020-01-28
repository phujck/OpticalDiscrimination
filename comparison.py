import os
import time
from multiprocessing import get_context, cpu_count

import numpy as np
from tqdm import tqdm

import definition as harmonic
import poolscripts

# This contains the stuff needed to calculate some expectations. Generally contains stuff
# that applies operators to the wave function

# Contains lots of important functions.
# Sets up the lattice for the system

# These also contain various important observable calculators

# input units: THz (field), eV (t, U), MV/cm (peak amplitude, F0), Angstroms (systems[0]tice cst, a)
# they're then converted to t-normalised atomic units. bc='pbc' for periodic and 'abc' for antiperiodic


if __name__ == '__main__':
    number = 3
    nelec = (number, number)
    nx = 6
    ny = 0
    t = 0.52
    # t=1.91
    # t=1
    U = 1 * t
    delta = 0.05
    cycles = 0.1
    field = 32.9
    # field=32.9*0.5
    F0 = 10
    a = 4

    libN = 100
    U_start = 1
    U_end = 2 * t
    sections = (cycles - 1) / libN
    U_list = np.linspace(U_start, U_end, libN)

    # threads = libN
    threads = 1

    print("threads =%s" % threads)
    os.environ["OMP_NUM_THREADS"] = "%s" % threads
    print(cpu_count())


    # pool = Pool(3)

    def setup(U_input):
        system = harmonic.hhg(cycles=cycles, delta=delta, libN=libN, field=field, nup=number, ndown=number, nx=nx, ny=0,
                              U=U_input, t=t, F0=F0, a=a, bc='pbc')
        return system


    systems = []
    for U in U_list:
        systems.append(setup(U))
    systems2 = systems.copy()

    N = int(cycles / (systems[0].freq * delta)) + 1
    serial_start = time.time()
    for k in tqdm(range(0, libN)):
        systems[k] = poolscripts.pump(systems[k])
    serial_end = time.time()

    print(type(serial_start))
    print(type(serial_end))

    print("Time for using serial pump: %.2f seconds" % (serial_end - serial_start))

    """returns systems with both their last current and . First index is process number, second number is either """
    parallel_start = time.time()
    # pool = get_context("spawn").Pool(processes=libN)
    pool = get_context("spawn").Pool(processes=cpu_count())
    systems2 = pool.map(poolscripts.pump, systems2)
    pool.close()

    # output=Queue()
    # # Setup a list of processes that we want to run
    # processes = [Process(target=poolscripts.process_pump, args=(systems2[x], output,x)) for x in range(0,libN)]
    #
    #
    # # Run processes
    # k=0
    # for p in processes:
    #     os.system("taskset -p -c %d %d" % (k % cpu_count(), os.getpid()))
    #     p.start()
    #     k+=1
    #
    # print("processes done")
    # # Exit the completed processes
    # for p in processes:
    #     p.join()
    #     if p.is_alive():
    #         print("Job is not finished!")
    # print("ended processes")
    # # Get process results from the output queue
    # results = [output.get() for p in processes]
    # results.sort()
    # systems2 = [r[1] for r in results]
    # print(results)
    parallel_end = time.time()
    print("parallel done!")

    print("Time for using multiprocessing pool: %s seconds" % (parallel_end - parallel_start))
    print("Time for using serial processing: %s seconds" % (serial_end - serial_start))

    for j in range(0, len(systems)):
        print("difference between last J in serial vs parallel for system %s is %s" % (
        j, (systems[j].last_J - systems2[j].last_J)))
