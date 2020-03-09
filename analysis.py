import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import definition as harmonic


def iFT(A):
    """
    Inverse Fourier transform
    :param A:  1D numpy.array
    :return:
    """
    A = np.array(A)
    minus_one = (-1) ** np.arange(A.size)
    result = np.fft.ifft(minus_one * A)
    result *= minus_one
    result *= np.exp(1j * np.pi * A.size / 2)
    return result


def FT(A):
    """
    Fourier transform
    :param A:  1D numpy.array
    :return:
    """
    # test
    # print(A.size)
    A = np.array(A)
    k = A.size
    # A = np.pad(A, (0, 4 * k), 'constant')
    minus_one = (-1) ** np.arange(A.size)
    # result = np.fft.fft(minus_one * A)
    result = np.fft.fft(minus_one * A)
    # minus_one = (-1) ** np.arange(result.size)
    result *= minus_one
    result *= np.exp(-1j * np.pi * A.size / 2)
    # print(result.size)
    return result


# These are Parameters I'm using
# number=2
# nelec = (number, number)
# nx = 4®
# ny = 0
# t = 0.191
# U = 0.1 * t
# delta = 2
# cycles = 10
params = {
    'axes.labelsize': 30,
    # 'legend.fontsize': 28,
    'legend.fontsize': 23,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'figure.figsize': [2 * 3.375, 2 * 3.375],
    'text.usetex': True
}

plt.rcParams.update(params)
print(plt.rcParams.keys())
# Load parameters and data. 2 suffix is for loading in a different simulation for comparison
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
delta = 0.02
field = 32.9
# field=25
F0 = 10
a = 4
ascale = 1
libN = 10
cycles = libN + 1
U_start = 0 * t
U_end = 1 * t
sections = (cycles - 1) / libN
U_list = np.linspace(U_start, U_end, libN)
threads = libN
print("threads =%s" % threads)
os.environ["OMP_NUM_THREADS"] = "%s" % threads
# pool = Pool(processes=cpu_count())


parameternames = '-%s-nsites-%s-cycles-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-libN-%s-U_start-%s-U_end.npy' % (
    nx, cycles, t, number, delta, field, F0, libN, U_start, U_end)
phi_discrim = np.load('./data/original/discriminatorfield' + parameternames)
times = np.linspace(0, cycles, len(phi_discrim))

plt.plot(times, phi_discrim)
plt.xlabel('Time (cycles)')
plt.ylabel('$\\Phi(t)$')
plt.title('Discriminating field for 9 species')
plt.yticks(np.arange(-3 * np.pi, 3 * np.pi, 0.5 * np.pi),
           [r"$" + format(r / np.pi, ".2g") + r"\pi$" for r in np.arange(-3 * np.pi, 3 * np.pi, .5 * np.pi)])
plt.show()


def setup(U_input):
    system = harmonic.hhg(cycles=cycles, delta=delta, libN=libN, field=field, nup=number, ndown=number, nx=nx, ny=0,
                          U=U_input, t=t, F0=F0, a=a, bc='pbc', phi=phi_discrim)
    return system


systems = []
J_fields = []
J_fields_alt = []
for U in U_list:
    systems.append(setup(U))

phi_discrim = np.load('./data/original/discriminatorfield' + parameternames)
times = np.linspace(0, cycles, len(phi_discrim))

plt.subplot(211)
plt.plot(times, phi_discrim)
plt.ylabel('$\\Phi(t)$')
plt.title('Discriminating field for 9 species')
plt.yticks(np.arange(-3 * np.pi, 3 * np.pi, 0.5 * np.pi),
           [r"$" + format(r / np.pi, ".2g") + r"\pi$" for r in np.arange(-3 * np.pi, 3 * np.pi, .5 * np.pi)])
plt.subplot(212)
for system in systems:
    Jparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-libN-%s-U_start-%s-U_end.npy' % (
        nx, cycles, system.U, t, number, delta, field, F0, libN, U_start, U_end)
    J_fields.append(np.load('./data/discriminate/Jfield' + Jparameternames))

    plt.plot(times, J_fields[-1], label='$\\frac{U}{t_0}=$%.2f' % (system.U))
plt.xlabel('(Time (cycles)')
plt.ylabel('J(t)')
plt.legend()
plt.show()

discrim_currents = []

# s = np.array(np.random.rand(libN)+1)
s = np.random.uniform(0.1, 1, libN)
s = s / np.sum(s)
J_mat = np.zeros((libN, libN))
discrim_J_sum = np.zeros(libN)
combined_current = 0


def all_points_error(J_fields):
    """Using all time points"""
    s = np.random.uniform(0.1, 1, libN)
    s = s / np.sum(s)
    combined_current = 0
    for k in range(libN):
        J_fields[k] = J_fields[k] ** 2
        combined_current += s[k] * J_fields[k]
    # inv_J=np.linalg.inv(J_mat)
    inv_J = np.transpose(np.linalg.pinv(J_fields))
    discrim_J_sum = np.array(combined_current)
    recalculated = np.dot(inv_J, discrim_J_sum)
    recalculated = np.dot(inv_J, discrim_J_sum)
    errors = np.linalg.norm((s - recalculated))/np.linalg.norm(s)
    # error_mean = 100 * np.mean(errors)
    # error_std = np.std(errors)
    return recalculated,errors


def averaged_error(J_fields):
    """averaging absolute values"""
    s = np.random.uniform(0.1, 1, libN)
    s = s / np.sum(s)
    J_mat = np.zeros((libN, libN))
    discrim_J_sum = np.zeros(libN)
    combined_current = 0
    for k in range(libN):
        combined_current += s[k] * J_fields[k] ** 2
    sec_length = int(len(phi_discrim) / cycles)
    for j in range(libN):
        for k in range(libN):
            J_n = J_fields[j] ** 2
            sec_start = (int(1.1 * (k + 1) * sec_length))
            sec_end = int((k + 2) * sec_length)
            if j == k:
                discrim_J_sum[j] = np.sum(np.abs(combined_current[sec_start:sec_end]))
            else:
                sum_J = np.sum(J_n[sec_start:sec_end])
                J_mat[k, j] = sum_J
    # inv_J = np.linalg.inv(J_mat)
    inv_J = np.linalg.pinv(J_mat)
    # discrim_J_sum=np.array(combined_current)
    recalculated = np.dot(inv_J, discrim_J_sum)
    errors = 100 * np.abs(s - recalculated) / s
    error_mean = np.mean(errors)
    error_std = np.std(errors)
    return recalculated, s, error_mean, error_std


def averaged_cond(J_fields, j):
    """averaging absolute values"""
    s = np.random.uniform(0.1, 1, j)
    s = s / np.sum(s)
    J_mat = np.zeros((j, j))
    discrim_J_sum = np.zeros(j)
    combined_current = 0
    for k in range(j):
        combined_current += s[k] * J_fields[k] ** 2
    sec_length = int(len(phi_discrim) / cycles)
    for l in range(j):
        for k in range(j):
            J_n = J_fields[l] ** 2
            sec_start = (int(1.1 * (k + 1) * sec_length))
            sec_end = int((k + 2) * sec_length)
            if l == k:
                discrim_J_sum[l] = np.sum(np.abs(combined_current[sec_start:sec_end]))
            else:
                sum_J = np.sum(J_n[sec_start:sec_end])
                J_mat[k, l] = sum_J
    # inv_J = np.linalg.inv(J_mat)
    condition = np.linalg.cond(J_mat)

    return condition


def all_points_cond(J_fields, j):
    """Using all time points"""
    s = np.random.uniform(0.1, 1, j)
    s = s / np.sum(s)
    combined_current = 0
    for k in range(j):
        J_fields[k] = J_fields[k] ** 2
        combined_current += s[k] * J_fields[k]
    # inv_J=np.linalg.inv(J_mat)
    condition = np.linalg.cond(J_fields)
    return condition


def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


for k in range(len(s)):
    # recalculated,s, error_mean, error_std=averaged_error(J_fields)
    recalculated,  error_mean= all_points_error(J_fields)
    print('Actual concentration is %s, Optical Discrimination predicts a concentration of %s' % (s[k], recalculated[k]))

print('mean error is %.2e %%' % (error_mean))
# print('error std is %.2e %%' % (error_std))
numbers = []
condition_av = []
condition_all = []
error_list=[]

# for m in range(5, 11):
#     print("calculating for %s species" % (m))
#     cycles = m + 1
#     U_list = np.linspace(U_start, U_end, m)
#     systems = []
#     J_fields = []
#
#     all_errors = []
#     for U in U_list:
#         blockPrint()
#         systems.append(setup(U))
#     for system in systems:
#         Jparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-libN-%s-U_start-%s-U_end.npy' % (
#             nx, cycles, system.U, t, number, delta, field, F0, m, U_start, U_end)
#         J_fields.append(np.load('./data/discriminate/Jfield' + Jparameternames))
#         print(len(J_fields))
#     # recalculated,s, error_mean, error_std=averaged_error(J_fields)
#     # recalculated,s, error_mean, error_std=all_points_error(J_fields)
#     condition_av.append(averaged_cond(J_fields, m))
#     condition_all.append(all_points_cond(J_fields, m))
#     numbers.append(m)
#     # all_errors.append(error_mean)
#     enablePrint()
#     print(m)
#
# print(numbers)
# print(condition_av)
# print(condition_all)
#
# plt.plot(numbers, condition_all)
# plt.ylabel('Condition Number')
# plt.xlabel('Number of Species')
# plt.show()


U_start=1*t
for U_end in [2,1.5,1.1,1.05,1.01,1.005,1.001,1.0005]:
    U_end=U_end*t
    print("calculating for delta %.2e " % (U_start-U_end))
    libN=2
    cycles = 3
    U_list = np.linspace(U_start, U_end, 2)
    systems = []
    J_fields = []

    all_errors = []
    for U in U_list:
        blockPrint()
        systems.append(setup(U))
    for system in systems:
        Jparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-libN-%s-U_start-%s-U_end.npy' % (
            nx, cycles, system.U, t, number, delta, field, F0, 2, U_start, U_end)
        J_fields.append(np.load('./data/discriminate/Jfield' + Jparameternames))
        print(len(J_fields))
    # recalculated,s, error_mean, error_std=averaged_error(J_fields)
    recalculated,errors=all_points_error(J_fields)
    error_list.append(errors)
    condition_av.append(averaged_cond(J_fields, 2))
    condition_all.append(all_points_cond(J_fields, 2))
    numbers.append(U_end-U_start)
    # all_errors.append(error_mean)
    enablePrint()

print(condition_all)
print(numbers)
plt.semilogx(numbers, condition_all)
plt.ylabel('Condition Number')
plt.xlabel('Number of Species')
plt.show()

plt.semilogx(numbers,error_list)
plt.show()