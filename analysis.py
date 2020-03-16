import matplotlib.pyplot as plt
import numpy as np
import os
import sys

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
libN = 3
cycles = libN + 1
U_start = 0.5 * t
U_end = 1.5 * t
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
J_fields_pumped = []
for U in U_list:
    systems.append(setup(U))

phi_discrim = np.load('./data/original/discriminatorfield' + parameternames)
phi_pumped = np.load('./data/original/pumpfield' + parameternames)

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
    J_fields_pumped.append(np.load('./data/discriminate/Jfieldpump' + Jparameternames))

    plt.plot(times, J_fields[-1], label='$\\frac{U}{t_0}=$%.2f' % (system.U))
plt.xlabel('(Time (cycles)')
plt.ylabel('J(t)')
plt.legend()
plt.show()

# plt.plot(phi_discrim)
# # plt.plot(phi_pumped)
# plt.axis('off')
# plt.show()
# J_comb=np.zeros(len(J_fields[0]))
# s = np.random.uniform(0.1, 1, libN)
# for j in range(0,3):
#     plt.plot(J_fields[j],color='red')
#     J_comb+= s[j]*J_fields[j]
#     plt.axis('off')
#     plt.show()
#
# plt.plot(J_comb,color='red')
# plt.axis('off')
# plt.show()
# fig,axs=plt.subplots(4)
# length=int(1*len(times)/4)
# xc=1
# axs[0].plot(times[:length],phi_discrim[:length])
# axs[0].axvline(x=xc, color='black', linestyle='dashed')
# axs[0].text(0.5, 4, 'Pump Pulse', fontsize=25,ha='center')
#
# # axs[0].axvline(x=xc+1, color='black', linestyle='dashed')
# # axs[0].text(1.5, 4, '$J_0=0$', fontsize=25,ha='center')
#
# # axs[0].axvline(x=xc+2, color='black', linestyle='dashed')
# # axs[0].text(2.5, 4, '$J_1=0$', fontsize=25,ha='center')
#
# # axs[0].text(3.5, 4, '$J_2=0$', fontsize=25,ha='center')
#
#
# axs[0].set( ylabel='$\\Phi(t)$',xlim=[0,cycles])
# for j in range(0,3):
#     axs[j+1].axvline(x=xc, color='black', linestyle='dashed')
#     # axs[j+1].axvline(x=xc+1, color='black', linestyle='dashed')
#     # axs[j+1].axvline(x=xc+2, color='black', linestyle='dashed')
#     axs[j+1].plot(times[:length],J_fields[j][:length],color='red')
#     axs[j+1].set(ylabel='$J_{%s}$' % j,xlim=[0,cycles])
#
# plt.xlabel('(Time (cycles)')
#
# plt.show()
discrim_currents = []

# s = np.array(np.random.rand(libN)+1)
s = np.random.uniform(0.1, 1, libN)
s = s / np.sum(s)
J_mat = np.zeros((libN, libN))
discrim_J_sum = np.zeros(libN)
combined_current = 0


def all_points_error(J_fields, rands):
    """Using all time points"""
    s = rands
    # s = np.random.uniform(0.1, 1, libN)
    # s = s / np.sum(s)
    combined_current = 0
    for k in range(libN):
        J_fields[k] = J_fields[k] ** 2
        # J_fields[k] = J_fields[k]
        combined_current += s[k] * J_fields[k]
    # inv_J=np.linalg.inv(J_mat)
    inv_J = np.transpose(np.linalg.pinv(J_fields))
    discrim_J_sum = np.array(combined_current)
    recalculated = np.dot(inv_J, discrim_J_sum)
    recalculated = np.dot(inv_J, discrim_J_sum)
    errors = np.linalg.norm((s - recalculated))/np.linalg.norm(s)
    # error_mean = 100 * np.mean(errors)
    # error_std = np.std(errors)
    return s, recalculated, errors


def averaged_error(J_fields):
    """averaging absolute values"""
    s = np.random.uniform(0.1, 1, libN)
    s = s / np.sum(s)
    J_mat = np.zeros((libN, libN))
    discrim_J_sum = np.zeros(libN)
    combined_current = 0
    for k in range(libN):
        # combined_current += s[k] * J_fields[k] ** 2
        combined_current += s[k] * J_fields[k]
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


def all_points_cond(J_fields, j, rands):
    """Using all time points"""
    s = rands
    # s = np.random.uniform(0.1, 1, j)
    # s = s / np.sum(s)
    combined_current = 0
    for k in range(j):
        J_fields[k] = J_fields[k] ** 2
        # J_fields[k] = J_fields[k]
        combined_current += s[k] * J_fields[k]
    # inv_J=np.linalg.inv(J_mat)
    condition = np.linalg.cond(J_fields)
    return condition


def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


s = np.random.uniform(0, 1, 2)
s = s / np.sum(s)
rands = s
#
# for k in range(len(s)):
#     # recalculated,s, error_mean, error_std=averaged_error(J_fields)
#     s,recalculated,  error_mean= all_points_error(J_fields,rands)
#     print('Actual concentration is %s, Optical Discrimination predicts a concentration of %s' % (s[k], recalculated[k]))
#
# print('mean error is %.2e %%' % (error_mean))
# print('error std is %.2e %%' % (error_std))
numbers = []
condition_av = []
pump_condition_av = []
pump_condition_all = []

condition_all = []
error_list = []
pump_error_list = []

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
for U_delta in [0, -1, -2, -3, -4, -5, -6, -8, -9, -10, -11, -12, -13, -14, -15]:
    U_start = 1 * t
    U_end = U_start + (10 ** U_delta) * t
    U_end = U_end
    print("calculating for delta %.2e " % (U_start - U_end))
    libN = 2
    cycles = 3
    U_list = np.linspace(U_start, U_end, 2)
    systems = []
    J_fields = []
    J_fields_pumped = []

    all_errors = []
    for U in U_list:
        blockPrint()
        systems.append(setup(U))
    for system in systems:
        Jparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-libN-%s-U_start-%s-U_end.npy' % (
            nx, cycles, system.U, t, number, delta, field, F0, 2, U_start, U_end)
        J_fields.append(np.load('./data/discriminate/Jfield' + Jparameternames))
        J_fields_pumped.append(np.load('./data/discriminate/Jfieldpump' + Jparameternames))

        print(len(J_fields))
    # recalculated,s, error_mean, error_std=averaged_error(J_fields)
    plt.plot(J_fields[0])
    plt.plot(J_fields[1])
    plt.show()
    s, recalculated, errors = all_points_error(J_fields, rands)
    error_list.append(errors)
    s, recalculated, errors = all_points_error(J_fields_pumped, rands)
    pump_error_list.append(errors)
    condition_av.append(averaged_cond(J_fields, 2))
    pump_condition_av.append(averaged_cond(J_fields_pumped, 2))

    condition_all.append(all_points_cond(J_fields, 2, rands))
    pump_condition_all.append(all_points_cond(J_fields_pumped, 2, rands))
    numbers.append(U_end - U_start)
    # all_errors.append(error_mean)
    enablePrint()

print(condition_all)
print(numbers)
plt.semilogx(numbers, condition_all, label='Optical Discrmination')
plt.semilogx(numbers, pump_condition_all, label='Naive approach')
plt.ylabel('Condition Number')
plt.xlabel('$\\Delta_U$')
plt.legend()
plt.show()

plt.subplot(211)
plt.semilogx(numbers, condition_all, label='Optical Discrmination', marker='x')
plt.semilogx(numbers, pump_condition_all, label='Naive Approach', marker='x')
plt.ylabel('${\\rm cond} (\\mathbf{A})$')
plt.legend()

plt.subplot(212)
# plt.semilogx(numbers,error_list,label='Optical Discrmination',linestyle='x-')
# plt.semilogx(numbers,pump_error_list,label='Naive Approach',linestyle='x-')
plt.loglog(numbers, error_list, label='Optical Discrmination', marker='x')
plt.loglog(numbers, pump_error_list, label='Naive Approach', marker='x')
plt.ylabel('$\\epsilon$')
plt.xlabel('$\\Delta_U$')

plt.show()
plt.show()
