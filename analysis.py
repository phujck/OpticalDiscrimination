import numpy as np
import matplotlib.pyplot as plt
import definition as hams
import numpy.ma as ma
from matplotlib import cm as cm
from scipy.signal import blackman
from scipy.signal import stft
import harmonic as har_spec
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


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


def smoothing(A, b=1, c=5, d=0):
    if b == 1:
        b = int(A.size / 50)
    if b % 2 == 0:
        b = b + 1
    j = savgol_filter(A, b, c, deriv=d)
    return j


def current(sys, phi, neighbour):
    conjugator = np.exp(-1j * phi) * neighbour
    c = sys.a * sys.t * 2 * np.imag(conjugator)
    return c


Tracking = False
Track_Branch = False


def plot_spectra(U, w, spec, min_spec, max_harm):
    # spec = np.log10(spec)
    xlines = [2 * i - 1 for i in range(1, 6)]
    for i, j in enumerate(U):
        plt.semilogy(w, spec[:, i], label='$\\frac{U}{t_0}=$ %.1f' % (j))
        axes = plt.gca()
        axes.set_xlim([0, max_harm])
        axes.set_ylim([10 ** (-15), spec.max()])
    for xc in xlines:
        plt.axvline(x=xc, color='black', linestyle='dashed')
        plt.xlabel('Harmonic Order')
        plt.ylabel('HHG spectra')
    plt.legend(loc='upper right')
    plt.show()


def plot_spectra_switch(U, w, spec, min_spec, max_harm):
    spec = np.log10(spec)
    xlines = [2 * i - 1 for i in range(1, 6)]
    for i, j in enumerate(U):
        plt.plot(w, spec[:, i], label='%s' % (j))
        axes = plt.gca()
        axes.set_xlim([0, max_harm])
        axes.set_ylim([-min_spec, spec.max()])
    for xc in xlines:
        plt.axvline(x=xc, color='black', linestyle='dashed')
        plt.xlabel('Harmonic Order')
        plt.ylabel('HHG spectra')
    plt.legend(loc='upper right')
    plt.show()


def plot_spectra_track(U, w, spec, min_spec, max_harm):
    # spec = np.log10(spec)
    xlines = [2 * i - 1 for i in range(1, 6)]
    for i, j in enumerate(U):
        print(i)
        print(i % 2)
        if i < 2:
            plt.semilogy(w, spec[:, i], label='%s' % (j))
        else:
            plt.semilogy(w, spec[:, i], linestyle='dashed', label='%s' % (j))
        axes = plt.gca()
        axes.set_xlim([0, max_harm])
        axes.set_ylim([10 ** (-min_spec), spec.max()])
    for xc in xlines:
        plt.axvline(x=xc, color='black', linestyle='dashed')
        plt.xlabel('Harmonic Order')
        plt.ylabel('HHG spectra')
    plt.legend(loc='upper right')
    plt.show()


def plot_spectrogram(t, w, spec, min_spec=11, max_harm=60):
    w = w[w <= max_harm]
    t, w = np.meshgrid(t, w)
    spec = np.log10(spec[:len(w)])
    specn = ma.masked_where(spec < -min_spec, spec)
    cm.RdYlBu_r.set_bad(color='white', alpha=None)
    plt.pcolormesh(t, w, specn, cmap='RdYlBu_r')
    plt.colorbar()
    plt.xlabel('Time [Cycles]')
    plt.ylabel('Harmonic Order')
    plt.title('Time-Resolved Emission')
    plt.show()


def FT_count(N):
    if N % 2 == 0:
        return int(1 + N / 2)
    else:
        return int((N + 1) / 2)


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
number = 3
number2 = number
nelec = (number, number)
nx = 6
nx2 = nx
ny = 0
t = 0.52
t1 = t
t2 = 0.52
U = 0.2 * t
U2 = 1 * t
delta = 0.01
delta2 = 0.01
cycles = 2
cycles2 = 2
# field= 32.9
field = 32.9
field2 = 32.9
F0 = 10
a = 4
scalefactor = 1
scalefactor2 = 1
ascale = 1
ascale2 = 1
Jscale = 1
libN=3

prop = hams.hhg(cycles=cycles,delta=delta,libN=libN,field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')
prop2 = hams.hhg(cycles=cycles,delta=delta,libN=libN,field=field, nup=number, ndown=number, nx=nx, ny=0, U=U2, t=t2, F0=F0, a=a, bc='pbc')

# load files
parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
    nx, cycles, U, t, number, delta, field, F0)
newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
    nx, cycles, U, t, number, delta, field, F0, ascale)
J_field = np.load('./data/original/Jfield' + parameternames)
phi_original = np.load('./data/original/phi' + parameternames)
phi_reconstruct = np.load('./data/original/phirecon' + parameternames)
neighbour = np.load('./data/original/neighbour' + parameternames)
# neighbour_check = np.load('./data/original/neighbour_check' + parameternames)
# energy = np.load('./data/original/energy' + parameternames)
# doublon_energy = np.load('./data/original/doublonenergy' + parameternames)
# doublon_energy_L = np.load('./data/original/doublonenergy2' + parameternames)
# singlon_energy = np.load('./data/original/singlonenergy' + parameternames)


parameternames2 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
    nx2, cycles2, U2, t2, number2, delta2, field2, F0)
newparameternames2 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
    nx2, cycles2, U2, t2, number2, delta2, field2, F0, ascale2)
J_field2 = np.load('./data/original/Jfield' + parameternames2)
# two_body2 = np.load('./data/original/twobody' + parameternames2)
neighbour2 = np.load('./data/original/neighbour' + parameternames2)
phi_original2 = np.load('./data/original/phi' + parameternames2)
# energy2 = np.load('./data/original/energy' + parameternames2)
# doublon_energy2 = np.load('./data/original/doublonenergy' + parameternames2)
# doublon_energy_L2 = np.load('./data/original/doublonenergy2' + parameternames2)
# singlon_energy2 = np.load('./data/original/singlonenergy' + parameternames2)
# error2 = np.load('./data/original/error' + parameternames2)

times = np.linspace(0, cycles, len(J_field))
times2 = np.linspace(0, cycles2, len(J_field2))
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
ax1.plot(times, J_field, label='$\\frac{U}{t_0}=0$')
ax2.plot(times2, J_field2, label='$\\frac{U}{t_0}=7$')
ax1.set_ylabel('$\\frac{{\\rm d}J}{{\\rm d}t}$')
ax2.set_ylabel('$\\frac{{\\rm d}J}{{\\rm d}t}$')

plt.xlabel('Time [cycles]')
plt.show()

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
ax1.plot(times, phi_original, label='$\\frac{U}{t_0}=0$')
ax2.plot(times2, phi_original2, label='$\\frac{U}{t_0}=7$')
ax1.set_ylabel('$\\frac{{\\rm d}J}{{\\rm d}t}$')
ax2.set_ylabel('$\\frac{{\\rm d}J}{{\\rm d}t}$')

plt.xlabel('Time [cycles]')
plt.show()

plt.plot(times, phi_original)
plt.plot(times2, phi_original2, linestyle='dashed')
plt.show()

plt.plot(times, J_field)
plt.plot(times2, J_field2, linestyle='dashed')
plt.show()

discrim_currents = []
for k in [0.9, 1]:
    U = k * t
    parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
        nx, cycles, U, t, number, delta, field, F0)
    J_field_discrim = np.load('./data/discriminate/Jfield' + parameternames)
    discrim_currents.append(J_field_discrim)
    plt.plot(times, discrim_currents[-1])
plt.show()

s = np.random.rand(2)
s = s / np.sum(s)

print(s)
combined_current = 0
for k in [0, 1]:
    combined_current += s[k] * discrim_currents[k]
plt.plot(times, combined_current)
plt.show()

cut = int(3 * len(discrim_currents[0]) / 4) + 10

sig = np.sum(np.abs(discrim_currents[0][cut:-10]))
print(sig)

combined_sum = np.sum(np.abs(combined_current[cut:-10]))

concentration = combined_sum / sig

print('Actual concentration is %s, Optical Discrimination predicts a concentration of %s' % (s[0], concentration))

plt.plot(times[:-5], combined_current[:-5], label='True Current')
plt.plot(times[:-5], concentration * discrim_currents[0][:-5] + (1 - concentration) * discrim_currents[1][:-5],
         linestyle='dashed', label='ODD reconstruction')
plt.ylabel('$J(t)$')
plt.xlabel('Time')
plt.legend()
plt.show()
