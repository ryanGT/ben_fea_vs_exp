from matplotlib.pyplot import *
from scipy import *
from scipy import optimize

from numpy.linalg import inv, eig

import rwkos, rwkbode
reload(rwkbode)

import txt_data_processing, bode_utils
import control
from rwkdataproc import thresh

import sys

import copy

from IPython.core.debugger import Pdb


def A_mat_from_M_K_C(M,K,C):
    nr, nc = M.shape
    zm = zeros_like(M)
    Mi = inv(M)
    A_ss_part1 = column_stack([zm, eye(nc)])
    A_ss_part2 = column_stack([-dot(Mi,K), -dot(Mi,C)])
    A_ss = row_stack([A_ss_part1, A_ss_part2])
    return A_ss


# Load Exp. Data
# --------------------

expfile = txt_data_processing.Data_File('swept_sine_kp_1_amp_50.txt')
expfile.bode_plot('v','theta',seedfreq=2.45, seedphase=-200.0, label='exp')
expfile.bode_plot('v','a',fignum=2, seedfreq=10.0, seedphase=-150.0, label='exp')

# Logarythmically downsample to get ready for curve fitting

f_exp = expfile.freq
bode0 = expfile.calc_bode('v','theta')
bode1 = expfile.calc_bode('v','a')

freqs = [0.7, 1.75, 3.25, 5.0, 15.0, 22.0, 30.0]
Ns = [15.0, 40.0, 15.0, 10.0, 70.0, 10.0]

ds_inds0, f_ds0, bode0_ds = bode0.log_downsample(f_exp, freqs, Ns)
ds_inds1, f_ds1, bode1_ds = bode1.log_downsample(f_exp, freqs, Ns)

f_ds = f_ds0

bode0_ds.seedfreq = 2.2
bode0_ds.seedphase = -180.0
bode0_ds.PhaseMassage(f_ds)

bode1_ds.seedfreq = 17.0
bode1_ds.seedphase = -180.0
bode1_ds.PhaseMassage(f_ds)


bode_utils.bode_plot3(f_ds, bode0_ds, fmt='ro', clear=False, \
                      label='log ds')
bode_utils.bode_plot3(f_ds, bode1_ds, fmt='ro', clear=False, \
                      label='log ds', fignum=2)

#setup the vectors we will use for curve fitting
dB_theta_ds = bode0_ds.dBmag()
phase_theta_ds = bode0_ds.phase

dB_accel_ds = bode1_ds.dBmag()
phase_accel_ds = bode1_ds.phase



# Load Beam Parameters
# -------------------------

pklpath = 'model_w_bm_opt.pkl'
import rwkmisc, misc_utils
import pylab_util as PU
mydict = rwkmisc.LoadPickle(pklpath)
tmm_dict = copy.copy(mydict)

beam_params = {}
beam_keys = ['mu','L','EI']
for key in beam_keys:
    beam_params[key] = mydict[key]

# FEA Model
# -------------------------

mu = beam_params['mu']
L = beam_params['L']
EI = beam_params['EI']

n_e = 4#number of elements

h = float(L)/n_e

m_e = mu*h/420.0*array([[156.0, 22, 54, -13], \
                        [22.0, 4, 13, -3], \
                        [54.0, 13, 156, -22], \
                        [-13, -3, -22, 4]])

k_scale = EI/(h**3)
k_e = array([[12.0, 6, -12, 6], \
             [6, 4, -6, 2], \
             [-12, -6, 12, -6], \
             [6, 2, -6, 4]])*k_scale

nr = 2*n_e + 2

M = zeros((nr,nr))
K = zeros((nr,nr))

for i in range(n_e):
    start_index = 2*i
    M[start_index:start_index+4, start_index:start_index+4] += m_e
    K[start_index:start_index+4, start_index:start_index+4] += k_e



# Apply Boundary Conditions
# ---------------------------------

# The beam will be pinned at the base, so drop the first row and
# column related to lateral displacement

M11 = M[1:,1:]
K11 = K[1:,1:]


def FEA_ss_model(J_motor=0.010, J_bm=0.0100, \
                 k_wall=100.0, k_clamp=1000.0, \
                 c_motor=5.0, c_clamp=10.0, \
                 c_wall=1.0, f_gain=0.05, \
                 m_accel=0.0, J_accel=0.0, \
                 ):

    nr1, nc1 = M11.shape
    N_bm = nr1+2

    M_bm = zeros((N_bm, N_bm))


    M_bm[2:,2:] = M11
    M_bm[0,0] = J_motor#not sure about units here
    M_bm[1,1] = J_bm

    M_bm[-2,-2] += m_accel
    M_bm[-1,-1] += J_accel


    K_bm = zeros((N_bm, N_bm))
    K_bm[2:,2:] = K11

    k_wall_mat = array([[k_wall, -k_wall],[-k_wall, k_wall]])
    K_bm[0:2,0:2] += k_wall_mat

    k_clamp_mat = array([[k_clamp, -k_clamp],[-k_clamp, k_clamp]])

    K_bm[1:3,1:3] += k_clamp_mat

    C_bm = zeros((N_bm, N_bm))
    c_wall_mat = array([[c_wall, -c_wall],[-c_wall, c_wall]])
    C_bm[0:2,0:2] += c_wall_mat

    c_clamp_mat = array([[c_clamp, -c_clamp],[-c_clamp, c_clamp]])

    C_bm[1:3,1:3] += c_clamp_mat

    C_bm[0,0] += c_motor


    A_bm = A_mat_from_M_K_C(M_bm,K_bm,C_bm)
    Mi_bm = inv(M_bm)
    nr_bm, nc_bm = M_bm.shape
    zv = zeros((nr_bm,1))
    F_bm = zeros((nr_bm,1))
    F_bm[0] = f_gain
    B_bm = row_stack([zeros((nr_bm,1)), dot(Mi_bm,F_bm)])
    y_row = zeros((1,nr_bm))

    counts_out = True
    y_theta = 1.0/h*180.0/pi
    if counts_out:
        y_theta *= 1024.0/360
    y_row[0,1] = y_theta

    C_bm_1 = column_stack([y_row, zeros((1,nr_bm))])

    C_bm_2 = copy.copy(A_bm[-2,:])
    a_gain = 1.1452654922936827
    use_a_gain = 1
    if use_a_gain:
        C_bm_2 *= a_gain
    C_bm_ss = row_stack([C_bm_1, C_bm_2])
    D_bm = zeros((2,1))
    D_bm[-1] = B_bm[-2]
    sys_bm = control.ss(A_bm, B_bm, C_bm_ss, D_bm)

    return sys_bm


def bode_from_FEA_sys(sys, freq=None):
    if freq is None:
        myom = om
    else:
        myom = 2.0*pi*freq


    mag, phase, omega = sys.freqresp(myom)
    dB = squeeze(20.0*log10(mag))
    phase = squeeze(phase)*180.0/pi

    phase2 = unwrap(phase*pi/180.0)*180.0/pi

    dB_theta = dB[0,:]
    phase_theta = phase2[0,:]
    dB_accel = dB[1,:]
    phase_accel = phase2[1,:]

    return dB_theta, phase_theta, dB_accel, phase_accel


def run_model_and_plot(label=None, freq=None, **kwargs):
    my_dict = build_params_dict(**kwargs)
    sys = FEA_ss_model(**my_dict)
    dB_theta, phase_theta, dB_accel, phase_accel = bode_from_FEA_sys(sys, \
                                                                     freq=freq)

    if freq is None:
        freq = def_freq

    bode_utils.bode_plot(freq, dB_theta, phase_theta, \
                         fignum=1, clear=False, \
                         label=label)


    bode_utils.bode_plot(freq, dB_accel, phase_accel, \
                         fignum=2, clear=False, \
                         label=label)


myfreqlim = [0.01,100.0]

def my_limits():
    PU.SetMagLim(1,[-60,50])
    PU.SetPhaseLim(1,[-360,100])
    PU.SetFreqLim(1,myfreqlim)
    PU.SetMagLim(2,[-40,30])
    PU.SetFreqLim(2,myfreqlim)
    PU.SetPhaseLim(2,[-600,300])
    PU.SetLegend(1, loc=2)
    PU.SetLegend(2, loc=2)


def my_save(note):
    PU.mysave('theta_bode_%s.png' % note, 1)
    PU.mysave('accel_bode_%s.png' % note, 2)



def build_params_dict(**kwargs):
    def_params = {'J_motor':0.10, \
                  'J_bm':0.0100, \
                  'k_wall':130.0, \
                  'k_clamp':200.0, \
                  'c_motor':4.0, \
                  'c_clamp':0.5, \
                  'c_wall':0.3, \
                  'f_gain':0.05, \
                  'm_accel':0.0061261022979705367, \
                  'J_accel':0.001, \
                  }
    my_params = copy.copy(def_params)
    my_params.update(kwargs)
    return my_params



def get_ig_vect_from_param_list(param_list):
    """Given a list of parameters you want to treat as unknown, get
    the corresponding initial guesses from the default params dict."""
    def_dict = build_params_dict()
    vals = []

    for key in param_list:
        cur_val = def_dict[key]
        vals.append(cur_val)

    return vals


def _get_f(fvect=None):
    if fvect is None:
        fvect = f_ds
    return fvect


def get_sys_from_params(C, unknown_params):
    kwargs = dict(zip(unknown_params, C))
    my_dict = build_params_dict(**kwargs)
    sys = FEA_ss_model(**my_dict)
    return sys


def mymodel(C, unknown_params, fvect=None):
    """This will be the underlying model file to be used with a
    similar cost function to be passed to optimize.fmin.  The list of
    unknown_params will be passed to optimize.fmin as the first arg in
    args.  This list will be used with C to build the dictionary of
    parameters that are different from the defaults."""
    sys = get_sys_from_params(C, unknown_params)    
    fvect = _get_f(fvect=fvect)


    dB_theta, phase_theta, dB_accel, phase_accel = bode_from_FEA_sys(sys, \
                                                                     freq=fvect)
    return dB_theta, phase_theta, dB_accel, phase_accel


def plot_model(C, unknown_params, fvect=None, fmt='k-', \
               clear=False, label=None, fi1=1):
    dB_theta, phase_theta, dB_accel, phase_accel = mymodel(C, \
                                                           unknown_params, \
                                                           fvect=fvect)
    fvect = _get_f(fvect=fvect)

    bode_utils.bode_plot(fvect, dB_theta, phase_theta, clear=clear, \
                         fmt=fmt, fignum=fi1, label=label)
    bode_utils.bode_plot(fvect, dB_accel, phase_accel, clear=clear, \
                         fmt=fmt, fignum=fi1+1, label=label)



def plot_from_dict(dict_in, **kwargs):
    vals = dict_in.values()
    keys = dict_in.keys()
    plot_model(vals, keys, **kwargs)
    

def plot_guesses(keys, factors, **kwargs):
    mydict = build_params_dict()
    for key, factor in zip(keys, factors):
        mydict[key] *= factor

    plot_from_dict(mydict, **kwargs)

    return mydict


def mycost0(C, unknown_params):
    fvect = None
    dB_theta, phase_theta, dB_accel, phase_accel = mymodel(C, \
                                                           unknown_params, \
                                                           fvect=fvect)
    dB_theta_error = dB_theta_ds - dB_theta
    dB_accel_error = dB_accel_ds - dB_accel
    e_dB_theta = (dB_theta_error**2).sum()
    e_dB_accel = (dB_accel_error**2).sum()
    sys = get_sys_from_params(C, unknown_params)
    unstable_vect = real(sys.pole()) > 0.0
    penalty = 0.0
    if unstable_vect.any():
        penalty += 1.0e6
    return e_dB_theta + e_dB_accel + penalty


