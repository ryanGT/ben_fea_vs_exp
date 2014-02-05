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

## util_path = rwkos.FindFullPath('siue/Research/work/2013/sabbatical_summer_fall_2013/DTTMM_vs_FEA_vs_Lagrange_TSDs')

## if util_path not in sys.path:
##     sys.path.append(util_path)

## import modeling_comparison_utils
## reload(modeling_comparison_utils)


# Load Exp. Data
# --------------------

myfile = txt_data_processing.Data_File('swept_sine_kp_1_amp_50.txt')
myfile.bode_plot('v','theta',seedfreq=2.45, seedphase=-200.0, label='exp')
myfile.bode_plot('v','a',fignum=2, seedfreq=10.0, seedphase=-150.0, label='exp')


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

k_wall = mydict['k_clamp']
c_wall = mydict['c_clamp']

n_e = 3#number of elements

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

C11 = zeros_like(K11)
C11[0,0] = (c_wall/(h**2))*0.010

M11_i = inv(M11)

nr2, nc2 = K11.shape

F11 = zeros((nr2,1))
#F11[-2] = 1.0#pure translation forcing (no bending moment) at the end of the beam
F11[0] = 100.0#torque at base

A_part1 = column_stack([zeros((nr2,nr2)), eye(nr2)])
A_part2 = column_stack([-dot(M11_i, K11), -dot(M11_i, C11)])

A = row_stack([A_part1, A_part2])
B = row_stack([zeros((nr2,1)), dot(M11_i,F11)])
y_row = zeros((1,nr2))
#y_row[0,-2] = 1.0#lateral displacement of the tip#<--- need to fix the
    #output for theta
y_row[0,0] = 1.0
C = column_stack([y_row, zeros((1,nr2))])

full_order_sys = control.ss(A, B, C, 0)


# Bode Analysis
# -------------------------

freq = logspace(-3, 2, 1000)
def_freq = freq

om = 2*pi*freq

run_simpler_model = 0

if run_simpler_model:
    mag_full_order, phase_full_order, omega_full_order = full_order_sys.freqresp(om)
    dB_full_order = squeeze(20.0*log10(mag_full_order))
    phase_full_order = squeeze(phase_full_order)*180.0/pi

    bode_utils.bode_plot(freq, dB_full_order, phase_full_order, \
                         fignum=1, clear=False)



# Model with Base Mass and Motor Dynamics
# -----------------------------------------------

run_old = 0

if run_old:
    
    nr1, nc1 = M11.shape
    N_bm = nr1+2

    M_bm = zeros((N_bm, N_bm))

    J_motor = 0.010
    J_bm = 0.0100

    M_bm[2:,2:] = M11
    M_bm[0,0] = J_motor#not sure about units here
    M_bm[1,1] = J_bm


    K_bm = zeros((N_bm, N_bm))
    K_bm[2:,2:] = K11
    k_wall = 100.0

    k_wall_mat = array([[k_wall, -k_wall],[-k_wall, k_wall]])
    K_bm[0:2,0:2] += k_wall_mat

    k_clamp = 1000.0
    k_clamp_mat = array([[k_clamp, -k_clamp],[-k_clamp, k_clamp]])

    K_bm[1:3,1:3] += k_clamp_mat

    c_motor = 5.0
    c_clamp = 1.0
    c_wall = 1.0

    C_bm = zeros((N_bm, N_bm))
    c_wall_mat = array([[c_wall, -c_wall],[-c_wall, c_wall]])
    C_bm[0:2,0:2] += c_wall_mat

    c_clamp = 10.0
    c_clamp_mat = array([[c_clamp, -c_clamp],[-c_clamp, c_clamp]])

    C_bm[1:3,1:3] += c_clamp_mat

    C_bm[0,0] += c_motor


    A_bm = A_mat_from_M_K_C(M_bm,K_bm,C_bm)
    Mi_bm = inv(M_bm)
    nr_bm, nc_bm = M_bm.shape
    zv = zeros((nr_bm,1))
    F_bm = zeros((nr_bm,1))
    #F11[-2] = 1.0#pure translation forcing (no bending moment) at the end of the beam
    F_bm[0] = 0.05#50.0#torque at base
    B_bm = row_stack([zeros((nr_bm,1)), dot(Mi_bm,F_bm)])
    y_row = zeros((1,nr_bm))
    #y_row[0,-2] = 1.0#lateral displacement of the tip#<--- need to fix the
        #output for theta
    y_row[0,1] = 1.0/h*180.0/pi
    C_bm_1 = column_stack([y_row, zeros((1,nr_bm))])
    C_bm_2 = A_bm[-2,:]
    C_bm_ss = row_stack([C_bm_1, C_bm_2])
    D_bm = zeros((2,1))
    D_bm[-1] = B_bm[-2]
    sys_bm = control.ss(A_bm, B_bm, C_bm_ss, D_bm)


    ## mag_bm, phase_bm, omega_bm = sys_bm.freqresp(om)
    ## dB_bm = squeeze(20.0*log10(mag_bm))
    ## phase_bm = squeeze(phase_bm)*180.0/pi

    ## phase_bm2 = unwrap(phase_bm*pi/180.0)*180.0/pi

    ## dB_bm_theta = dB_bm[0,:]
    ## phase_bm_theta = phase_bm2[0,:]
    ## dB_bm_accel = dB_bm[1,:]
    ## phase_bm_accel = phase_bm2[1,:]


    ## bode_utils.bode_plot(freq, dB_bm_theta, phase_bm_theta, \
    ##                      fignum=1, clear=False)


    ## bode_utils.bode_plot(freq, dB_bm_accel, phase_bm_accel, \
    ##                      fignum=2, clear=False)



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

    y_row[0,1] = 1.0/h*180.0/pi
    C_bm_1 = column_stack([y_row, zeros((1,nr_bm))])
    C_bm_2 = A_bm[-2,:]
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

def my_save():
    PU.mysave('theta_bode.png',1)
    PU.mysave('accel_bode.png',2)



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
                  'J_accel':0.0, \
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


## dict1 = build_params_dict()
## print('dict1 = ' + str(dict1))

run_model_and_plot()

# vary one param
case = 2

if case == 1:
    fmt = '$J_{motor} = %0.5g$'
    param = 'J_motor'
    values = [0.001, 0.01, 0.1, 1]
elif case == 2:
    fmt = '$J_{bm} = %0.5g$'
    param = 'J_bm'
    values = [0.005, 0.007, 0.01, 0.015]
elif case == 3:
    fmt = '$k_{wall} = %0.5g$'
    param = 'k_wall'
    values = [50.0, 125.0, 200.0]
elif case == 4:
    fmt = '$c_{wall} = %0.5g$'
    param = 'c_wall'
    values = [0.3, 0.5, 0.6]
elif case == 5:
    fmt = '$k_{clamp} = %0.5g$'
    param = 'k_clamp'
    values = [200.0, 300.0, 400.0]
elif case == 6:
    fmt = '$c_{clamp} = %0.5g$'
    param = 'c_clamp'
    values = [0.1, 0.5, 1.0, 5.0]
elif case == 7:
    fmt = '$c_{motor} = %0.5g$'
    param = 'c_motor'
    values = [4.0, 5.0, 6.0]


my_dict = build_params_dict()

#Pdb().set_trace()

plot_variations = False

if plot_variations:
    for val in values:
        my_dict[param] = val
        mylabel = fmt % val
        run_model_and_plot(label=mylabel, **my_dict)



# Logarythmically downsample to get ready for curve fitting

f_exp = myfile.freq
bode0 = myfile.calc_bode('v','theta')
bode1 = myfile.calc_bode('v','a')

freqs = [0.7, 1.75, 3.25, 5.0, 15.0, 22.0, 40.0]
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


dB_theta_ds = bode0_ds.dBmag()
phase_theta_ds = bode0_ds.phase

dB_accel_ds = bode1_ds.dBmag()
phase_accel_ds = bode1_ds.phase


def mycost0(C, unknown_params, fvect=None):
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


def find_valley(dB_theta, fvect):
    ind1 = thresh(fvect, 1.0)
    ind2 = thresh(fvect, 5.0)
    min_ind = dB_theta[ind1:ind2].argmin() + ind1
    f_valley = fvect[min_ind]
    return f_valley


def find_valley2(C, unknown_params, N=50):
    f2 = 5.0
    f1 = 1.0
    df = (f2-f1)/N
    fvect = arange(f1, f2, df)
    dB_theta, phase_theta, dB_accel, phase_accel = mymodel(C, \
                                                           unknown_params, \
                                                           fvect=fvect)
    min_ind = dB_theta.argmin()
    f_valley = fvect[min_ind]
    return f_valley


def find_second_mode_peak(dB_accel, fvect):
    ind1 = thresh(fvect, 12.0)
    ind2 = thresh(fvect, 25.0)
    max_ind = dB_accel[ind1:ind2].argmax() + ind1
    f_peak = fvect[max_ind]
    return f_peak


exp_valley = find_valley(bode0.dBmag(), f_exp)
exp_peak2 = find_second_mode_peak(bode1.dBmag(), f_exp)

#set up cost function using peak and valley frequencies
def mycost_peak_and_valley(C, unknown_params, fvect=None):
    dB_theta, phase_theta, dB_accel, phase_accel = mymodel(C, \
                                                           unknown_params, \
                                                           fvect=fvect)
    fvect = _get_f(fvect)
    f_valley = find_valley(dB_theta, fvect)
    f_peak2 = find_second_mode_peak(dB_accel, fvect)
    
    e_valley = (exp_valley - f_valley)**2
    e_peak = (exp_peak2 - f_peak2)**2

    return e_valley*10.0 + e_peak


def mycost_valley2(C, unknown_params):
    f_valley = find_valley2(C, unknown_params, N=200)
    e_valley = ((exp_valley - f_valley)*10.0)**2
    return e_valley


#Pdb().set_trace()

#plot_model([0.05],['f_gain'],fmt='g^')

#C_opt = optimize.fmin(mycost0, [0.05], args=(['f_gain'],))
ig_dict = build_params_dict()

restart = 1

if restart:
    ig_vals = ig_dict.values()
else:
    ig_vals = array([  1.59264466e+02,   9.00376923e-01,   8.92998034e-03,
                       3.55755885e+00,   2.46644338e-02,   7.05546550e+00,
                       1.85877501e-02,  -6.30390372e-01])
    
ig_keys = ig_dict.keys()

rerun_all = 0

if rerun_all:
    C_opt_all = optimize.fmin(mycost0, ig_vals, args=(ig_keys,))
else:
    C_opt_all = array([  2.07431592e+02,   3.09193352e-01,   9.48302326e-03,
                         4.99248425e-01,   1.02703297e-01,   1.33489394e+02,
                         5.04511182e-02,   3.87155458e+00])

#plot_model(ig_vals, ig_keys, fmt='g-', label='fit i.g.')
#plot_model(C_opt_all, ig_keys, fmt='r-', label='fit opt')

## run_model_and_plot(f_gain=0.05)
## run_model_and_plot()

my_limits()

## dict2 = build_params_dict()
## print('dict2 = ' + str(dict2))


# find the valley in the theta/v bode and the second mode peak in
# accel/v and vary k's and J's to drive the model frequencies to the
# experimental ones
fvect = _get_f(fvect=None)
## params1 = ['J_motor','J_bm','k_wall','k_clamp']
params1 = ['J_motor', \
           'J_bm', \
           'k_wall', \
           'k_clamp', \
           'c_motor', \
           'c_clamp', \
           'c_wall', \
           'f_gain', \
           ]
                  

f_valley = find_valley(dB_theta, fvect)
f_peak2 = find_second_mode_peak(dB_accel, fvect)


fvect = logspace(-1,2,1000)

mydict = build_params_dict()
dict1 = copy.copy(mydict)
dict2 = copy.copy(mydict)
dict3 = copy.copy(mydict)
param1 = 'k_clamp'
dict1[param1] = 0.3*mydict[param1]
param1b = 'c_clamp'
dict1[param1b] = 0.5*mydict[param1b]
param2 = 'J_bm'
dict2[param2] = 500.0*mydict[param2]

param3 = 'k_wall'
dict2[param3] = 0.1*mydict[param3]
param4 = 'c_wall'
dict2[param4] = 0.1*mydict[param4]

plot_3_dicts = 0

if plot_3_dicts:
    fi1 = 10
    fi2 = fi1+1
    myfile.bode_plot('v','theta',seedfreq=2.45, seedphase=-200.0, \
                     label='exp', fignum=fi1)
    myfile.bode_plot('v','a',fignum=fi2, seedfreq=10.0, seedphase=-150.0, \
                     label='exp')

    plot_model(dict1.values(), dict1.keys(), fmt='m-', label='dict1', \
               fvect=fvect, fi1=fi1)
    plot_model(dict2.values(), dict2.keys(), fmt='c-', label='dict2', \
               fvect=fvect, fi1=fi1)
    plot_model(dict3.values(), dict3.keys(), fmt='k-', label='dict3', \
               fvect=fvect, fi1=fi1)

fi1 = 100
myfile = txt_data_processing.Data_File('swept_sine_kp_1_amp_50.txt')
myfile.bode_plot('v','theta',seedfreq=2.45, seedphase=-200.0, label='exp', \
                 fignum=fi1)
myfile.bode_plot('v','a',fignum=fi1+1, seedfreq=10.0, seedphase=-150.0, \
                 label='exp')

dict_m = copy.copy(mydict)
m_list = [0.003,0.005,0.007,0.01]

colors = ['k','r','b','g','c','y']

for m,c in zip(m_list,colors):
    dict_m['m_accel'] = m
    mylabel = '$m = %0.4g$' % m
    plot_model(dict_m.values(), dict_m.keys(), fmt=c, label=mylabel, \
               fvect=fvect, fi1=fi1)


fi1 = 200
myfile = txt_data_processing.Data_File('swept_sine_kp_1_amp_50.txt')
myfile.bode_plot('v','theta',seedfreq=2.45, seedphase=-200.0, label='exp', \
                 fignum=fi1)
myfile.bode_plot('v','a',fignum=fi1+1, seedfreq=10.0, seedphase=-150.0, \
                 label='exp')


dict_J = copy.copy(mydict)
J_list = [0.0,0.0001,0.001,]


for J, c in zip(J_list,colors):
    dict_J['J_accel'] = J
    mylabel = '$J = %0.4g$' % J
    plot_model(dict_J.values(), dict_J.keys(), fmt=c, label=mylabel, \
               fvect=fvect, fi1=fi1)


vals1 = dict1.values()
keys1 = dict1.keys()

#trying to find good initial guesses
fi1 = 300
myfile = txt_data_processing.Data_File('swept_sine_kp_1_amp_50.txt')
myfile.bode_plot('v','theta',seedfreq=2.45, seedphase=-200.0, label='exp', \
                 fignum=fi1)
myfile.bode_plot('v','a',fignum=fi1+1, seedfreq=10.0, seedphase=-150.0, \
                 label='exp')

dict_ig = copy.copy(dict_J)
dict_ig['J_accel'] = 0.001
plot_model(dict_ig.values(), dict_ig.keys(), fmt='g', label=mylabel, \
           fvect=fvect, fi1=fi1)
mykey = 'k_wall'
#mykey = 'k_clamp'
mykey2 = 'k_clamp'
dict_ig[mykey] *= 2.0
dict_ig[mykey2] *= 0.5
plot_model(dict_ig.values(), dict_ig.keys(), fmt='r', label=mylabel, \
           fvect=fvect, fi1=fi1)


## C_opt_all2 = optimize.fmin(mycost0, vals1, args=(keys1,))

## plot_model(C_opt_all2, keys1, fmt='r-', label='opt_all2', \
##            fvect=fvect, fi1=fi1)


## for i, param in enumerate(params1):
##     fi1 = (i+1)*10
##     fi2 = fi1 + 1
##     myfile.bode_plot('v','theta',seedfreq=2.45, seedphase=-200.0, \
##                      label='exp', fignum=fi1)
##     myfile.bode_plot('v','a',fignum=fi2, seedfreq=10.0, seedphase=-150.0, \
##                      label='exp')
##     mydict = build_params_dict()
##     dict1 = copy.copy(mydict)
##     dict2 = copy.copy(mydict)
##     dict1[param] = 0.5*mydict[param]
##     dict2[param] = 2*mydict[param]
##     plot_model(dict1.values(), dict1.keys(), fmt='m-', label='dict1', \
##                fvect=fvect, fi1=fi1)
##     plot_model(dict2.values(), dict1.keys(), fmt='c-', label='dict2', \
##                fvect=fvect, fi1=fi1)
##     PU.SetTitle(fi1, param)
    


sys_ig = get_sys_from_params(ig_vals, ig_keys)
#sys = get_sys_from_params(C_opt_all, ig_keys)
sys = get_sys_from_params(vals1, keys1)


myfile = txt_data_processing.Data_File('OL_pulse_test_sys_check_SLFR_RTP_OL_Test_uend=0.txt')
t = myfile.t
u = myfile.u

do_time_domain_plot = 0

if do_time_domain_plot:
    myfile.Time_Plot(labels=['u','theta','a'], fignum=3)

    y, to, xo = control.lsim(sys, u, t)

    figure(3)
    plot(t, y[:,0], label='model $\\theta$')
    plot(t, y[:,1], label='model $\\ddot{x}_{tip}$')

    figure(3)
    legend(loc=8)


figure(1)
subplot(211)
legend(loc=3)

###################################
#
# Time Domain fit
#
###################################
ind1 = 500
t_fit = t[0:ind1]
u_fit = u[0:ind1]
theta_fit = myfile.theta[0:ind1]
a_fit = myfile.a[0:ind1]

figure(4)
clf()
plot(t_fit, u_fit, t_fit, theta_fit, t_fit, a_fit)

def my_td_model(C, unknown_params):
    """This will be the underlying model file to be used with a
    similar cost function to be passed to optimize.fmin.  The list of
    unknown_params will be passed to optimize.fmin as the first arg in
    args.  This list will be used with C to build the dictionary of
    parameters that are different from the defaults."""
    sys = get_sys_from_params(C, unknown_params)
    y, to, xo = control.lsim(sys, u_fit, t_fit)
    return y[:,0], y[:,1]


def my_td_cost(C, unknown_params):
    theta_model, a_model = my_td_model(C, unknown_params)
    theta_error = theta_fit - theta_model
    a_error = a_fit - a_model
    e_theta = (theta_error**2).sum()
    e_a = (a_error**2).sum()
    return 2*e_theta + e_a


## ig2 = array([  5.81112498e+01,   1.04038679e+00,   2.14609263e-02,
##                6.19398918e-01,   4.64824662e-02,   1.75469591e+02,
##                5.39174678e-02,   2.86777116e+00])
 
## ig2 = array([  5.80979184e+01,   1.04201505e+00,   2.14538932e-02,
##                6.19694766e-01,   4.65436579e-02,   1.75605482e+02,
##                5.39150127e-02,   2.86759670e+00])

ig2 = array([  5.80979184e+01,   1.04201505e+00,   2.14538932e-02,
               6.19694766e-01,   4.65436579e-02,   1.75605482e+02,
               5.39150127e-02,   2.86759670e+00])

theta_td_ig2, a_td_ig2 = my_td_model(ig2, keys1)
plot(t_fit, theta_td_ig2, t_fit, a_td_ig2)

## fi1 = 5
## fi2 = fi1+1

## myfile.bode_plot('v','theta',seedfreq=2.45, seedphase=-200.0, \
##                  label='exp', fignum=fi1)
## myfile.bode_plot('v','a',fignum=fi2, seedfreq=10.0, seedphase=-150.0, \
##                  label='exp')

plot_model(ig2, keys1, fmt='m-', label='dict1', \
           fvect=fvect, fi1=1)

## C_opt_td = optimize.fmin(my_td_cost, vals1, args=(keys1,))
## C_opt_td = optimize.fmin(my_td_cost, ig2, args=(keys1,))
## theta_td_opt, a_td_opt = my_td_model(C_opt_td, keys1)
## plot(t_fit, theta_td_opt, t_fit, a_td_opt)

show()
