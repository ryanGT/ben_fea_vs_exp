from matplotlib.pyplot import *
from scipy import *

import control, bode_utils

import SFLR_SS_model as SS
reload(SS)

#import bode_options
#reload(bode_options)
#bode_options.bode_opt_th_v.maglim=[-80,20]
#bode_options.bode_opt_a_v.maglim=[-80,20]
#OL_opts = bode_options.OL_bode_opts

import txt_data_processing as TDP
reload(TDP)

#Specify Bode Plotting Options
myfreqlim = [0.5,30]
#myfreqlim = [0.1,100]

bode_opt_th_v = TDP.Bode_Options(input_label='v', \
                                 output_label='theta', \
                                 freqlim=myfreqlim, \
                                 maglim=[-50,20], \
                                 phaselim=[-250,25], \
                                 seedfreq=1.0, \
                                 seedphase=-120.0)

bode_opt_th_u = TDP.Bode_Options(input_label='u', \
                                 output_label='theta', \
                                 freqlim=myfreqlim, \
                                 maglim=[-50,20], \
                                 phaselim=[-250,25], \
                                 seedfreq=1.0, \
                                 seedphase=-30.0)

bode_opt_a_v = TDP.Bode_Options(input_label='v', \
                                output_label='a', \
                                freqlim=myfreqlim, \
                                maglim=[-20, 20], \
                                phaselim=[-400,200], \
                                seedfreq=1.0, \
                                seedphase=90.0)

bode_opt_a_u = TDP.Bode_Options(input_label='u', \
                                output_label='a', \
                                freqlim=myfreqlim, \
                                maglim=[-20, 35], \
                                phaselim=[-400,200], \
                                seedfreq=1.0, \
                                seedphase=150.0)


OL_opts = [bode_opt_th_v, bode_opt_a_v]


myfile = TDP.Data_File('swept_sine_kp_1_amp_50.txt')
myfile.bode_plot('v','theta',seedfreq=2.45, seedphase=-200.0, label='exp')
myfile.bode_plot('v','a',fignum=2, seedfreq=10.0, seedphase=-150.0, label='exp')


#down sample the Bodes
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


fit_bodes = [bode0_ds, bode1_ds]


def build_CCF_model(poles, zeros, bode_opts=OL_opts):
    mymodel = SS.SFLR_CCF_model(poles, \
                                zeros=zeros, \
                                bode_opts=bode_opts)
    mymodel.bode_input = 'v'
    mymodel.set_fit_parames(fit_bodes, f_ds)
    mymodel.find_C_gains()
    mymodel.check_C_signs()
    return mymodel


wn1 = 2.8*2*pi
zeta1 = 0.25
wn2 = 18.5*2*pi
zeta2 = 0.01
wd1 = wn1*sqrt(1-zeta1**2)
wd2 = wn2*sqrt(1-zeta2**2)
wz1 = 2.47*2*pi
zz1 = 0.03
wdz1 = wz1*sqrt(1-zz1**2)
z1 = -wz1*zz1+1.0j*wdz1
z1c = conj(z1)
p1 = -zeta1*wn1+1.0j*wd1
p1c = conj(p1)
p2 = -zeta2*wn2+1.0j*wd2
p2c = conj(p2)

pr1 = -1.5*2*pi
pr2 = -45.0*2*pi

wz2 = 18.0*2*pi
zz2 = 0.03
wdz2 = wz2*sqrt(1-zz2**2)
z2 = -wz2*zz2+1.0j*wdz2
z2c = conj(z2)

          
poles = array([0.0, p1, p1c, p2, p2c, pr1])#, pr2])
#poles = array([0.0, p1, p1c, p2, p2c])
zeros1 = array([z1,z1c,z2,z2c])
zeros2 = array([0,0.0])
zeros = [zeros1, zeros2]

mymodel = build_CCF_model(poles, zeros)

fvect = logspace(-1,2,1000)
om = 2*pi*fvect

sys = control.ss(mymodel.A, mymodel.B, mymodel.C, [[0],[0]])
mag, phase, omega = sys.freqresp(om)
dB = squeeze(20.0*log10(mag))
phase_r = squeeze(phase)
phase_r = unwrap(phase_r)
phase_d = phase_r*180.0/pi

bode_utils.bode_plot(fvect, dB[0,:], phase_d[0,:], \
                     fignum=1, clear=False)

bode_utils.bode_plot(fvect, dB[1,:], phase_d[1,:], \
                     fignum=2, clear=False)


# need the accel/v bode to have B1 and B2 in the numerators
# separately; add the terms together to find the corresponding zero


show()
