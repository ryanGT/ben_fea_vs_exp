from pylab import *
from scipy import *

from scipy import optimize

import TMM, rwkbode
import controls
import pdb
import time

pklname = 'model_w_bm_opt.pkl'
import rwkmisc, misc_utils
import pylab_util as PU
mydict = rwkmisc.LoadPickle(pklname)

from IPython.Debugger import Pdb

beam_params = {}
beam_keys = ['mu','L','EI']
for key in beam_keys:
    beam_params[key] = mydict[key]



df_name = 'SFLR_OL_pulse_test_no_decline_trunc_trunc_downsampled.txt'
data = loadtxt(df_name, skiprows=1)

t = data[:,0]
v = data[:,1]
theta_exp = data[:,2]
a_exp = data[:,3]



figure(1)
clf()
plot(t,v,label='$v_{exp}$')
plot(t,theta_exp, label='$\\theta_{exp}$')
plot(t,a_exp, label='$\\ddot{x}_{exp}$')


N2 = 10#the beam will be broken into N2 masses and N2-1 springs (TSDs)
mu = beam_params['mu']
L = beam_params['L']
EI = beam_params['EI']
L2 = L/N2#length of each discretized beam section
L2b = L/(N2-1)#number of springs
k2 = EI/L2b#<-- using L2b seems to improve the agreement with the
    #continuous beam model
m2 = mu*L2

h_i = 1.0/32#inches
i_to_m = 25.4/1000.0#convert inches to meters
h = h_i * i_to_m
I2 = (1.0/12)*m2*(L2**2+h**2)#h is the height or thickness of the beam

m2_params = {'m':m2, 'L':L2, 'r':L2*0.5, 'I': I2}
tsd2_params = {'k':k2, 'c':0.0}

## bode_dict2= {'input':'F','output':'x','type':'abs',\
##              'ind':last_mass,'post':'','dof':0}
## bode_out2 = rwkbode.bodeout(**bode_dict2)

######################################################
#
# DT-TMM
#
######################################################
import DTTMM
reload(DTTMM)

#T = 2.0
dt = t[2]-t[1]
T = t.max()
#T = 10.0
#dt = 1.0/500
#dt = 1.0/500
#t = arange(0,T,dt)
N = len(t)

#g = mydict['K_act']

p = mydict['p_act1']
g = mydict['num_act']/p

b_m = mydict['b_m']
b_L = mydict['b_L']
b_r = mydict['b_r']
b_I = mydict['b_I']
bm_params = {'m':b_m, 'L':b_L, 'r':b_r, 'I':b_I}

case = 2

if case == 1:
    k_clamp = 1.0e2
    c_clamp = 1.0e-6

    k_spring = 5.0
    c_spring = 1.0e-2

elif case == 2:
    C_opt1 = array([  8.96725909e-01,   3.63151344e-02,   4.00905919e+00,
                      8.12525142e-14])
    
    k_spring = C_opt1[0]
    c_spring = C_opt1[1]
    k_clamp = C_opt1[2]
    c_clamp = C_opt1[3]

else:
    k_clamp = mydict['k_clamp']
    c_clamp = mydict['c_clamp']

    k_spring = mydict['k_spring']
    c_spring = mydict['c_spring']


a_m = mydict['a_m']
a_L = mydict['a_L']
a_r = mydict['a_r']
a_I = mydict['a_I']
a_gain = mydict['a_gain']
am_params = {'m':a_m, 'L':a_L, 'r':a_r, 'I':a_I}

actuator = DTTMM.DT_TMM_DC_motor_4_states(g, p, dt, v=v)
tsd_spring_DTTMM = DTTMM.DT_TMM_TSD_4_states(k_spring, c_spring, \
                                             prev_element=actuator, \
                                             name='spring')
base_mass = DTTMM.DT_TMM_rigid_mass_4_states(f=None, \
                                             prev_element=tsd_spring_DTTMM, \
                                             name='base_mass', \
                                             **bm_params)
tsd_clamp_DTTMM = DTTMM.DT_TMM_TSD_4_states(k_clamp, c_clamp, \
                                            prev_element=base_mass, \
                                            name='clamp')

rigid_mass0_DTTMM = DTTMM.DT_TMM_rigid_mass_4_states(f=None, \
                                                     prev_element=tsd_clamp_DTTMM, \
                                                     **m2_params)
tsd2_DTTMM_params = {'k':tsd2_params['k'], \
                     'b':tsd2_params['c']}

elem_list_DTTMM = [actuator, tsd_spring_DTTMM, base_mass, \
                   tsd_clamp_DTTMM, rigid_mass0_DTTMM]

debug_list = [actuator, tsd_spring_DTTMM, base_mass]

initial_conditions = None
prev_mass = rigid_mass0_DTTMM

for i in range(N2-1):
    TSD_i = DTTMM.DT_TMM_TSD_4_states(prev_element=prev_mass, \
                                      **tsd2_DTTMM_params)
    elem_list_DTTMM.append(TSD_i)
    mass_i = DTTMM.DT_TMM_rigid_mass_4_states(prev_element=TSD_i, \
                                              **m2_params)
    elem_list_DTTMM.append(mass_i)
    prev_mass = mass_i

last_mass_DTTMM = mass_i
accel_mass = DTTMM.DT_TMM_rigid_mass_4_states(f=None, \
                                              prev_element=last_mass_DTTMM, \
                                              name='accel_mass', \
                                              **am_params)
elem_list_DTTMM.append(accel_mass)

actuator.v = v
y_out = zeros(N)


class SFLR_DTTMM_sys(DTTMM.DT_TMM_System_clamped_free_four_states):
    def mymodel(self, C):
        """This method will probably need to be overwritten based on
        the output for a specific system."""
        self.set_params(C)
        self.Run_Simulation(self.N, self.dt, \
                            initial_conditions=self.initial_conditions, \
                            int_case=self.int_case)

        a_tip_DTTMM = accel_mass.xddot
        theta_mass0 = base_mass.theta*mydict['H']
        return theta_mass0, a_tip_DTTMM


    def mycost(self, C):
        theta_DTTMM, a_tip_DTTMM = self.mymodel(C)
        e_theta_vect = theta_exp - theta_DTTMM
        e_theta = ((e_theta_vect)**2).sum()
        e_a_vect = a_exp - a_tip_DTTMM
        e_a = ((e_a_vect)**2).sum()

        penalty = self.check_limits(C)

        return e_theta + e_a + penalty
 
        
#unknown_params=[], \
#param_limits={}):

fit_res1 = {'accel_mass.I': 9.2839468774339442e-06,
            'base_mass.I': 0.00073011949112655847,
            'base_mass.m': 0.33904070790691632}


fit_res2 = {'accel_mass.I': 2.9686323382398135e-13,
            'base_mass.I': 0.0016325506191516344,
            'base_mass.m': 0.2021801932515101,
            'clamp.b': 9.3106021614701947e-14,
            'clamp.k': 3.7390590148730007,
            'spring.b': 0.11864406696414578,
            'spring.k': 0.39394023330870281}


unknown_params = ['base_mass.m', \
                  'base_mass.I', \
                  'accel_mass.I', \
                  'spring.k', \
                  'spring.b', \
                  'clamp.k', \
                  'clamp.b', \
                  ]


param_limits = {'base_mass.m': [0.5*b_m, 10.0], \
                }


sys = SFLR_DTTMM_sys(elem_list_DTTMM, N_states=4, N=N, dt=dt, int_case=1, \
                     initial_conditions=initial_conditions, \
                     unknown_params=unknown_params, \
                     param_limits=param_limits)

sys.set_params_from_dict(fit_res2)

run_new_fit = 0

if run_new_fit:
    ig = sys.get_ig()
    theta_ig, accel_ig = sys.mymodel(ig)
    plot(t,theta_ig, label='$\\theta_{ig}$')
    plot(t,accel_ig, label='$\\ddot{x}_{ig}$')
    initial_cost = sys.mycost(ig)

    C_opt = sys.run_fit()
    theta_opt, accel_opt = sys.mymodel(C_opt)
    plot(t,theta_opt, label='$\\theta_{opt}$')
    plot(t,accel_opt, label='$\\ddot{x}_{opt}$')
    opt_cost = sys.mycost(C_opt)


legend(loc=1)

show()
