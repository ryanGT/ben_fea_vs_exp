from pylab import *
from scipy import *

from scipy import optimize

import TMM, rwkbode
import controls
import pdb
import time

import DTTMM

import rwkmisc, misc_utils
import pylab_util as PU

#from IPython.Debugger import Pdb

update_params = True

JVC_model_dict = {'EI': 0.17457571410633099,
                  'H': 162.97466172610083,
                  'K_act': 0.069630106141704312,
                  'L': 0.41354374999999999,
                  'L2': 0.0071437499999999999,
                  'a_I': 7.0627000000000004e-07,
                  'a_L': 0.011112499999999999,
                  'a_gain': 1.0802616846169428,
                  'a_m': 0.0060566908955645912,
                  'a_r': 0.0055562499999999996,
                  'b_I': 0.0016121636,
                  'b_L': 0.011124999999999999,
                  'b_m': 0.40436032,
                  'b_r': 0.0055560000000000002,
                  'c_beam': 0.0,
                  'c_clamp': 0.0038970760000000002,
                  'c_spring': 0.079451019402440995,
                  'k_clamp': 2.4977992000000002,
                  'k_spring': 0.10796957049311773,
                  'mu': 0.12171171088558749,
                  'num_act': 8.1111591586068279,
                  'p_act1': 209.35291659094389,
                  'p_act2': 314.15926535897933,
                  'tau': 62.831853071795862,
                  'z_act': 18.849555921538759}

beam_params = {}
beam_keys = ['mu','L','EI']
for key in beam_keys:
    beam_params[key] = JVC_model_dict[key]



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

#g = JVC_model_dict['K_act']

p = JVC_model_dict['p_act1']
g = JVC_model_dict['num_act']/p

b_m = JVC_model_dict['b_m']
b_L = JVC_model_dict['b_L']
b_r = JVC_model_dict['b_r']
b_I = JVC_model_dict['b_I']
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
    k_clamp = JVC_model_dict['k_clamp']
    c_clamp = JVC_model_dict['c_clamp']

    k_spring = JVC_model_dict['k_spring']
    c_spring = JVC_model_dict['c_spring']


a_m = JVC_model_dict['a_m']
a_L = JVC_model_dict['a_L']
a_r = JVC_model_dict['a_r']
a_I = JVC_model_dict['a_I']
a_gain = JVC_model_dict['a_gain']
am_params = {'m':a_m, 'L':a_L, 'r':a_r, 'I':a_I}

dt = 1.0/500#<-- this should not be hard coded

actuator = DTTMM.DT_TMM_DC_motor_4_states(g, p, dt, v=None, name='actuator')#<-- this is where dt is needed
                                                                            #in the c2d of the actuator
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



class SFLR_DTTMM_OL_sys(DTTMM.DT_TMM_System_clamped_free_four_states):
    def Run_Simulation(self, N, dt, initial_conditions=None, \
                       int_case=1):
        DTTMM.DT_TMM_System_clamped_free_four_states.Run_Simulation(self, N, dt, \
                                                                    initial_conditions=initial_conditions, \
                                                                    int_case=int_case)
        self.theta = base_mass.theta*JVC_model_dict['H']
        self.accel = accel_mass.xddot

    
    def mymodel(self, C):
        """This method will probably need to be overwritten based on
        the output for a specific system."""
        self.set_params(C)
        self.Run_Simulation(self.N, self.dt, \
                            initial_conditions=self.initial_conditions, \
                            int_case=self.int_case)

        a_tip_DTTMM = accel_mass.xddot
        theta_mass0 = base_mass.theta*JVC_model_dict['H']
        return theta_mass0, a_tip_DTTMM


    def mycost(self, C):
        theta_DTTMM, a_tip_DTTMM = self.mymodel(C)
        e_theta_vect = theta_exp - theta_DTTMM
        e_theta = ((e_theta_vect)**2).sum()
        e_a_vect = a_exp - a_tip_DTTMM
        e_a = ((e_a_vect)**2).sum()

        penalty = self.check_limits(C)

        return e_theta + e_a + penalty


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


OL_sys = SFLR_DTTMM_OL_sys(elem_list_DTTMM, N_states=4, int_case=1, \
                           initial_conditions=initial_conditions, \
                           unknown_params=unknown_params, \
                           param_limits=param_limits)

if update_params:
    OL_sys.set_params_from_dict(fit_res2)



class SFLR_DTTMM_theta_feedback(SFLR_DTTMM_OL_sys):
    def __init__(self, element_list, N_states, \
                 feedback_mass, actuator, kp=1.0, \
                 **kwargs):
        SFLR_DTTMM_OL_sys.__init__(self, element_list, N_states, **kwargs)
        self.feedback_mass = feedback_mass
        self.actuator = actuator
        self.kp = kp



    def Run_Simulation(self, theta_d, N, dt, initial_conditions=None, int_case=1):
        self._initialize_vectors(N)
        self.t = arange(0, N*dt, dt)
        self._assign_t_vect()
        if initial_conditions is not None:
            self._set_initial_conditions(initial_conditions)

        if self.actuator.v is None:
            self.actuator.v = zeros(N)
            
        if self.actuator.dt != dt:
            self.actuator.dt = dt
            self.actuator.c2d()
            
        for i in range(1,N):    # Time loop
            th_d_i = theta_d[i]
            th_a_i = self.feedback_mass.theta[i-1]*JVC_model_dict['H']#get theta encoder measurement from previous time step
            e = th_d_i - th_a_i
            v = e*self.kp
            #trying to debug
            ## if (i > 10) and (i < 100):
            ##     v = 20.0
            ## else:
            ##     v = 0.0
            self.actuator.v[i] = v
            self.calculate_ABDE(i, dt, int_case=int_case)
            self.calculate_transfer_matrices(i)
            self.calculate_system_transfer_matrix(i)
            self.solve_boundary_conditions(i)#must set self.z0
            self.calculate_state_vectors(i)
            self.calculate_velocity_and_accel(i)


        self.theta = self.feedback_mass.theta*JVC_model_dict['H']
        self.accel = accel_mass.xddot


CL_sys_P_control = SFLR_DTTMM_theta_feedback(elem_list_DTTMM, N_states=4,
                                             feedback_mass=base_mass, \
                                             actuator=actuator, \
                                             int_case=1, \
                                             initial_conditions=initial_conditions, \
                                             unknown_params=unknown_params, \
                                             param_limits=param_limits)

if update_params:
    CL_sys_P_control.set_params_from_dict(fit_res2)


class SFLR_DTTMM_theta_feedback_digcomp(SFLR_DTTMM_OL_sys):
    """Model the SFLR with theta feedback using a digital compensator."""
    def __init__(self, element_list, N_states, \
                 feedback_mass, actuator, Gth, \
                 **kwargs):
        SFLR_DTTMM_OL_sys.__init__(self, element_list, N_states, **kwargs)
        self.feedback_mass = feedback_mass
        self.actuator = actuator
        self.Gth = Gth



    def Run_Simulation(self, theta_d, N, dt, initial_conditions=None, int_case=1):
        self._initialize_vectors(N)
        self.t = arange(0, N*dt, dt)
        self._assign_t_vect()
        if initial_conditions is not None:
            self._set_initial_conditions(initial_conditions)

        if self.actuator.v is None:
            self.actuator.v = zeros(N)

        if self.actuator.dt != dt:
            self.actuator.dt = dt
            self.actuator.c2d()

        self.evect = zeros(N)
        self.vvect = zeros(N)
        self.Gth.input = self.evect
        self.Gth.output = self.vvect
        self.actuator.v = self.vvect

        for i in range(1,N):    # Time loop
            th_d_i = theta_d[i]
            th_a_i = self.feedback_mass.theta[i-1]*JVC_model_dict['H']#get theta encoder measurement from previous time step
            self.evect[i] = th_d_i - th_a_i
            y_out = self.Gth.calc_out(i) # parenthesis was missing
            if y_out > 200.0:
                y_out = 200
            elif y_out < -200.0:
                y_out = -200
            self.Gth.output[i] = y_out

            self.actuator.v[i] = y_out
            self.calculate_ABDE(i, dt, int_case=int_case)
            self.calculate_transfer_matrices(i)
            self.calculate_system_transfer_matrix(i)
            self.solve_boundary_conditions(i)#must set self.z0
            self.calculate_state_vectors(i)
            self.calculate_velocity_and_accel(i)


        self.theta = self.feedback_mass.theta*JVC_model_dict['H']
        self.accel = accel_mass.xddot


Gth_num = [18.32547182, -17.20911971]
Gth_den = [1., -0.77672958]
Gth = controls.Digital_Compensator(Gth_num,Gth_den)

digcomp_sys = SFLR_DTTMM_theta_feedback_digcomp(elem_list_DTTMM, N_states=4,
                                                feedback_mass=base_mass, \
                                                actuator=actuator, \
                                                Gth=Gth, \
                                                int_case=1, \
                                                initial_conditions=initial_conditions, \
                                                unknown_params=unknown_params, \
                                                param_limits=param_limits)

if update_params:
    digcomp_sys.set_params_from_dict(fit_res2)




# Accel Feedback

class SFLR_DTTMM_theta_accel_feedback_digcomp(SFLR_DTTMM_OL_sys):
    """Model the SFLR with theta feedback using a digital compensator."""
    def __init__(self, element_list, N_states, \
                 feedback_mass, actuator, Gth, \
                 Ga, \
                 **kwargs):
        SFLR_DTTMM_OL_sys.__init__(self, element_list, N_states, **kwargs)
        self.feedback_mass = feedback_mass
        self.actuator = actuator
        self.Gth = Gth
        self.Ga = Ga


    def Run_Simulation(self, theta_d, N, dt, initial_conditions=None, int_case=1):
        self._initialize_vectors(N)
        self.t = arange(0, N*dt, dt)
        self._assign_t_vect()
        if initial_conditions is not None:
            self._set_initial_conditions(initial_conditions)

        if self.actuator.v is None:
            self.actuator.v = zeros(N)

        if self.actuator.dt != dt:
            self.actuator.dt = dt
            self.actuator.c2d()

        self.evect = zeros(N)
        self.vvect = zeros(N)
        self.Gth.input = self.evect
        self.Gth.output = self.vvect
        self.actuator.v = self.vvect

        self.evect_acc = zeros(N)
        self.va = zeros(N)
        self.Ga.input = self.evect_acc
        self.Ga.output= self.va

        for i in range(1,N):    # Time loop
            Ga.input[i] = accel_mass.xddot[i-1]
            self.va[i] = Ga.calc_out(i) # Output of Ga
            th_d_i = theta_d[i]-self.va[i] # theta_hat
            th_a_i = self.feedback_mass.theta[i-1]*JVC_model_dict['H']#get theta encoder measurement from previous time step
            self.evect[i] = th_d_i - th_a_i
            y_out = self.Gth.calc_out(i) # parenthesis was missing
            if y_out > 200.0:
                y_out = 200
            elif y_out < -200.0:
                y_out = -200
            self.Gth.output[i] = y_out

            self.actuator.v[i] = y_out
            self.calculate_ABDE(i, dt, int_case=int_case)
            self.calculate_transfer_matrices(i)
            self.calculate_system_transfer_matrix(i)
            self.solve_boundary_conditions(i)#must set self.z0
            self.calculate_state_vectors(i)
            self.calculate_velocity_and_accel(i)


        self.theta = self.feedback_mass.theta*JVC_model_dict['H']
        self.accel = accel_mass.xddot


Ga_num=[0.00045097, 0.00045152, -0.00044989, -0.00045043]
Ga_den=[1., -2.9289362, 2.85853684, -0.92960052]

Ga = controls.Digital_Compensator(Ga_num, Ga_den)

accel_fb_sys = SFLR_DTTMM_theta_accel_feedback_digcomp(elem_list_DTTMM, N_states=4,
                                                       feedback_mass=base_mass, \
                                                       actuator=actuator, \
                                                       Gth=Gth, \
                                                       Ga=Ga, \
                                                       int_case=1, \
                                                       initial_conditions=initial_conditions, \
                                                       unknown_params=unknown_params, \
                                                       param_limits=param_limits)

if update_params:
    accel_fb_sys.set_params_from_dict(fit_res2)
