from matplotlib.pyplot import *
from scipy import *

from scipy import optimize

import fea_vs_exp_utils as utils
reload(utils)

import pylab_util as PU

import rwkmisc
import copy

ig_dict = utils.build_params_dict()
ig_vals = ig_dict.values()
ig_keys = ig_dict.keys()

#utils.plot_model(ig_vals, ig_keys)

#plot_guesses(['k_wall'],[0.5], fmt='g-')
#plot_guesses(['k_clamp'],[0.6], fmt='c-')
#plot_guesses(['k_wall','J_accel'],[2.0,2.0], fmt='r-')
new_dict = utils.plot_guesses(['k_clamp','J_accel','k_wall'],[0.6,0.5,2.0], fmt='g-', label='ig')

utils.my_limits()


known_params = ['m_accel']#assuming this is measured using a scale (I
                          #think I did this awhile ago)

unknown_params = new_dict.keys()

for param in known_params:
    index = unknown_params.index(param)
    unknown_params.pop(index)


ig_list = [new_dict[key] for key in unknown_params]

run_opt0 = False

if run_opt0:
    C_opt0 = optimize.fmin(utils.mycost0, ig_list, args=(unknown_params,))
    utils.plot_model(C_opt0, unknown_params, fmt='r-')


# Try using the parameters from the TMM fit
pklpath = 'model_w_bm_opt.pkl'
tmm_dict = rwkmisc.LoadPickle(pklpath)
dict2 = copy.copy(ig_dict)
map_keys = {'a_m':'m_accel', \
            'a_I':'J_accel', \
            'k_spring':'k_wall', \
            'c_spring':'c_wall', \
            'k_clamp':'k_clamp', \
            'c_clamp':'c_clamp', \
            'b_I':'J_bm', \
            }

for key1, key2 in map_keys.iteritems():
    dict2[key2] = tmm_dict[key1]


# in order for this to work, I need to deal with the fact that Meirovitch
# uses theta*h for rotational states and tau/h for torque inputs
rotational_params = ['k_clamp', \
                     'J_accel', \
                     'J_bm', \
                     'c_clamp', \
                     'J_motor', \
                     'k_wall', \
                     'c_wall', \
                     'c_motor']

h = utils.h
h_factor = 1.0/(h**2)

for key in rotational_params:
    dict2[key] *= h_factor


#dict2['f_gain'] *= h_factor

# do the TMM actuator modeling stuff
K_act = tmm_dict['K_act']
p_act1 = tmm_dict['p_act1']
s1 = 1.0*2.0j*pi
m1 = abs(s1+p_act1)
num = K_act*m1

#dict2['f_gain'] = 1.0/(h**2)
dict2['f_gain'] = 0.12
J_motor = 1.0/(num)
dict2['J_motor'] = J_motor
c_motor = p_act1*J_motor
dict2['c_motor'] = c_motor

utils.plot_from_dict(dict2, fmt='m-', label='TMM params')

PU.SetLegend(1, loc=2)
PU.SetLegend(2, loc=2)

show()
