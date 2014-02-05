from matplotlib.pyplot import *
from scipy import *

from scipy import optimize

import fea_vs_exp_utils as utils
reload(utils)

ig_dict = utils.build_params_dict()
ig_vals = ig_dict.values()
ig_keys = ig_dict.keys()

utils.plot_model(ig_vals, ig_keys)

#plot_guesses(['k_wall'],[0.5], fmt='g-')
#plot_guesses(['k_clamp'],[0.6], fmt='c-')
#plot_guesses(['k_wall','J_accel'],[2.0,2.0], fmt='r-')
new_dict = utils.plot_guesses(['k_clamp','J_accel','k_wall'],[0.6,0.5,2.0], fmt='g-')

utils.my_limits()


known_params = ['m_accel']#assuming this is measured using a scale (I
                          #think I did this awhile ago)

unknown_params = new_dict.keys()

for param in known_params:
    index = unknown_params.index(param)
    unknown_params.pop(index)


ig_list = [new_dict[key] for key in unknown_params]

C_opt0 = optimize.fmin(utils.mycost0, ig_list, args=(unknown_params,))

utils.plot_model(C_opt0, unknown_params, fmt='r-')


show()
