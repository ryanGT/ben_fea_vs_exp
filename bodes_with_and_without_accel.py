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

figure(1)
clf()

figure(2)
clf()

args1 = ('v','theta')
kwargs1 = {'seedfreq':2.45, 'seedphase':-200.0, 'label':'old data'}
args2 = ('v','a')
kwargs2 = {'fignum':2, 'seedfreq':10.0, 'seedphase':-150.0, 'label':'old data'}

myfile = txt_data_processing.Data_File('swept_sine_kp_1_amp_50.txt')
#myfile.bode_plot('v','theta',seedfreq=2.45, seedphase=-200.0, label='exp')
#myfile.bode_plot('v','a',fignum=2, seedfreq=10.0, seedphase=-150.0, label='exp')
myfile.bode_plot(*args1, **kwargs1)
myfile.bode_plot(*args2, **kwargs2)

kwargs1['clear'] = False
kwargs2['clear'] = False

myfile2 = txt_data_processing.Data_File('with_accel_01_28_14_test_1.txt')
kwargs1['label'] = 'with accel'
kwargs2['label'] = 'with accel'
args2b = ('v','accel')
myfile2.bode_plot(*args1, **kwargs1)
myfile2.bode_plot(*args2b, **kwargs2)

myfile3 = txt_data_processing.Data_File('without_accel_01_28_14_test_2.txt')
kwargs1['label'] = 'no accel'
myfile3.bode_plot(*args1, **kwargs1)

import pylab_util as PU

PU.SetLegend(1, loc=3, axis=0)
PU.SetPhaseLim(1,[-260,180])
PU.SetFreqLim(1,[0.1,50])
PU.SetMagLim(1,[-50,30])

PU.mysave('with_and_without_accel_attached.png', fi=1)

show()
