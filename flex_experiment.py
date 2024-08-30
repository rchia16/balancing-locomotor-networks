from os import makedirs
from os.path import join
import matplotlib.pyplot as plt
import numpy as np

import pdb

from argparse import ArgumentParser

from brian2 import *
from brian2cuda import *

from biophysical_parameters import *
from utils import *

import time
import warnings
warnings.simplefilter('ignore')

'''
------
 ARGS
------
'''

parser = ArgumentParser()
parser.add_argument('--run_id', type=int, default=0,
                    help='Unique number for run identification')
parser.add_argument('--stim_w', type=float, default=0,
                    help='Stim in mV for poisson input to axons')
parser.add_argument('--stim_freq', type=float, default=0,
                    help='Stim frequency for poisson input to axons')
parser.add_argument('--is_bws', type=int, default=0,
                    choices=[0, 1],
                    help='Offset variable for GM Ia and II afferent firing '\
                    'rates. Either 0 or 0.6. 0 being no offset and 0.6 being '\
                    'reduced EMG activity by 60%. (x - offset*K*EMGnorm)')
parser.add_argument('--use_quip', type=int, default=0,
                    choices=[0, 1],
                    help='Use quipazine or not')
parser.add_argument('--is_sci', type=int, default=0,
                    choices=[0, 1],
                    help='Activate SCI GABA settings')
parser.add_argument('--is_spastic', type=int, default=0,
                    choices=[0, 1],
                    help='Activate Spasticity GABA settings')
parser.add_argument('--debug', action='store_true',
                    help='save or not save run or make directories')

args = parser.parse_args()

t0 = time.time()

run_id = str(args.run_id).zfill(3)
is_sci = args.is_sci
is_spastic = args.is_spastic
debug = args.debug

# GM Ia Settings
is_bws = args.is_bws

# Quipazine Settings
use_quip = args.use_quip

"""
===============
 SYSTEM PARAMS
===============
"""
plt.close('all')
plt.rcParams.update({'axes.titlesize': 14,
                     'axes.labelsize': 12})

results_dir = '/data/rqchia/snn_results/experiment_log'
makedirs(results_dir, exist_ok=True)

output_dir = '/scratch/rqchia/brian2/output'
makedirs(output_dir, exist_ok=True)
set_device('cpp_standalone', directory=output_dir)

# output_dir = '/scratch/rqchia/brian2cuda/output'
# makedirs(output_dir, exist_ok=True)
# set_device('cuda_standalone', directory=output_dir)
defaultclock.dt = 0.005*ms
devices.device.seed(42)
np.random.seed(42)

def create_stim_profiles(perc, N, stim_period=1, pulse_width=0.5*ms,
                         shuffle=True):
    # duration length indexes

    def shuffle_along_axis(a, axis):
        idx = np.random.rand(*a.shape).argsort(axis=axis)
        return np.take_along_axis(a,idx,axis=axis)
    # stim period in seconds, pulse_width in ms
    perc = np.abs(perc)
    perc /= 100
    if perc > 1:
        perc = 1.
    stim_N = int(perc*N)

    idx_to_sec = int( np.ceil(1/defaultclock.dt) )

    stim_pcl = [0] + [1] + [0]*int((stim_period*second/pulse_width) - 2)

    rest_pcl = [0]*len(stim_pcl)

    stim_pcl = np.repeat(np.array(stim_pcl).reshape(-1, 1), stim_N, axis=1)
    rest_pcl = np.repeat(np.array(rest_pcl).reshape(-1, 1), N - stim_N,
                            axis=1)

    stim_pcl = np.concatenate((stim_pcl, rest_pcl), axis=1)
    if shuffle:
        stim_pcl = shuffle_along_axis(stim_pcl, 1)

    return stim_pcl

"""
=======================
 Experiment Parameters
=======================
"""
rat_gait = read_gait_profile('input-files/ratGaitCycles.p')
duration = 1.17*second # gait cycle duration
gauss_width = 25*ms
bg_noise = 0.001
if debug:
    duration = rat_gait[1]*second
else:
    duration = rat_gait[-1]*second

# PSI settings
if is_sci == 1:
    # Khristy W et al 2009
    gaba_mn_k *= gaba_sci_scale
elif is_spastic == 1:
    gaba_mn_k /= gaba_sci_scale

bws_ratio = 0.6
# multiplier to a normalised EMG component
if is_bws == 1:
    # extensors 3.6km/h (40% BWS)
    gm_offset = gm_offset_dict[bws_ratio]
    ta_offset = ta_offset_dict[bws_ratio]
else:
    gm_offset = 0
    ta_offset = 0

kappa = 5 # target rate
p_init = np.clip(1 - psi_beta*kappa, 0, 1)
gaba_w_gain = 1
gaba_mn_n = gaba_params['N']/((2*gaba_mn_k)-1)

# gain for gaba conductance (how much is released per spike)
gaba_g_gain = gaba_w_gain/gaba_mn_n

'''
------------
 STIM INPUT 
------------
'''
stim_w = args.stim_w*mV # 5.0
stim_freqs = [0, 20, 40, 60, 80, 100]
stim_freq = args.stim_freq
if stim_freq == 0:
    stim_period = 1
else:
    stim_period = 1/stim_freq
pulse_width = 0.2*ms

use_poisson = True

ia_stim = TimedArray(np.zeros((1, axon_params['N']))*pA, dt=pulse_width)
ii_stim = TimedArray(np.zeros((1, axon_params['N']))*pA, dt=pulse_width)
mn_stim = TimedArray(np.zeros((1, mn_params['N']))*pA, dt=pulse_width)

amplitudes = [I_TH*stim_w/mV]
percIf  = set_fibre_rec_curve('ia', amplitudes)
percIIf = set_fibre_rec_curve('ii', amplitudes)
percMn  = set_fibre_rec_curve('mta', amplitudes)

idx_to_sec = int(np.ceil( (1/defaultclock.dt)/Hz ))
scale = int(np.ceil(pulse_width*idx_to_sec)/second)
pw_to_sec = int( (1/pulse_width)/Hz )
exo_stim_single = [1] + [0]*int(pw_to_sec*stim_period - 1)
exo_stim = np.tile(
    np.repeat(exo_stim_single, scale),
    len(amplitudes)
)

# Supra-threshold primary and secondary afferent recruitment stimulation
if not use_poisson:
    stim_scale = np.ceil(duration/pulse_width).astype(int)

    profile_args = (percIf, axon_params['N'])
    profile_kwargs = {'stim_period': stim_period,
                      'pulse_width': pulse_width}

    ia_stim_list = [
        create_stim_profiles(percIf, axon_params['N'], **profile_kwargs)
        for i in range(stim_scale)
    ]
    ii_stim_list = [
        create_stim_profiles(percIIf, axon_params['N'], **profile_kwargs)
        for i in range(stim_scale)
    ]
    mn_stim_list = [
        create_stim_profiles(percMn, mn_params['N'], **profile_kwargs)
        for i in range(stim_scale)
    ]

    ia_stim_pcl = np.concatenate(ia_stim_list, axis=0)
    ii_stim_pcl = np.concatenate(ii_stim_list, axis=0)
    mn_stim_pcl = np.concatenate(mn_stim_list, axis=0)

    thr = 70*uA
    thr_perc = 0.8

    ia_stim = TimedArray(ia_stim_pcl*thr*thr_perc, dt=pulse_width)
    ii_stim = TimedArray(ii_stim_pcl*thr*thr_perc, dt=pulse_width)
    mn_stim = TimedArray(mn_stim_pcl*thr*thr_perc, dt=pulse_width)

    print('ia {0}: {1}'.format(int(percIf[0]), ia_stim_pcl.sum(axis=0)))
    print('ii {0}: {1}'.format(int(percIIf[0]), ii_stim_pcl.sum(axis=0)))

poisson_stim_scale = 3
poisson_stim = PoissonGroup(int(axon_params['N']//poisson_stim_scale), 
                            stim_freq*Hz)

'''
---------
 CONFIGS
---------
'''
cfg_dict = {
    'duration'    : duration/second,
    'is_sci'      : is_sci,
    'is_bws'      : is_bws,
    'use_quip'    : use_quip,
    'beta'        : psi_beta,
    'kappa'       : kappa,
    'gaba_mn_n'   : gaba_mn_n,
    'p_ax_gaba'   : p_ax_gaba,
    'gaba_w_gain' : gaba_w_gain,
    'stim_w'      : args.stim_w,
    'stim_freq'   : stim_freq,
    'gauss_width' : gauss_width/ms,
    'p_ax_in'     : p_ax_in,
    'p_ax_v2a'    : p_ax_v2a,
    'p_iaIN'      : p_iaIN,
    'p_in_mn'     : p_in_mn,
}

"""
===============
 NeuronGroups 
===============
"""

"""
------------
 AXON GROUP 
------------
"""

# synapse delay of 2ms
ax_N       = axon_params['N']
ax_Vth     = axon_params['Vth']
ax_Vr      = axon_params['V_reset']
ax_tau_ref = axon_params['tau_ref']
ax_ia_area = np.pi*(ia_diameter_mu/2)**2
ax_ii_area = np.pi*(ii_diameter_mu/2)**2

ax_v_mu = (ax_Vr + ax_Vth)/2
ax_v_sd = 10*mV
axon_init_v = ax_v_sd*np.random.randn(ax_N) + ax_v_mu

ax_ia_eqn = '''
dv/dt = (El - v)/tau + (I_in + i_stim + i_noise)/(Cm*ax_ia_area) \
        : volt (unless refractory)
i_noise : amp
i_stim = ia_stim(t, i) : amp
I_in : amp
'''

ax_ii_eqn = '''
dv/dt = (El - v)/tau + (I_in + i_stim + i_noise)/(Cm*ax_ii_area) \
        : volt (unless refractory)
i_noise : amp
i_stim = ii_stim(t, i) : amp
I_in : amp
'''

ta_ia_axon = NeuronGroup(ax_N, ax_ia_eqn, threshold='v>=ax_Vth',
                         reset='v=ax_Vr', method='euler',
                         refractory=ax_tau_ref,
                         namespace=axon_params)
ta_ia_axon.v = axon_init_v
ta_ia_axon.run_regularly(f"i_noise = bg_noise*randn()*pA", dt=1/fs)

gm_ia_axon = NeuronGroup(ax_N, ax_ia_eqn, threshold='v>=ax_Vth',
                         reset='v=ax_Vr', method='euler',
                         refractory=ax_tau_ref,
                         namespace=axon_params)
gm_ia_axon.v = axon_init_v
gm_ia_axon.run_regularly(f"i_noise = bg_noise*randn()*pA", dt=1/fs)

ta_ii_axon = NeuronGroup(ax_N, ax_ii_eqn, threshold='v>=ax_Vth',
                         reset='v=ax_Vr', method='euler',
                         refractory=ax_tau_ref, namespace=axon_params)
ta_ii_axon.v = axon_init_v
ta_ii_axon.run_regularly(f"i_noise = bg_noise*randn()*pA", dt=1/fs)

gm_ii_axon = NeuronGroup(ax_N, ax_ii_eqn, threshold='v>=ax_Vth',
                         reset='v=ax_Vr', method='euler',
                         refractory=ax_tau_ref,
                         namespace=axon_params)
gm_ii_axon.v = axon_init_v
gm_ii_axon.run_regularly(f"i_noise = bg_noise*randn()*pA", dt=1/fs)

"""
-----------
 v2a GROUP 
-----------
"""

## v2a
# from Touboul_Brette_2008
v2a_N = v2a_params['N']

# NOTE : change this for quip
if use_quip and is_bws:
        v2a_g_l = v2a_params['g_l_quip_lo']
elif use_quip and not is_bws:
        v2a_g_l = v2a_params['g_l_quip_hi']
else:
    v2a_g_l = v2a_params['g_l']

v2a_v_mu = (v2a_params['v_t'] + v2a_params['v_r_mu'])/2
v2a_v_sd = 10*mV
v2a_init_v = v2a_v_sd*np.random.randn(v2a_N) + v2a_v_mu

v2a_eqn = """
dv/dt = (gL*(e_l - v) + gL*d_t*exp((v-v_t)/d_t) \
        + i_stim - w)/c_m + g_syn_e*(0*mV-v)/c_m : volt
dw/dt  = (a*(v - e_l) - w)/tau_w : amp
dg_syn_e/dt = -g_syn_e/exc_alpha_tau : siemens

i_stim : amp
gL : siemens (shared, constant)
"""

v2a = NeuronGroup(
        v2a_N,
        model=v2a_eqn,
        threshold="v>0*mV",
        reset="v=v_r_mu; w+=b",
        method="euler",
        namespace=v2a_params,
)
v2a.v = v2a_init_v
v2a.w = 0
v2a.gL = v2a_g_l

"""
------------
 IaIN GROUP 
------------
"""

iaIN_N   = iaIN_params['N']
iaIN_Vth = iaIN_params['Vth']
iaIN_Vr  = iaIN_params['Vr']
iaIN_tau_ref = iaIN_params['tau_ref']

iaIN_v_mu = (iaIN_Vr + iaIN_Vth)/2
iaIN_v_sd = 5*mV
iaIN_init_v = iaIN_v_sd*np.random.randn(iaIN_N) + iaIN_v_mu
iaIN_eqn = '''
dv/dt = gL*(El - v)/Cm + (I_syn_e + I_syn_i)/Cm : volt

I_syn_e = g_syn_e * (0*mV - v) : amp
dg_syn_e/dt = -g_syn_e/exc_alpha_tau : siemens

I_syn_i = g_syn_i * (El - v) : amp
dg_syn_i/dt = (s_i - g_syn_i)/inh_alpha_tau2 : siemens
ds_i/dt = -s_i/inh_alpha_tau1  : siemens
'''
gm_iaIN = NeuronGroup(iaIN_N, iaIN_eqn, threshold='v>=iaIN_Vth', 
                      reset='v=iaIN_Vr', refractory=iaIN_tau_ref, 
                      method='euler', namespace=iaIN_params)
gm_iaIN.v = iaIN_init_v

ta_iaIN = NeuronGroup(iaIN_N, iaIN_eqn, threshold='v>=iaIN_Vth', 
                      reset='v=iaIN_Vr', refractory=iaIN_tau_ref, 
                      method='euler', namespace=iaIN_params)
ta_iaIN.v = iaIN_init_v

"""
------------
 GABA GROUP 
------------
"""
gaba_N = gaba_params['N']
gaba_Vth = gaba_params['v_t']
gaba_Vr = gaba_params['v_r']
gaba_tau_ref = gaba_params['tau_ref']
gaba_gIE = gaba_params['gIE']

gaba_v_mu = (gaba_Vr + gaba_Vth)/2
gaba_v_sd = 10*mV
gaba_init_v = gaba_v_sd*np.random.randn(gaba_N) + gaba_v_mu

gaba_eqn = """
dv/dt = (g_l*(e_l - v) + g_l*d_t*exp((v-v_t)/d_t) + i_stim - w)/c_m \
        + g_syn_e*(0*mV-v)/c_m: volt
dw/dt  = (a*(v - e_l) - w)/tau_w : amp

dg_syn_e/dt = -g_syn_e/exc_alpha_tau : siemens

i_stim : amp
"""

gaba = NeuronGroup(
        gaba_params['N'],
        model=gaba_eqn,
        threshold="v>0*mV",
        reset="v=gaba_Vr; w+=b",
        method="euler",
        namespace=gaba_params,
)
gaba.w = 0
gaba.v = gaba_init_v

"""
----------
 MN GROUP 
----------
"""

# Model as Leaky IntFire but also have alpha shape synapse for ipsp
mn_Vth     = mn_params['Vth']
mn_Vr      = mn_params['Vr']
mn_tau_ref = mn_params['tau_ref']

if use_quip and is_bws:
    mn_gL = mn_params['gL_quip_lo']
elif use_quip and not is_bws:
    mn_gL = mn_params['gL_quip_hi']
else:
    mn_gL = mn_params['gL']

mn_v_mu = (mn_Vr + mn_Vth)/2
mn_v_sd = 2*mV
mn_init_v = mn_v_sd*np.random.randn(N_mn) + mn_v_mu

# disable if PSI off
etap = 1
if psi_beta == 0:
    etap = 0
taup = 20*ms # approx from Fink et al 2015
taui = 10*ms

tau_gaba = mn_params['tau_GABA']

# exponential leaky int and fire
mn_conductance_exp_eqn = '''
I = mn_gL*(El - v + (I_noise + i_stim + I_in)*R ) + mn_gL*deltaV*exp((v - \
        mn_Vth)/deltaV) : amp

dv/dt = (I + I_syn_i) / Cm + g_syn_e*(-v)/Cm : volt 

dg_syn_e/dt = -g_syn_e/exc_alpha_tau : siemens

I_syn_GABA = g_syn_GABA * (El-10*mV - v) : amp
dg_syn_GABA/dt = (s_GABA - g_syn_GABA) / tau_GABA : siemens
ds_GABA/dt = -s_GABA/taui : siemens

dp/dt = (-p + clip(1-beta*(s_GABA/nS), 0, 1))*etap/taup : 1

I_syn_i = g_syn_i * (El-10*mV - v) : amp
dg_syn_i/dt = (s_i - g_syn_i)/inh_alpha_tau2 : siemens
ds_i/dt = -s_i/inh_alpha_tau1  : siemens

I_noise : amp
i_stim = mn_stim(t, i) : amp
I_in : amp

beta : 1 (shared, constant)

'''

mn = NeuronGroup(N_mn, mn_conductance_exp_eqn, threshold='v>0*mV', 
                 refractory='v>0*mV',
                 reset='v=mn_Vr', method='euler', namespace=mn_params)
mn.v = mn_init_v
mn.p = p_init
mn.beta = psi_beta

# Set with 0 noise or 0.5*randn()*nA
mn.run_regularly("I_noise = 0*nA", dt=1/fs) 

"""
==========
 Afferent 
==========
"""

ta_ia_data = get_afferent_signal('ta', 'ia') + ta_offset*50
ta_ia_data = np.clip(ta_ia_data, a_min=0, a_max=None)
ta_ia_signal = TimedArray(ta_ia_data*Hz, dt=1/fs)

ta_ii_data = get_afferent_signal('ta', 'ii') + ta_offset*20
ta_ii_data = np.clip(ta_ii_data, a_min=0, a_max=None)
ta_ii_signal = TimedArray(ta_ii_data*Hz, dt=1/fs)

gm_ia_data = get_afferent_signal('gm', 'ia') + gm_offset*50
gm_ia_data = np.clip(gm_ia_data, a_min=0, a_max=None)
gm_ia_signal = TimedArray(gm_ia_data*Hz, dt=1/fs)

gm_ii_data = get_afferent_signal('gm', 'ii') + gm_offset*20
gm_ii_data = np.clip(gm_ii_data, a_min=0, a_max=None)
gm_ii_signal = TimedArray(gm_ii_data*Hz, dt=1/fs)

# PoissonGroup
ta_ia_input = PoissonGroup(ax_N, rates='ta_ia_signal(t)')
gm_ia_input = PoissonGroup(ax_N, rates='gm_ia_signal(t)')
ta_ii_input = PoissonGroup(ax_N, rates='ta_ii_signal(t)')
gm_ii_input = PoissonGroup(ax_N, rates='gm_ii_signal(t)')

"""
==========
 Synapses 
==========
"""
syn_ax_w = axon_params['w']

'''
--------------------
 Afferents to Axons 
--------------------
'''

syn_ta_ia_ax = Synapses(ta_ia_input, ta_ia_axon, on_pre='v_post+=syn_ax_w')
syn_ta_ia_ax.connect(condition='abs(i-j)<2')

syn_gm_ia_ax = Synapses(gm_ia_input, gm_ia_axon, on_pre='v_post+=syn_ax_w')
syn_gm_ia_ax.connect(condition='abs(i-j)<2')

syn_ta_ii_ax = Synapses(ta_ii_input, ta_ii_axon, on_pre='v_post+=syn_ax_w')
syn_ta_ii_ax.connect(condition='abs(i-j)<2', p=0.9)

syn_gm_ii_ax = Synapses(gm_ii_input, gm_ii_axon, on_pre='v_post+=syn_ax_w')
syn_gm_ii_ax.connect(condition='abs(i-j)<2', p=0.9)

'''
--------------
 Axons to INs
--------------
'''
v2a_g_e = v2a_params['g_e']
iaIN_g_e = iaIN_params['g_e']
ax_delay_syn = 'ax_delay + 0.3*ms*randn()*i/i'

# ta ia and ii
syn_ta_ax_ii = Synapses(ta_ii_axon, v2a, on_pre='g_syn_e+=v2a_g_e')
syn_ta_ax_ii.connect(p=p_ax_v2a)
syn_ta_ax_ii.delay = ax_delay_syn

syn_ta_ii_ia = Synapses(ta_ii_axon, ta_iaIN, on_pre='g_syn_e+=iaIN_g_e/3')
syn_ta_ii_ia.connect(p=p_ax_in)
syn_ta_ii_ia.delay = ax_delay_syn

syn_ta_ax_ia = Synapses(ta_ia_axon, ta_iaIN, on_pre='g_syn_e+=iaIN_g_e')
syn_ta_ax_ia.connect(p=p_ax_in)
syn_ta_ax_ia.delay = ax_delay_syn

# gm ia and ii
syn_gm_ax_ia = Synapses(gm_ia_axon, gm_iaIN, on_pre='g_syn_e+=iaIN_g_e')
syn_gm_ax_ia.connect(p=p_ax_in)
syn_gm_ax_ia.delay = ax_delay_syn

syn_gm_ii_ia = Synapses(gm_ii_axon, gm_iaIN, on_pre='g_syn_e+=iaIN_g_e/3')
syn_gm_ii_ia.connect(p=p_ax_in)
syn_gm_ii_ia.delay = ax_delay_syn

gaba_g_e = gaba_params['g_e']
syn_gm_ax_gaba = Synapses(gm_ia_axon, gaba, on_pre='g_syn_e+=gaba_g_e')
syn_gm_ax_gaba.connect(p=p_ax_gaba)
syn_gm_ax_gaba.delay = ax_delay_syn

'''
--------------
 INs to INs
--------------
'''
iaIN_g_i = iaIN_params['g_i']
syn_taIaIN_gmIaIN = Synapses(ta_iaIN, gm_iaIN, on_pre='g_syn_i+=iaIN_g_i',
                             delay=iaIN_delay)
syn_taIaIN_gmIaIN.connect(p=p_iaIN)
syn_gmIaIN_taIaIN = Synapses(gm_iaIN, ta_iaIN, on_pre='g_syn_i+=iaIN_g_i',
                             delay=iaIN_delay)
syn_gmIaIN_taIaIN.connect(p=p_iaIN)

'''
------------
 INs to MN
------------
'''

mn_g_e = mn_params['g_e']
mn_g_i = mn_params['g_i']

syn_ta_ia_mn = Synapses(ta_ia_axon, mn, on_pre='g_syn_e+=p*mn_g_e')
syn_ta_ia_mn.connect(p=p_in_mn)
syn_ta_ia_mn.delay = ax_delay_syn

syn_ta_ii_mn = Synapses(v2a, mn, on_pre='g_syn_e+=p*mn_g_e',
                        delay=v2a_delay)
syn_ta_ii_mn.connect(p=p_in_mn) # 0.4

syn_gm_ia_mn = Synapses(gm_iaIN, mn, on_pre='g_syn_i+=mn_g_i',
                        delay=iaIN_delay)
syn_gm_ia_mn.connect(p=p_iaIN_mn) # 0.8

syn_gm_gaba_mn = Synapses(gaba, mn,
                          on_pre='s_GABA+=mn_g_i*gaba_g_gain') 
syn_gm_gaba_mn.connect(condition='abs(i-j)<gaba_mn_k', skip_if_invalid=True)

'''
--------------
 STIM CONNECT
--------------
'''
if use_poisson:
    syn_stim_ta_ia_ax = Synapses(poisson_stim, ta_ia_axon,
                                 on_pre='v+=stim_w')
    syn_stim_gm_ia_ax = Synapses(poisson_stim, gm_ia_axon,
                                 on_pre='v+=stim_w')
    syn_stim_ta_ii_ax = Synapses(poisson_stim, ta_ii_axon,
                                 on_pre='v+=stim_w')
    syn_stim_gm_ii_ax = Synapses(poisson_stim, gm_ii_axon,
                                 on_pre='v+=stim_w')

    i_c = np.repeat(np.arange(poisson_stim.N), poisson_stim_scale)
    j_c = np.arange(axon_params['N'])

    syn_stim_ta_ia_ax.connect(i=i_c, j=j_c)
    syn_stim_ta_ii_ax.connect(i=i_c, j=j_c)
    syn_stim_gm_ia_ax.connect(i=i_c, j=j_c)
    syn_stim_gm_ii_ax.connect(i=i_c, j=j_c)

"""
==========
 Monitors 
==========
"""
ax_monitor_vars = ['v', 'i_stim']
in_monitor_vars = ['v', 'g_syn_e']
iaIN_monitor_vars = ['v', 'g_syn_e', 'g_syn_i']
mn_monitor_vars = ['v', 'g_syn_e', 'g_syn_i', 's_GABA', 'p']

monitor_units_dict = {
    'v'       : 'mV',
    'g_syn_e' : 'nS',
    'g_syn_i' : 'nS',
    's_GABA'  : 'nS',
    'p'       : 'a.u',
}
idx = 0

M_ta_ia_ax = StateMonitor(ta_ia_axon, ax_monitor_vars, record=[idx])
P_ta_ia_in = PopulationRateMonitor(ta_ia_input)
P_ta_ia_ax = PopulationRateMonitor(ta_ia_axon)

M_gm_ia_ax = StateMonitor(gm_ia_axon, ax_monitor_vars, record=[idx])
P_gm_ia_in = PopulationRateMonitor(gm_ia_input)
P_gm_ia_ax = PopulationRateMonitor(gm_ia_axon)

M_ta_ii_ax = StateMonitor(ta_ii_axon, ax_monitor_vars, record=[idx])
P_ta_ii_in = PopulationRateMonitor(ta_ii_input)
P_ta_ii_ax = PopulationRateMonitor(ta_ii_axon)

M_gm_ii_ax = StateMonitor(gm_ii_axon, ax_monitor_vars, record=[idx])
P_gm_ii_in = PopulationRateMonitor(gm_ii_input)
P_gm_ii_ax = PopulationRateMonitor(gm_ii_axon)

M_ta_v2a  = StateMonitor(v2a, in_monitor_vars, record=[idx])
S_ta_v2a  = SpikeMonitor(v2a, record=True)
P_ta_v2a  = PopulationRateMonitor(v2a)

M_gm_iaIN  = StateMonitor(gm_iaIN, iaIN_monitor_vars, record=[idx])
S_gm_iaIN  = SpikeMonitor(gm_iaIN, record=True)
P_gm_iaIN  = PopulationRateMonitor(gm_iaIN)

M_ta_iaIN  = StateMonitor(ta_iaIN, iaIN_monitor_vars, record=[idx])
S_ta_iaIN  = SpikeMonitor(ta_iaIN, record=True)
P_ta_iaIN  = PopulationRateMonitor(ta_iaIN)

M_gm_gaba  = StateMonitor(gaba, in_monitor_vars, record=[idx])
S_gaba     = SpikeMonitor(gaba, record=True)
P_gm_gaba  = PopulationRateMonitor(gaba)

M_mn       = StateMonitor(mn, mn_monitor_vars, record=[idx])
S_mn       = SpikeMonitor(mn, record=True)
P_mn       = PopulationRateMonitor(mn)

run(duration)

"""
==========
 Outputs 
==========
"""

run_dir = join(results_dir, run_id)
if not debug:
    makedirs(run_dir, exist_ok=True)

units_fname = join(results_dir, 'units.pkl')

cfg_fname   = join(run_dir, 'configs.json')
emg_fname   = join(run_dir, 'emg.pkl')
m_mn_fname  = join(run_dir, 'state_mn.pkl')
s_mn_fname  = join(run_dir, 'spike_mn.pkl')
p_mn_fname  = join(run_dir, 'prate_mn.pkl')

m_ta_ia_axon_fname = join(run_dir, 'state_ta_ia_axon.pkl')
p_ta_ia_axon_fname = join(run_dir, 'prate_ta_ia_axon.pkl')

m_gm_ia_axon_fname = join(run_dir, 'state_gm_ia_axon.pkl')
p_gm_ia_axon_fname = join(run_dir, 'prate_gm_ia_axon.pkl')

m_ta_ii_axon_fname = join(run_dir, 'state_ta_ii_axon.pkl')
p_ta_ii_axon_fname = join(run_dir, 'prate_ta_ii_axon.pkl')

m_gm_ii_axon_fname = join(run_dir, 'state_gm_ii_axon.pkl')
p_gm_ii_axon_fname = join(run_dir, 'prate_gm_ii_axon.pkl')

m_taIaIN_fname = join(run_dir, 'state_taIaIN.pkl')
s_taIaIN_fname = join(run_dir, 'spike_taIaIN.pkl')
p_taIaIN_fname = join(run_dir, 'prate_taIaIN.pkl')

m_gmIaIN_fname = join(run_dir, 'state_gmIaIN.pkl')
s_gmIaIN_fname = join(run_dir, 'spike_gmIaIN.pkl')
p_gmIaIN_fname = join(run_dir, 'prate_gmIaIN.pkl')

m_gaba_fname = join(run_dir, 'state_gaba.pkl')
s_gaba_fname = join(run_dir, 'spike_gaba.pkl')
p_gaba_fname = join(run_dir, 'prate_gaba.pkl')

m_v2a_fname = join(run_dir, 'state_v2a.pkl')
s_v2a_fname = join(run_dir, 'spike_v2a.pkl')
p_v2a_fname = join(run_dir, 'prate_v2a.pkl')

m_v2a_fname = join(run_dir, 'state_v2a.pkl')
s_v2a_fname = join(run_dir, 'spike_v2a.pkl')
p_v2a_fname = join(run_dir, 'prate_v2a.pkl')


'''
-----
 EMG 
-----
'''
state_time = M_mn.t/(1000*ms)
srate = np.round(1/np.mean(np.diff(state_time)), 1)
synth_emg = np.empty(len(state_time))
synth_emg[:] = np.nan
if np.any(S_mn.count > 0):
    spike_trains = S_mn.spike_trains().copy()
    firings = map_spikes_to_matrix(spike_trains, state_time)
    synth_emg = synth_rat_emg(firings, samplingRate=srate)

'''
-------
 Plot 
-------
'''

figsize = (14, 12)
def plot_afferent_to_axon():
    fig, axs = plt.subplots(4, 3, figsize=figsize)

    axs[0,0].plot(P_ta_ia_in.t/ms, P_ta_ia_in.smooth_rate(window='gaussian', 
                                                          width=gauss_width))
    axs[0,0].set_title("ta ia poisson input (Hz)")

    axs[0,1].plot(M_ta_ia_ax.t/ms, M_ta_ia_ax.v[idx]/mV)
    axs[0,1].plot(M_ta_ia_ax.t/ms, M_ta_ia_ax.i_stim[idx]/uA)
    axs[0,1].set_title("ta ia axon0 (mV)")

    axs[0,2].plot(P_ta_ia_ax.t/ms, P_ta_ia_ax.smooth_rate(window='gaussian', 
                                                          width=gauss_width))
    axs[0,2].set_title("ta ia axon (Hz)")

    axs[1,0].plot(P_gm_ia_in.t/ms, P_gm_ia_in.smooth_rate(window='gaussian', 
                                                          width=gauss_width))
    axs[1,0].set_title("gm poisson input (Hz)")

    axs[1,1].plot(M_gm_ia_ax.t/ms, M_gm_ia_ax.v[idx]/mV)
    axs[1,1].set_title("gm ia axon0 (mV)")

    axs[1,2].plot(P_gm_ia_ax.t/ms, P_gm_ia_ax.smooth_rate(window='gaussian', 
                                                          width=gauss_width))
    axs[1,2].set_title("gm ia axon (Hz)")

    axs[2,0].plot(P_gm_ii_in.t/ms, P_gm_ii_in.smooth_rate(window='gaussian', 
                                                          width=gauss_width))
    axs[2,0].set_title("gm poisson input (Hz)")
        
    axs[2,1].plot(M_gm_ii_ax.t/ms, M_gm_ii_ax.v[idx]/mV)
    axs[2,1].plot(M_gm_ii_ax.t/ms, M_gm_ii_ax.i_stim[idx]/uA)
    axs[2,1].set_title("gm ii axon0 (mV)")

    axs[2,2].plot(P_gm_ii_ax.t/ms, P_gm_ii_ax.smooth_rate(window='gaussian', 
                                                         width=gauss_width))
    axs[2,2].set_title("gm ii axon (Hz)")

    axs[3,0].plot(P_ta_ii_in.t/ms, P_ta_ii_in.smooth_rate(window='gaussian', 
                                                          width=gauss_width))
    axs[3,0].set_title("ta ii poisson input (Hz)")

    axs[3,1].plot(M_ta_ii_ax.t/ms, M_ta_ii_ax.v[idx]/mV)
    axs[3,1].set_title("ta ii axon (mV)")

    axs[3,2].plot(P_ta_ii_ax.t/ms, P_ta_ii_ax.smooth_rate(window='gaussian', 
                                                          width=gauss_width))
    axs[3,2].set_title("ta ii axon (Hz)")

    if not debug:
        fig.savefig(join(run_dir, f"{run_id}_axon.png"))

def plot_axon_to_in():
    fig, axs = plt.subplots(4, 3, figsize=figsize)
    axs[0,0].plot(P_ta_ia_ax.t/ms, P_ta_ia_ax.smooth_rate(window='gaussian', 
                                                          width=gauss_width))
    axs[0,0].set_title("ta axon poisson input (Hz)")

    ta_v2a_vs = np.clip(M_ta_v2a[idx].v / mV, a_min=None, a_max=0)
    axs[0,1].plot(M_ta_v2a.t/ms, ta_v2a_vs)
    axs[0,1].set_title("ta v2a_0 (mV)")

    axs[0,2].plot(P_ta_v2a.t/ms, P_ta_v2a.smooth_rate(window='gaussian', 
                                                      width=gauss_width))
    axs[0,2].set_title("ta v2a (Hz)")

    axs[1,0].plot(P_ta_ii_ax.t/ms, P_ta_ii_ax.smooth_rate(window='gaussian', 
                                                          width=gauss_width))
    axs[1,0].set_title("ta ii axon (Hz)")

    axs[1,1].plot(M_ta_iaIN.t/ms, M_ta_iaIN.v[idx]/mV)
    axs[1,1].set_title("ta iaIN_0 (mV)")

    axs[1,2].plot(P_ta_iaIN.t/ms, P_ta_iaIN.smooth_rate(window='gaussian', 
                                                        width=gauss_width))
    axs[1,2].set_title("ta iaIN (Hz)")

    axs[2,0].plot(P_gm_ia_ax.t/ms, P_gm_ia_ax.smooth_rate(window='gaussian', 
                                                          width=gauss_width))
    axs[2,0].set_title("gm ia axon (Hz)")

    axs[2,1].plot(M_gm_iaIN.t/ms, M_gm_iaIN.v[idx]/mV)
    axs[2,1].set_title("gm iaIN_0 (mV)")

    axs[2,2].plot(P_gm_iaIN.t/ms, P_gm_iaIN.smooth_rate(window='gaussian', 
                                                        width=gauss_width))
    axs[2,2].set_title("gm iaIN (Hz)")

    axs[3,0].plot(P_gm_ii_ax.t/ms, P_gm_ii_ax.smooth_rate(window='gaussian', 
                                                          width=gauss_width))
    axs[3,0].set_title("gm ii axon (Hz)")

    axs[3,1].plot(M_gm_gaba.t/ms, M_gm_gaba.v[idx]/mV)
    axs[3,1].set_title("gm gaba_0 (mV)")

    axs[3,2].plot(P_gm_gaba.t/ms, P_gm_gaba.smooth_rate(window='gaussian', 
                                                        width=gauss_width))
    axs[3,2].set_title("gm gaba (Hz)")
    if not debug:
        fig.savefig(join(run_dir, f"{run_id}_IN.png"))

def plot_inputs_to_mn():
    fig = plt.figure(figsize=figsize)
    plt.subplot(321)
    plt.plot(M_mn.t/ms, M_mn.v[idx]/mV)
    plt.title('mn_0 (mV)')
    plt.subplot(322)
    mn_freq = P_mn.smooth_rate(window='gaussian', width=gauss_width)
    plt.plot(P_mn.t/ms, mn_freq)

    mask = mn_freq > 2*Hz
    mask0 = mask[1:] ^ mask[:-1]
    mask0 = np.insert(mask0, 0, 0)
    idxs = np.arange(len(mask0))[mask0]
    start_idxs = idxs[::2]
    end_idxs = idxs[1::2]

    # dt = 5*1e-5
    dt = defaultclock.dt
    f = 1/dt
    fire_rates, fire_duration = [], []
    for start, end in zip(start_idxs, end_idxs):
        burst_duration = dt*(end-start)
        if burst_duration < 0.01*second: continue

        fire_duration.append(burst_duration)

        if len(fire_rates) == 0:
            fire_rates = np.ravel(mn_freq[start:end])
        else:
            fire_rates = np.concatenate(
                (fire_rates, np.ravel(mn_freq[start:end])), axis=0
            )

        # print("{0} +- {1}".format(mn_freq[start:end].mean(),
        #                           mn_freq[start:end].std())
        #      )
    
    print("mean fire_rates: {}\t{}".format(
        np.mean(fire_rates), np.std(fire_rates)))
    print("mean fire_duration: {}\t{}".format(
        np.mean(fire_duration), np.std(fire_duration)))

    plt.title('mn pop. rate (Hz)')
    plt.subplot(323)
    plt.plot(S_mn.t/ms, S_mn.i, '.k')
    plt.title('Spikes')

    plt.subplot(324)
    plt.plot(M_mn.t/ms, M_mn.g_syn_e[idx]/mV)
    plt.title('EPSP conductance mn_0 (nS)')
    plt.subplot(325)
    plt.plot(M_mn.t/ms, M_mn.g_syn_i[idx]/mV)
    plt.title('IPSP conductance mn_0 (nS)')

    ax = plt.subplot(3,4,11)
    plt.plot(M_mn.t/ms, M_mn.s_GABA[idx]/nS)
    plt.title("s_GABA (nS)")

    plt.subplot(3,4,12)
    plt.plot(M_mn.t/ms, M_mn.p[idx])
    plt.title("p scale (a.u.)")

    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(state_time, synth_emg)
    plt.title("Synthetic EMG")

    if not debug:
        fig.savefig(join(run_dir, f"{run_id}_MN.png"))
        fig1.savefig(join(run_dir, f"{run_id}_EMG.png"))

plot_afferent_to_axon()
plot_axon_to_in()
plot_inputs_to_mn()

if debug:
    visualise_connectivity(syn_stim_gm_ii_ax)
    plt.show()

'''
-------
 WRITE 
-------
'''

if not debug:
    save_dict({'emg': synth_emg, 'time': state_time}, emg_fname)
    save_dict(monitor_units_dict, units_fname)
    save_json(cfg_dict, cfg_fname)

    def write_monitor_data(statemon, state_fname, keys,
                           spikemon, spike_fname,
                           popmon, pop_fname):
        if statemon is not None:
            save_monitor_to_pickle(statemon, state_fname, keys=keys)
        if spikemon is not None:
            save_dict(spikemon.all_values(), spike_fname)
        if popmon is not None:
            save_dict(
                {'rate': popmon.smooth_rate(window='gaussian',width=gauss_width)},
                pop_fname
            )

    write_monitor_data(M_mn, m_mn_fname, mn_monitor_vars,
                       S_mn, s_mn_fname,
                       P_mn, p_mn_fname)

    write_monitor_data(M_ta_ia_ax, m_ta_ia_axon_fname, ax_monitor_vars,
                       None, None,
                       P_ta_ia_ax, p_ta_ia_axon_fname)

    write_monitor_data(M_ta_ii_ax, m_ta_ii_axon_fname, ax_monitor_vars,
                       None, None,
                       P_ta_ii_ax, p_ta_ii_axon_fname)

    write_monitor_data(M_gm_ia_ax, m_gm_ia_axon_fname, ax_monitor_vars,
                       None, None,
                       P_gm_ia_ax, p_gm_ia_axon_fname)

    write_monitor_data(M_gm_ii_ax, m_gm_ii_axon_fname, ax_monitor_vars,
                       None, None,
                       P_gm_ii_ax, p_gm_ii_axon_fname)

    write_monitor_data(M_ta_iaIN, m_taIaIN_fname, iaIN_monitor_vars,
                       S_ta_iaIN, s_taIaIN_fname,
                       P_ta_iaIN, p_taIaIN_fname)

    write_monitor_data(M_gm_iaIN, m_gmIaIN_fname, iaIN_monitor_vars,
                       S_gm_iaIN, s_gmIaIN_fname,
                       P_gm_ia_in, p_gmIaIN_fname)

    write_monitor_data(M_ta_v2a, m_v2a_fname, in_monitor_vars,
                       S_ta_v2a, s_v2a_fname,
                       P_ta_v2a, p_v2a_fname)

    write_monitor_data(M_gm_gaba, m_gaba_fname, in_monitor_vars,
                       S_gaba, s_gaba_fname,
                       P_gm_gaba, p_gaba_fname)

print("wall time: ", time.time()-t0)
