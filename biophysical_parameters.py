from brian2 import (
    ms, second,
    cm, mm, um, meter,
    uF, pF, nF,
    mV, uvolt,
    ohm, amp, Mohm,
    nA, pA,
    Hz,
    siemens, msiemens, nS, uS
)
from numpy import pi
from utils import read_gait_profile

fs = 200*Hz
rat_gait = read_gait_profile('input-files/ratGaitCycles.p')
max_duration = rat_gait[-1]*second

''' Delays '''
iaIN_delay = 1.3*ms # source (Meehan et al, 2010)
ax_delay = 2*ms
v2a_delay = 13.67*ms # source (Hayashi et al., 2018)

''' Synapse probabilities '''
p_ax_in   = 0.3 # axon to interneurons
p_ax_v2a  = p_ax_in*2 # axon to v2a
p_iaIN    = 0.1 # recurrent iaIN
p_in_mn   = 0.3 # mn inputs
p_iaIN_mn = 0.3 # mn inputs

p_ax_gaba = 0.4

''' MEP properties '''
er_ms = [1*ms, 3*ms]
mr_ms = [4*ms, 6*ms]
lr_ms = 7*ms

''' Axon properties '''
cv = 14.5*mm/ms # harper & lawson 1985
max_firing_rate = 200
cable_len = cv*2.5*ms

ia_n_fibres = 60
ii_n_fibres = 116

ia_diameter_mu = 9*um
ia_diameter_sd = 0.2*um

ii_diameter_mu = 4.4*um
ii_diameter_sd = 0.5*um

''' Passive axon channels '''
gl_axon = 7e-3*siemens/cm**2
El_axon = -80*mV
Es_axon = 0*mV

# Nerst Na+
ENa_axon = 60*mV
# Nerst K+
EK_axon = -80*mV
# Fast Na+ conductance
gNaf_axon = 3*siemens/cm**2
# Slow/persistent Na+ conductance
gNap_axon = 0.01*siemens/cm**2
# slow K+ conductance
gK_axon = 0.08*siemens/cm**2

''' CSF '''
csf_sigma = 1.7*siemens/meter

'''
v2a INs tonic spiking
'''
N_v2a = 196
v2a_params = {
    "N": 196,
    "c_m": 45*pF,
    "g_l": 1.2 * nS, # Changed to match tau Table1 (Dougherty & Kiehn 2010)
    "g_l_quip_lo": .9 * nS, # Husch et al 2015 (reduce ~15%)
    "g_l_quip_hi": .7 * nS,
    "e_l": -53*mV,
    "v_t": -42*mV, # -40, -35
    "d_t": 0.5 * mV, # controls exponential decay
    "a": 2.0 * nS, # conductance of adaptation
    "tau_w": 55.0 * ms, # decay of adaptation
    "b": 0.0 * pA, # w change on reset
    "v_r_mu": -47*mV,
    "v_r_sd": 1*mV,
    "tau_mu": 55*ms,
    "tau_sd": 3*ms,
    "Rm": 295*Mohm,
    "g_e" : 1*nS, # Dai Y & Jordan LM 2010
}

'''
Ia Inh INs
'''
iaIN_params = {
    "N": 196,
    "tau" : 30*ms,
    "Vth" : -50*mV,
    "Vr" : -65*mV,
    "El" : -70*mV,
    "tau_ref" : 1.0*ms,
    "gL" : 5*nS, # (Bui et al., 2003)
    "Cm" : (3113*um**2)*(1*uF/cm**2), # (Hille , 2001) (Bui et al., 2003)
    "Vpk" : 5.24*mV,
    "g_e" : 7*nS, # 3*nS
    "g_i" : 3*nS, # 1*nS
}

'''
Ia Axon
'''
# 1 : 1 axon to MN connection
# 
axon_params = {
    "N": 60,
    "tau": 30*ms,
    "tau_ref": 1.6*ms, # refractory period
    "Cm" : 0.1*uF/cm**2, # myelinated capacitance
    # "Cm" : 0.1*uF/cm**2, # myelinated capacitance
    "Ci" : 2*uF/cm**2, # unmyelinated capacitance
    "Ri" : 70*ohm*cm, # 
    'gl' : 7e-3*siemens/cm**2,
    'El' : -80*mV,
    'Vth': -60*mV,
    'V_reset': -70*mV,
    'w' : 7*mV,
}

'''
GABA INs
(Macdonald, Rogers and Twyman, 1989)
(Fink et al 2014) tau ~114ms , Vpk~112uV
'''

# Default GABA
psi_beta = 0.4
gaba_mn_k = 4 # 4.5
gaba_sci_scale = 1.6

gaba_params = {
    "N": 196,
    "c_m": 100*pF, # 200
    "g_l": 1.2 * nS,
    "e_l": -70*mV,
    "v_t": -50*mV,
    "d_t": 2.0 * mV, # controls exponential decay
    "a": 2.0 * nS, # conductance of adaptation
    "tau_w": 20.0 * ms, # decay of adaptation, up to 50Hz (Fink)
    "b": 0.0 * pA, # w change on reset
    "v_r": -62.3*mV,
    "tau_ref": 55*ms,
    "gIE" : 0.0272*nS, # (Macdonald et al 1989)
    "g_e" : 12*nS,
}

'''
MN variables
'''
N_mn = 169
v_mn_reset = -65*mV
El_mn = -75*mV # reverse potential
Vth_mn = -50*mV
tau_mn = 6*ms
tau_ref_mn = 20*ms
# Table 4 Caillet et al., 2022
R_mn = 7.1e9*((tau_mn/ms)*1e-3)**1.64*ohm
mn_params = {
    "N": 169,
    "Vr" : -65*mV,
    "El" : -75*mV,
    "Vth": -50*mV,
    "Vbuff": -58*mV,
    "deltaV" : 0.05*mV,
    "tau": 6*ms,
    "tau_ref": 2*ms,
    "R" : 7.1e9*((tau_mn/ms)*1e-3)**1.64*ohm,
    "V_I" : gaba_params['e_l'],
    "tau_GABA" : 10*ms,
    "gL" : 27*nS, # (Ozyurt M G et al 2022, weaned average)
    "gL_quip_lo" : 22*nS, # Booth et al 1997 (~15%)
    "gL_quip_hi" : 16*nS, #  (~40%)
    "Cm" : 27*nS*6*ms,
    # --> 0.2 - 0.8uS (Gustafsson Pinter 1984, Delestree N et al., 2014)
    'g_e' : 0.03*uS,
    'g_i' : 0.01*uS, # (Branchereau P, et al., 2019 [eLife])
}

'''
conductance synapses
'''


'''
alpha-synapse
'''
exc_alpha_tau = 0.25*ms
inh_alpha_tau1 = 2*ms
inh_alpha_tau2 = 4.5*ms

ia_exp_epsp = 0.212*mV # (Harrison and Taylor, 1981)
ia_exp_ipsp = 0.052*mV
ii_exp_epsp = 0.33*ia_exp_epsp

mn_Vth = abs(mn_params['Vth'])
ia_exp_epsp = 0.014*mn_Vth # (Harrison and Taylor, 1981)
ia_exp_ipsp = 0.002*mn_Vth
ii_exp_epsp = 0.007*mn_Vth

'''
BWS Offset
'''
# Kristiansen M et al 2019
gm_offset_dict = {0.8: -0.319,
                  0.6: -0.600,
                  0.4: -0.676}
ta_offset_dict = {0.8: -0.171,
                  0.6: -0.122,
                  0.4: -0.221}
