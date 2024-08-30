import pandas as pd
import numpy as np
from scipy.interpolate import splev, splrep
from glob import glob
import pickle
import json
import matplotlib.pyplot as plt
import random as rnd
from brian2 import second

I_TH = 250

def read_recruitment_file(fname):
    return np.loadtxt(fname)

def read_afferent_file(fname):
    with open(fname, 'r') as f:
        data = f.readlines()
    return np.asarray(data, dtype=np.float32)

def read_gait_profile(fname):
    return pd.read_pickle(fname)

def get_afferent_signal(muscle, afferent, animal='rat'):
    afferent_files = sorted(glob('input-files/*.txt'))

    afferent_files = [f for f in afferent_files if animal in f.lower()]

    afferent_file = [f for f in afferent_files if muscle in f.lower() and
                     afferent in f.lower()].pop()
    data = read_afferent_file(afferent_file)
    return data

def get_recruitment_data(muscle, afferent, animal='rat'):
    recruit_files = sorted(glob('recruitmentData/*S1*'))
    if afferent == 'ia': afferent = 'S1'
    recruit_files = [f for f in recruit_files if animal in f.lower()]
    mu_files = [f for f in recruit_files if muscle == f.split('_')[1].lower()]
    recruit_file = [f for f in mu_files if afferent == f.split('_')[3]].pop()

    data = read_recruitment_file(recruit_file)
    return data

def get_gait_data(muscle, afferent, animal='rat'):
    raise NotImplementedError
    gait_files = sorted(glob('input-files/*.p'))
    gait_files = [f for f in gait_files if animal in f.lower()]
    return gait_file

def save_monitor_to_pickle(monitor, fname, keys=['t']):
    data = monitor.get_states(keys, units=False)
    with open(fname, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_dict(m_dict, fname):
    with open(fname, 'wb') as f:
        pickle.dump(m_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_json(m_dict, fname):
    with open(fname, 'w') as f:
        json.dump(m_dict, f)

def get_ind_diff(signal_inds):
    ''' Now to offset all active indices and subtract from above array to
    determine where the gap between the bursts are '''
    diff = signal_inds[1::] - signal_inds[0:-1]

    # from the active indexes, where are the gaps along the signal in
    # indices? Note, now working relative to signal_inds
    d_mask = diff>1 # len(d_mask) = len(sig) - 1, we skipped 0
    return diff, d_mask

def get_rising_edge(signal_inds):
    i0 = signal_inds[0]
    i1 = signal_inds[-1]

    diff, d_mask = get_ind_diff(signal_inds)

    rising = signal_inds[1::].copy()
    rising = rising[d_mask]
    rising = np.insert(rising, 0, i0)
    return rising

def get_falling_edge(signal_inds):
    i0 = signal_inds[0]
    i1 = signal_inds[-1]

    diff, d_mask = get_ind_diff(signal_inds)

    falling = signal_inds[0:-1].copy()
    falling = falling[d_mask]
    falling = np.append(falling, i1)
    return falling

def set_fibre_rec_curve(afferent, amplitudes):
    if afferent in ['mta', 'mgm']:
        ta_rec = get_recruitment_data(afferent, 'ia')
        gm_rec = get_recruitment_data(afferent, 'ia')
    else:
        ta_rec = get_recruitment_data('ta', afferent)
        gm_rec = get_recruitment_data('gm', afferent)

    current_range = np.linspace(0, 600, 20)
    tck = splrep(current_range, (ta_rec+gm_rec)/2)

    perc = []
    for amp in amplitudes:
        perc.append(splev(amp, tck))
    return perc

def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(np.zeros(Ns), np.arange(Ns), 'ok', ms=5)
    plt.plot(np.ones(Nt), np.arange(Nt), 'ok',  ms=5)
    for i, j in zip(S.i, S.j):
        plt.plot([0, 1], [i, j], '-k')
    plt.xticks([0, 1], ['Source', 'Target'])
    plt.ylabel('Neuron index')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-1, max(Ns, Nt))
    plt.subplot(122)
    plt.plot(S.i, S.j, 'ok')
    plt.xlim(-1, Ns)
    plt.ylim(-1, Nt)
    plt.xlabel('Source neuron index')
    plt.ylabel('Target neuron index')

def map_spikes_to_matrix(spike_trains:dict, time):
    n_cells = len(spike_trains.keys())
    n_samples = len(time)
    matrix = np.zeros((n_cells, n_samples))

    df_time = pd.DataFrame(range(len(time)), index=time, columns=['time'])
    for cell, spikes in spike_trains.items():
        spikes = spikes/second
        df_spikes = pd.DataFrame(range(len(spikes)), index=spikes,
                                 columns=['spikes'])
        pair_idxs = df_time.join(df_spikes, how='inner')
        matrix[cell, pair_idxs['time']] = 1
        
        assert matrix.sum(axis=1)[cell] == len(spikes), \
                "Matrix cell sum does not match number of spikes"
    return matrix

def synth_rat_emg(firings, samplingRate=20000., delay_ms=2):
    """ Formento et al., 2018
    Return the EMG activity given the cell firings.

    Keyword arguments:
        firings -- Cell firings, a 2d numpy array (nCells x time).
        samplingRate -- Sampling rate of the extracted signal in Hz 
        (default = 20000).
        delay_ms -- delay in ms between an action potential (AP) and a motor 
        unit action potential (MUAP).
    """
    EMG = None
    nCells = firings.shape[0]
    nSamples = firings.shape[1]

    logBase = 1.05
    dt = 1000./samplingRate
    delay = int(delay_ms/dt)

    # MUAP duration between 5-10ms (Day et al 2001) -> 7.5 +-2
    meanLenMUAP = int(7.5/dt)
    stdLenMUAP  = int(2/dt)

    nS = [int(meanLenMUAP + rnd.gauss(0,stdLenMUAP)) for i in
          range(nCells)]

    Amp = [abs(1+rnd.gauss(0,0.2)) for i in range(firings.shape[0])]
    EMG = np.zeros(nSamples + max(nS)+delay);

    # create MUAP shape
    for i in range(nCells):
        n40perc = int(nS[i]*0.4)
        n60perc = nS[i] - n40perc
        amplitudeMod = (1-(np.linspace(0,1,nS[i])**2)) \
                * np.concatenate(
                    (np.ones(n40perc),1/np.linspace(1,3,n60perc))
                )
        freqMod = np.log(np.linspace(1,logBase**(4*np.pi),nS[i])) \
                / np.log(logBase)
        EMG_unit = Amp[i]*amplitudeMod*np.sin(freqMod)
                # + np.random.normal(0, 0.5);
        for j in range(nSamples):
            # for each sample, check if firing.
            # TODO: adjust so that each value is a point in time where an event
            # occurred
            if firings[i,j]==1:
                sect = np.arange(j+delay, j+delay+nS[i])
                EMG[sect] += EMG_unit
    EMG = EMG[:nSamples]
    return EMG

def mn_burst_period(mn_freq, dt, min_freq=20, min_burst=0.08):
    # thold: burst threshold in seconds
    mask = mn_freq > min_freq
    mask0 = mask[1:] ^ mask[:-1]
    mask0 = np.insert(mask0, 0, 0)
    idxs = np.arange(len(mask0))[mask0]

    start_idxs = idxs[::2]
    end_idxs = idxs[1::2]

    f = 1/dt
    fire_rates, max_fire_rates, fire_duration = [], [], []
    for start, end in zip(start_idxs, end_idxs):
        burst_duration = dt*(end-start)
        if burst_duration < min_burst: continue

        fire_duration.append(burst_duration)

        if len(fire_rates) == 0:
            fire_rates = np.ravel(mn_freq[start:end])
        else:
            fire_rates = np.concatenate(
                (fire_rates, np.ravel(mn_freq[start:end])), axis=0
            )
        max_fire_rates.append(mn_freq[start:end].max())

        # print("{0} +- {1}".format(mn_freq[start:end].mean(),
        #                           mn_freq[start:end].std())
        #      )
    
    print("mean fire_rates: {}\t{}".format(
        np.mean(fire_rates), np.std(fire_rates)))
    print("mean max fire_rates: {}\t{}".format(
        np.mean(max_fire_rates), np.std(max_fire_rates)))
    print("mean fire_duration: {}\t{}".format(
        np.mean(fire_duration), np.std(fire_duration)))
    return fire_rates, fire_duration
