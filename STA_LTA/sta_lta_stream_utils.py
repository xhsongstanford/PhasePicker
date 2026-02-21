import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy import read
from obspy import Stream, Trace
import os
from pathlib import Path
from glob import glob

####################################################################################
#####       STA_LTA DETECTION
####################################################################################



def SNR_Rough(tr_square_rough, dt, nss_win = 10, sgn_win = 4):
    
    nss_len = int((nss_win/dt))
    sgn_len = int((sgn_win/dt))
    snr = tr_square_rough*0

    idx = nss_len
    nss_2 = np.sum(tr_square_rough[idx - nss_len : idx])
    sgn_2 = np.sum(tr_square_rough[idx : idx + sgn_len])
    
    for idx in range(nss_len, len(tr_square_rough) - sgn_len - 1):
        if nss_2 <= 2e-17:
            snr[idx] = 0
        else:
            snr[idx] = np.sqrt(sgn_2/nss_2 * nss_win/sgn_win)
        nss_2 += tr_square_rough[idx] - tr_square_rough[idx - nss_len]
        if nss_2 <= 0:
            nss_2 = np.sum(tr_square_rough[idx - nss_len : idx]**2)
        sgn_2 += tr_square_rough[idx + sgn_len] - tr_square_rough[idx]
        if sgn_2 <= 0:
            sgn_2 = np.sum(tr_square_rough[idx : idx + sgn_len]**2)

    return snr

def window_length_relation(t, win_min, win_max, rate):
    if t <= win_min:
        return win_min
    else:
        return min(win_min + (t - win_min) * rate, win_max) # rate should be smaller than 1


def SNR_Fine(tr_data, dt, nss_win = 10, sgn_win = 2, adj_nss_win = False, nss_win_min = 7, nss_win_max = 15, rate_nss = 0.2, adj_sgn_win = False, sgn_win_min = 3, sgn_win_max = 6, rate_sgn = 0.1, NET='XZ'):

    Empty_warning = False

    snr = tr_data * 0
    nss_len = int(round(nss_win/dt))
    sgn_len = int(round(sgn_win/dt))
    idx_start = int(round(nss_win/dt))
    idx_end = len(tr_data) - int(round(sgn_win/dt))
    
    if adj_nss_win:
        idx_start = int(round(nss_win_min/dt))
    if adj_sgn_win:
        idx_end = len(tr_data) - int(round(sgn_win_max/dt))

    epsilon = 3e-10

    for idx in range(idx_start, idx_end-1):

        if adj_nss_win:
            nss_len = int(round(window_length_relation(idx * dt, nss_win_min, nss_win_max, rate_nss)/dt))
        if adj_sgn_win:
            sgn_len = int(round(window_length_relation(idx * dt, sgn_win_min, sgn_win_max, rate_sgn)/dt))

        nss = tr_data[idx - nss_len : idx]
        sgn = tr_data[idx : idx + sgn_len]
        Noise = np.sum(nss**2 * np.linspace(0.5, 1.0, len(nss)) )
        Signal = np.sum(sgn**2 * np.linspace(1.0, 0.5, len(sgn)) )
        #sgn = sgn * np.append(np.linspace(0.5, 1.0, int(0.2*len(sgn))), np.linspace(1.0, 0.5, len(sgn) - int(0.2*len(sgn)) ))
        #Noise = np.sum(nss**2)
        if NET == 'AM':
            epsilon = 2e-8
        if np.sqrt(Noise) <= epsilon or np.sqrt(Signal) <= 2 * epsilon:
            snr[idx] = 0
            Empty_warning = True
        else:   
            snr[idx] = np.sqrt(Signal/Noise * nss_win/sgn_win)

    return snr, Empty_warning


####################################################################################
#####       get picks from SNR array
####################################################################################

def pick_rough_picks(snr_rough, dt_rough, threshold_rough=3,):

    last_rough_pick = -100
    last_rough_snr = 0

    rough_picks = []
    for i in range(len(snr_rough)):
        if snr_rough[i] > threshold_rough:
            if i*dt_rough - last_rough_pick > 20:
                rough_picks.append(i*dt_rough)
                last_rough_pick = i*dt_rough
                last_rough_snr = snr_rough[i]
            elif i*dt_rough - last_rough_pick < 5 and snr_rough[i] > last_rough_snr:
                rough_picks[-1] = i*dt_rough
                last_rough_pick = i*dt_rough
                last_rough_snr = snr_rough[i]
    rough_picks = np.array(rough_picks)

    return rough_picks


def pick_fine_picks(rough_picks, st_high, st_low, st_amp, threshold_p=3, threshold_s=2.5,):

    dt_fine = st_high[0].meta.delta

    p_picks = []
    p_snrs = []
    p_ampls_z = []
    p_ampls_n = []
    p_ampls_e = []

    p_nois_z = []
    p_nois_n = []
    p_nois_e = []

    p_rms_z = []
    p_rms_n = []
    p_rms_e = []

    p_warning = []
    for rough_pick in rough_picks:

        i_start = int(round((rough_pick - 15)/dt_fine))
        i_end   = int(round((rough_pick + 15)/dt_fine))
        time_start = i_start * dt_fine

        snr_p = Stream([])
        Warning3 = []

        for i in range(len(st_high)):

            tr = st_high[i]
            snr_p_data, Empty_warning = SNR_Fine(tr.data[i_start:i_end], dt_fine, 8, 2, NET=tr.meta.network)
            snr_p.append(Trace(data = snr_p_data, header={'delta':dt_fine, 'channel':tr.meta.channel}))
            Warning3.append(Empty_warning)

        Empty_warning = np.bool_(np.sum(Warning3))
        #print(snr_p_data)

        snr_p_avg = merge_snr(snr_p).data

        #snr_p_avg = snr_p_data

        i_p = -1

        if len(snr_p_avg) == 0:
            continue

        for i in range(len(snr_p_avg)):
            if snr_p_avg[i] > threshold_p and i_p == -1:
                i_p = i
                last_p_snr = snr_p_avg[i]
            if i_p != -1 and i*dt_fine - i_p*dt_fine < 5 and snr_p_avg[i] > last_p_snr:
                i_p = i
                last_p_snr = snr_p_avg[i]
            
        if i_p != -1:
            p_picks.append(i_p*dt_fine + time_start)
            p_snrs.append(last_p_snr)
            p_warning.append(Empty_warning)
            findz = 0
            findn = 0
            finde = 0
            for tr in st_amp:
                if tr.meta.channel[-1] == 'Z':
                    p_ampls_z.append( np.max(np.abs(tr.data[i_start+i_p-int(1/dt_fine): i_start+i_p+int(5/dt_fine)])) )
                    p_nois_z.append( np.sqrt(np.sum(tr.data[i_start+i_p-int(5/dt_fine): i_start+i_p-int(1/dt_fine)] ** 2)/(4/dt_fine)) )
                    p_rms_z.append( np.sqrt(np.sum(tr.data[i_start+i_p-int(0/dt_fine): i_start+i_p-int(4/dt_fine)] ** 2)/(4/dt_fine)) )
                    findz = 1
                elif tr.meta.channel[-1] in ['N' , '1'] :
                    p_ampls_n.append( np.max(np.abs(tr.data[i_start+i_p-int(1/dt_fine): i_start+i_p+int(5/dt_fine)])) )
                    p_nois_n.append( np.sqrt(np.sum(tr.data[i_start+i_p-int(5/dt_fine): i_start+i_p-int(1/dt_fine)] ** 2)/(4/dt_fine)) )
                    p_rms_n.append( np.sqrt(np.sum(tr.data[i_start+i_p-int(5/dt_fine): i_start+i_p-int(4/dt_fine)] ** 2)/(4/dt_fine)) )
                    findn = 1
                elif tr.meta.channel[-1] in ['E' , '2']:
                    p_ampls_e.append( np.max(np.abs(tr.data[i_start+i_p-int(1/dt_fine): i_start+i_p+int(5/dt_fine)])) )
                    p_nois_e.append( np.sqrt(np.sum(tr.data[i_start+i_p-int(5/dt_fine): i_start+i_p-int(1/dt_fine)] ** 2)/(4/dt_fine)) )
                    p_rms_e.append( np.sqrt(np.sum(tr.data[i_start+i_p-int(0/dt_fine): i_start+i_p-int(4/dt_fine)] ** 2)/(4/dt_fine)) )
                    finde = 1
            if not findz:
                p_ampls_z.append(0)
                p_nois_z.append(0)
                p_rms_z.append(0)
            if not findn:
                p_ampls_n.append(0)
                p_nois_n.append(0)
                p_rms_n.append(0)
            if not finde:
                p_ampls_e.append(0)
                p_nois_e.append(0)
                p_rms_e.append(0)
            
            if len(p_picks) >= 2:
                if p_picks[-1] - p_picks[-2] < 100:
                    old_p_amps = [p_rms_z[-2], p_rms_n[-2], p_rms_e[-2]]
                    # old_p_nois = [p_nois_z[-2], p_nois_n[-2], p_nois_e[-2]]
                    # new_p_amps = [p_rms_z[-1], p_rms_n[-1], p_rms_e[-1]]
                    new_p_nois = [p_nois_z[-1], p_nois_n[-1], p_nois_e[-1]]
                    i_chn = np.argmax([p_rms_z[-2], p_rms_n[-2], p_rms_e[-2]])
                    if new_p_nois[i_chn] > 0.4 * old_p_amps[i_chn]:
                        p_picks.pop()
                        p_snrs.pop()
                        p_warning.pop()
                        p_rms_z.pop()
                        p_rms_n.pop()
                        p_rms_e.pop()
                        p_nois_n.pop()
                        p_nois_e.pop()
                        p_nois_z.pop()
                    

    #print('finish p')

    s_picks = []
    s_snrs = []
    s_ampls_z = []
    s_ampls_n = []
    s_ampls_e = []
    s_nois_z = []
    s_nois_n = []
    s_nois_e = []
    s_rms_z = []
    s_rms_n = []
    s_rms_e = []
    s_warning = []
    sp_ratio = []
    for k in range(len(p_picks)):

        p_pick = p_picks[k]
        if p_warning[k] == True:
            continue

        i_start = int(round((p_pick +   0)/dt_fine))
        i_end   = int(round((p_pick + 110)/dt_fine))
        time_start = i_start * dt_fine

        snr_s = Stream([])

        Warning3 = []

        for i in range(len(st_low)):

            tr = st_low[i]
            snr_s_data, Empty_warning = SNR_Fine(tr.data[i_start:i_end], dt_fine, adj_nss_win=True, adj_sgn_win=True, NET=tr.meta.network)
            snr_s.append(Trace(data = snr_s_data, header={'delta':dt_fine, 'channel':tr.meta.channel}))
            Warning3.append(Empty_warning)
        
        Empty_warning = np.bool_(np.sum(Warning3))

        
        snr_s_avg = merge_snr(snr_s).data

        if len(snr_s_avg) == 0:
            continue

        i_s = []
        s_snr = []
        for i in range(len(snr_s_avg)):
            if snr_s_avg[i] > threshold_s:
                if i_s == [] or i*dt_fine - i_s[-1]*dt_fine > 10:
                    i_s.append(i)
                    s_snr.append(snr_s_avg[i])
                if i_s != [] and i*dt_fine - i_s[-1]*dt_fine < 5 and snr_s_avg[i] > s_snr[-1]:
                    i_s[-1] = i
                    s_snr[-1] = snr_s_avg[i]

                

            if len(s_snr) >= 3:
                break

        # i_s = [np.argmax(snr_s_avg)]
        # s_snr = [snr_s_avg[i_s[-1]]]
        # if s_snr[-1] < threshold_s:
        #     continue
        
        if i_s != []:
            for j in range(len(i_s)):
                s_picks.append(i_s[j]*dt_fine + time_start)
                findz = 0
                findn = 0
                finde = 0
                for tr in st_amp:
                    if tr.meta.channel[-1] == 'Z':
                        s_ampls_z.append( np.max(np.abs(tr.data[i_start+i_s[j]-int(1/dt_fine): i_start+i_s[j]+int(2/dt_fine)])) )
                        s_nois_z.append( np.sqrt(np.sum(tr.data[i_start+i_s[j]-int(6/dt_fine): i_start+i_s[j]-int(1/dt_fine)] ** 2)/(5/dt_fine)) )
                        s_rms_z.append( np.sqrt(np.sum(tr.data[i_start+i_s[j]-int(0/dt_fine): i_start+i_s[j]-int(5/dt_fine)] ** 2)/(5/dt_fine)) )
                        findz = 1
                    elif tr.meta.channel[-1] in ['N', '1'] :
                        s_ampls_n.append( np.max(np.abs(tr.data[i_start+i_s[j]-int(1/dt_fine): i_start+i_s[j]+int(2/dt_fine)])) )
                        s_nois_n.append( np.sqrt(np.sum(tr.data[i_start+i_s[j]-int(6/dt_fine): i_start+i_s[j]-int(1/dt_fine)] ** 2)/(5/dt_fine)) )
                        s_rms_n.append( np.sqrt(np.sum(tr.data[i_start+i_s[j]-int(0/dt_fine): i_start+i_s[j]-int(5/dt_fine)] ** 2)/(5/dt_fine)) )
                        findn = 1
                    elif tr.meta.channel[-1] in ['E', '2']:
                        s_ampls_e.append( np.max(np.abs(tr.data[i_start+i_s[j]-int(1/dt_fine): i_start+i_s[j]+int(2/dt_fine)])) )
                        s_nois_e.append( np.sqrt(np.sum(tr.data[i_start+i_s[j]-int(6/dt_fine): i_start+i_s[j]-int(1/dt_fine)] ** 2)/(5/dt_fine)) )
                        s_rms_e.append( np.sqrt(np.sum(tr.data[i_start+i_s[j]-int(0/dt_fine): i_start+i_s[j]-int(5/dt_fine)] ** 2)/(5/dt_fine)) )
                        finde = 1
                if not findz:
                    s_ampls_z.append(0)
                    s_nois_z.append(0)
                    s_rms_z.append(0)
                if not findn:
                    s_ampls_n.append(0)
                    s_nois_n.append(0)
                    s_rms_n.append(0)
                if not finde:
                    s_ampls_e.append(0)
                    s_nois_e.append(0)
                    s_rms_e.append(0)

                s_warning.append(Empty_warning)

                if s_ampls_n[-1]==0 and s_ampls_e[-1] == 0:
                    s_amp = s_ampls_z[-1]
                    s_nos = s_nois_z[-1]
                else:
                    s_amp = max(s_ampls_n[-1], s_ampls_e[-1])
                    s_nos = s_nois_n[-1] + s_nois_e[-1]

                if p_ampls_n[k] == 0 and p_ampls_e[k] == 0:
                    p_amp = p_ampls_z[k]
                    p_rms = p_rms_z[k]
                else:
                    p_amp = max(p_ampls_n[k], p_ampls_e[k])
                    p_rms = p_rms_n[k] + p_rms_e[k]
                
                sp_ratio.append(s_amp/p_amp)

                if not ( ( (s_amp/p_amp)**2 > 0.3 * s_snr[-1] or s_snr[-1] > 3.2 ) and s_nos > 0.2 * p_rms ) :
                    #if not (s_snr[-1] > 3.5 and s_amp/p_amp > 1.0):
                    s_ampls_z.pop()
                    s_ampls_n.pop()
                    s_ampls_e.pop()
                    s_nois_z.pop()
                    s_nois_n.pop()
                    s_nois_e.pop()
                    s_rms_z.pop()
                    s_rms_n.pop()
                    s_rms_e.pop()
                    s_picks.pop()
                    s_snr.pop()
                    s_warning.pop()
                    sp_ratio.pop()

                #sp_ratio.append(s_amp/p_amp)
            
            s_snrs += s_snr
        

    #print('finish s')

    r_picks = np.array(rough_picks)
    s_picks = np.array(s_picks)
    p_picks = np.array(p_picks)


    return (r_picks, p_picks, s_picks, p_snrs, s_snrs, 
     p_ampls_z, p_ampls_n, p_ampls_e, 
     p_nois_z, p_nois_n, p_nois_e, 
     p_rms_z, p_rms_n, p_rms_e, 
     s_ampls_z, s_ampls_n, s_ampls_e, 
     s_nois_z, s_nois_n, s_nois_e, 
     s_rms_z, s_rms_n, s_rms_e, 
     p_warning, s_warning, sp_ratio)


def get_rough_tr_square(tr_data, dt, new_dt):

    num_dt = int(round(new_dt/dt))

    tr_square_rough = []
    for i in range(len(tr_data)):
        if i % num_dt == 0:
            tr_square_rough.append(0)
        else:
            tr_square_rough[-1] += tr_data[i]**2
    
    return np.array(tr_square_rough)


def merge_snr(snr_stream):

    snr_stream_avg = snr_stream[0].copy()

    if len(snr_stream) == 2:
        snr_stream_avg.data = np.max(np.array([snr_stream[0].data, snr_stream[1].data]), axis=0)
    elif len(snr_stream) >= 3:
        #snr_stream_avg = 1.5 * np.max(np.array([snr_stream[0].data, snr_stream[1].data]), axis=0)
        for i in range(len(snr_stream_avg.data)):
            num_nzeros = np.sum(np.bool_([snr_stream[0].data[i], snr_stream[1].data[i], snr_stream[2].data[i]]))
            if num_nzeros == 1:
                snr_stream_avg.data[i] = max(snr_stream[0].data[i], snr_stream[1].data[i], snr_stream[2].data[i])
            elif num_nzeros >= 2:
                snr_stream_avg.data[i] = (snr_stream[0].data[i] + snr_stream[1].data[i] + snr_stream[2].data[i] - min(snr_stream[0].data[i], snr_stream[1].data[i], snr_stream[2].data[i])) / 1.9
    
    return snr_stream_avg


def collapse_phases(picks_df, threshold):

    collapsed = picks_df.loc[0:0]

    for i in range(1, len(picks_df)):

        if ( np.abs(UTCDateTime(picks_df['time'][i]) - UTCDateTime(collapsed['time'][len(collapsed['time'])-1])) < threshold and picks_df['phase'][i] == collapsed['phase'][len(collapsed['phase'])-1] ):
                pass
        else:
            collapsed = pd.concat([collapsed, picks_df.loc[i:i]], axis=0, ignore_index=True)

    collapsed = collapsed.sort_values(by=['time'])
    collapsed = collapsed.reset_index(drop=True)
    return collapsed


def process_one_stream(st, rough_dt, P_freq_band = (0.5, 5), S_freq_band=(0.3, 3), amp_freq_band=(0.5, 2), wfBaseDir = 'Picks'):


    st_high = st.copy()
    st_low = st.copy()
    st_amp = st.copy()
    st_high.filter('bandpass', freqmin=P_freq_band[0], freqmax=P_freq_band[1], corners=4, zerophase=True)
    st_low.filter('bandpass', freqmin=S_freq_band[0], freqmax=S_freq_band[1], corners=4, zerophase=True)
    st_amp.filter('bandpass', freqmin=amp_freq_band[0], freqmax=amp_freq_band[1], corners=4, zerophase=True)

    net = st[0].meta.network
    sta = st[0].meta.station

    snr_rough = st.copy()
    for i in range(len(snr_rough)):
        snr_rough[i].meta.delta = rough_dt

    dt = st[0].meta.delta
    crr_date = st[0].meta.starttime + 1

    Dir_path = os.path.join(wfBaseDir, "%s/%s/%s/" % (crr_date.year, net, sta))
    if not os.path.exists(Dir_path):
        Path(Dir_path).mkdir(parents=True, exist_ok=True)

    csv_filename = Dir_path + '.'.join([net, sta, str(crr_date.strftime(format='%Y%m%d')), 'csv'])

    # if os.path.exists(csv_filename):
    #     print(f'file {csv_filename} already exists')
    #     return

    earliest = UTCDateTime('2050-01-01')
    latest = UTCDateTime('1970-01-01')

    for i in range(len(st_high)):
        tr = st_high[i]
        trace_start_time = tr.meta.starttime
        earliest = min(trace_start_time, earliest)
        trace_end_time = tr.meta.endtime
        latest = max(trace_end_time, latest)
        tr_square_rough = get_rough_tr_square(tr.data, dt, rough_dt)
        snr_rough_data = SNR_Rough(tr_square_rough, rough_dt, 10, 4)
        snr_rough[i].data = snr_rough_data

    snr_rough.trim(earliest, latest, pad=True, fill_value=0)
    st_high.trim(earliest, latest, pad=True, fill_value=0)
    st_low.trim(earliest, latest, pad=True, fill_value=0)
    st_amp.trim(earliest, latest, pad=True, fill_value=0)

    snr_rough_avg = merge_snr(snr_rough)

    rough_picks = pick_rough_picks(snr_rough_avg.data, rough_dt, 3.0)

    (r_picks, p_picks, s_picks, p_snrs, s_snrs, 
     p_ampls_z, p_ampls_n, p_ampls_e, 
     p_nois_z, p_nois_n, p_nois_e, 
     p_rms_z, p_rms_n, p_rms_e, 
     s_ampls_z, s_ampls_n, s_ampls_e, 
     s_nois_z, s_nois_n, s_nois_e, 
     s_rms_z, s_rms_n, s_rms_e, 
     p_warning, s_warning, sp_ratio)= pick_fine_picks(rough_picks, st_high, st_low, st_amp, 2.8, 2.9)

    pick_data = {
        'station':[],
        'phase':[],
        'time':[],
        'snr':[],
        'amp_z':[],
        'amp_n':[],
        'amp_e':[],
        'nos_z':[],
        'nos_n':[],
        'nos_e':[],
        'rms_z':[],
        'rms_n':[],
        'rms_e':[],
        'warning':[],
        'sp_ratio':[]
    }

    for i in range(len(p_picks)):
        pick_data['station'].append(net+'.'+sta)
        pick_data['phase'].append('P')
        pick_data['time'].append((earliest + p_picks[i]).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
        pick_data['snr'].append(p_snrs[i])
        pick_data['amp_z'].append(p_ampls_z[i])
        pick_data['amp_n'].append(p_ampls_n[i])
        pick_data['amp_e'].append(p_ampls_e[i])
        pick_data['nos_z'].append(p_nois_z[i])
        pick_data['nos_n'].append(p_nois_n[i])
        pick_data['nos_e'].append(p_nois_e[i])
        pick_data['rms_z'].append(p_rms_z[i])
        pick_data['rms_n'].append(p_rms_n[i])
        pick_data['rms_e'].append(p_rms_e[i])
        pick_data['warning'].append(p_warning[i])
        pick_data['sp_ratio'].append(1)
    for i in range(len(s_picks)):
        pick_data['station'].append(net+'.'+sta)
        pick_data['phase'].append('S')
        pick_data['time'].append((earliest + s_picks[i]).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
        pick_data['snr'].append(s_snrs[i])
        pick_data['amp_z'].append(s_ampls_z[i])
        pick_data['amp_n'].append(s_ampls_n[i])
        pick_data['amp_e'].append(s_ampls_e[i])
        pick_data['nos_z'].append(s_nois_z[i])
        pick_data['nos_n'].append(s_nois_n[i])
        pick_data['nos_e'].append(s_nois_e[i])
        pick_data['rms_z'].append(s_rms_z[i])
        pick_data['rms_n'].append(s_rms_n[i])
        pick_data['rms_e'].append(s_rms_e[i])
        pick_data['warning'].append(s_warning[i])
        pick_data['sp_ratio'].append(sp_ratio[i])
    # for i in range(len(r_picks)):
    #     pick_data['station'].append(net+'.'+sta)
    #     pick_data['phase'].append('R')
    #     pick_data['time'].append(trace_start_time + r_picks[i])
    #     pick_data['snr'].append(0)
    #     pick_data['amp_z'].append(0)
    #     pick_data['amp_n'].append(0)
    #     pick_data['amp_e'].append(0)
    #     pick_data['warning'].append(0)

    df = pd.DataFrame(pick_data)

    df['snr_z'] = df['amp_z']/df['nos_z']
    df['snr_n'] = df['amp_n']/df['nos_n']
    df['snr_e'] = df['amp_e']/df['nos_e']
    df_p = df[df['phase']=='P'].reset_index(drop=True)
    df_s = df[df['phase']=='S'].reset_index(drop=True)
    df_s = collapse_phases(df_s, 3)
    df = pd.concat([df_p, df_s], axis=0, ignore_index=True)

    #df = df.sort_values(by='time')
    
    df.to_csv(csv_filename, index=False)


    print(f'saved file {csv_filename} with STA-LTA picked {len(df)} phases')

    #return snr_rough_avg, df
