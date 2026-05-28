import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy import read, read_inventory
import os, sys, gc
from pathlib import Path
import seisbench.models as sbm
from obspy.geodetics import gps2dist_azimuth
from seisbench.util.annotations import ClassifyOutput
from seisbench.util.annotations import PickList
from mpi4py import MPI
import warnings
import time
import signal
from glob import glob

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)


warnings.filterwarnings("ignore")


class TimeoutError (RuntimeError):
    pass


def handler(signum, frame):
    raise TimeoutError()

signal.signal(signal.SIGALRM, handler)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_amplitudes(stream_amp, stream_disp, picks_df ,label):
    ampls_z = []
    ampls_n = []
    ampls_e = []
    ampdis_z = []
    ampdis_n = []
    ampdis_e = []
    nois_z = []
    nois_n = []
    nois_e = []
    amprms_z = []
    amprms_n = []
    amprms_e = []

    dt = stream_amp[0].meta.delta
    for i in range(len(picks_df)):
        pick_time = UTCDateTime(picks_df['time'][i])
        findz = 0
        findn = 0
        finde = 0
        for i_tr in range(len(stream_amp)):
            tr = stream_amp[i_tr]
            tr_disp = stream_disp[i_tr]
            init_time = int(round((pick_time - tr.meta.starttime)/dt))
            if picks_df['phase'][i] == 'P':
                i_start = init_time-int(1/dt)
                i_end   = init_time+int(2/dt)
                i_start_dis = init_time-int(1/dt)
                i_end_dis   = init_time+int(5/dt)
                nois_start = init_time-int(5/dt)
                nois_end   = init_time-int(1/dt)
                amprms_start = init_time+int(0/dt)
                amprms_end   = init_time+int(4/dt)
            else:
                i_start = init_time-int(1/dt)
                i_end   = init_time+int(5/dt)
                i_start_dis = init_time-int(1/dt)
                i_end_dis   = init_time+int(15/dt)
                # if label == 'skynet':
                #     nois_start = init_time-int(6/dt)
                #     nois_end   = init_time-int(1/dt)
                # else:
                nois_start = init_time-int(6/dt)
                nois_end   = init_time-int(1/dt)
                amprms_start = init_time+int(0/dt)
                amprms_end   = init_time+int(5/dt)
            if tr.meta.channel[-1] == 'Z':
                try:
                    ampls_z.append( np.max(np.abs(tr.data[i_start: i_end])) )
                except:
                    ampls_z.append(0)
                try:
                    ampdis_z.append( np.max(np.abs(tr_disp.data[i_start_dis: i_end_dis])) )
                except:
                    ampdis_z.append(0)
                try:
                    nois_z.append( np.sqrt(np.sum(tr.data[nois_start: nois_end]**2)/(nois_end - nois_start)) )
                except:
                    nois_z.append(0)
                try:
                    amprms_z.append( np.sqrt(np.sum(tr.data[amprms_start: amprms_end]**2)/(amprms_end - amprms_start)) )
                except:
                    amprms_z.append(0)
                findz = 1
            elif tr.meta.channel[-1] in ['N', '1']:
                if findn == 1:
                    print(f" ********** Warining ****** Both N and 1 channels exists for station {tr.meta.station} on {tr.meta.starttime.date} ******* ")
                    continue
                try:
                    ampls_n.append( np.max(np.abs(tr.data[i_start: i_end])) )
                except:
                    ampls_n.append(0)
                try:
                    ampdis_n.append( np.max(np.abs(tr_disp.data[i_start_dis: i_end_dis])) )
                except:
                    ampdis_n.append(0)
                try:
                    nois_n.append( np.sqrt(np.sum(tr.data[nois_start: nois_end]**2)/(nois_end - nois_start)) )
                except:
                    nois_n.append(0)
                try:
                    amprms_n.append( np.sqrt(np.sum(tr.data[amprms_start: amprms_end]**2)/(amprms_end - amprms_start)) )
                except:
                    amprms_n.append(0)
                findn = 1
            elif tr.meta.channel[-1] in ['E', '2']:
                if finde == 1:
                    print(f" ********** Warining ****** Both E and 2 channels exist for station {tr.meta.station} on {tr.meta.starttime.date} ******* ")
                    continue
                try:
                    ampls_e.append( np.max(np.abs(tr.data[i_start: i_end])) )
                except:
                    ampls_e.append(0)
                try:
                    ampdis_e.append( np.max(np.abs(tr_disp.data[i_start_dis: i_end_dis])) )
                except:
                    ampdis_e.append(0)
                try:
                    nois_e.append( np.sqrt(np.sum(tr.data[nois_start: nois_end]**2)/(nois_end - nois_start)) )
                except:
                    nois_e.append(0)
                try:
                    amprms_e.append( np.sqrt(np.sum(tr.data[amprms_start: amprms_end]**2)/(amprms_end - amprms_start)) )
                except:
                    amprms_e.append(0)
                finde = 1
        if not findz:
            ampls_z.append(0)
            ampdis_z.append(0)
            nois_z.append(0)
            amprms_z.append(0)
        if not findn:
            ampls_n.append(0)
            ampdis_n.append(0)
            nois_n.append(0)
            amprms_n.append(0)
        if not finde:
            ampls_e.append(0)
            ampdis_e.append(0)
            nois_e.append(0)
            amprms_e.append(0)

    ampls_z = [float(f"{i:.{3}g}") for i in ampls_z]
    ampls_n = [float(f"{i:.{3}g}") for i in ampls_n]
    ampls_e = [float(f"{i:.{3}g}") for i in ampls_e]
    ampdis_z = [float(f"{i:.{3}g}") for i in ampdis_z]
    ampdis_n = [float(f"{i:.{3}g}") for i in ampdis_n]
    ampdis_e = [float(f"{i:.{3}g}") for i in ampdis_e]
    amprms_z = [float(f"{i:.{3}g}") for i in amprms_z]
    amprms_n = [float(f"{i:.{3}g}") for i in amprms_n]
    amprms_e = [float(f"{i:.{3}g}") for i in amprms_e]
    nois_z = [float(f"{i:.{3}g}") for i in nois_z]
    nois_n = [float(f"{i:.{3}g}") for i in nois_n]
    nois_e = [float(f"{i:.{3}g}") for i in nois_e]

    return ampls_z, ampls_n, ampls_e, ampdis_z, ampdis_n, ampdis_e, nois_z, nois_n, nois_e, amprms_z, amprms_n, amprms_e


def get_snr(tr, sgn_start, sgn_end, nos_start, nos_end):
    warning = False
    sgn_A = np.sqrt(np.sum(tr.data[sgn_start: sgn_end] ** 2)/(sgn_end - sgn_start))
    nos_A = np.sqrt(np.sum(tr.data[nos_start: nos_end] ** 2)/(nos_end - nos_start))

    nos_Check = np.max(np.abs( tr.data[nos_start: nos_start+15] ))
    epsilon = 5e-10
    if tr.meta.network == 'AM': # For accelerometer
        epsilon = 0.5e-8
    if nos_A < epsilon or nos_Check < epsilon:
        out_snr = sgn_A/max(epsilon, nos_A)
        warning = True
    else:
        out_snr = sgn_A/nos_A

    # if warning:
    #     print(sgn_A, nos_A, sgn_start, sgn_end)

    return out_snr, warning


def get_snrs(streams_for_snr, picks_df, label):
    stream1 = streams_for_snr[0]

    stream2 = streams_for_snr[1]

    stream3 = streams_for_snr[2]

    # stream1.filter('bandpass', freqmin=0.3, freqmax=3, corners=4, zerophase=True)
    # stream2.filter('bandpass', freqmin=0.5, freqmax=5, corners=4, zerophase=True)
    # stream3.filter('bandpass', freqmin=1, freqmax=10, corners=4, zerophase=True)
    snrs_z = []
    snrs_n = []
    snrs_e = []
    warnings = []
    dt = stream1[0].meta.delta
    for i in range(len(picks_df)):
        pick_time = UTCDateTime(picks_df['time'][i])
        findz = 0
        findn = 0
        finde = 0
        warningz = False
        warningn = False
        warninge = False
        for j in range(len(stream1)):
            tr = stream1[j]
            init_time = int(round((pick_time - tr.meta.starttime)/dt))
            if picks_df['phase'][i] == 'P':
                sgn_start = init_time
                sgn_end = init_time+int(1.5/dt)
                nos_start = init_time-int(10/dt)
                nos_end = init_time-int(2/dt)
            else:
                #if label == 'skynet':
                sgn_start = init_time
                sgn_end = init_time+int(4/dt)
                # else:
                #     sgn_start = init_time
                #     sgn_end = init_time+int(3/dt)
                nos_start = init_time-int(6/dt)
                nos_end   = init_time-int(2/dt)
            if tr.meta.channel[-1] == 'Z':
                try:
                    snr_A1, warning1 = get_snr(stream1[j], sgn_start, sgn_end, nos_start, nos_end)
                    snr_A2, warning2 = get_snr(stream2[j], sgn_start, sgn_end, nos_start, nos_end)
                    snr_A3, warning3 = get_snr(stream3[j], sgn_start, sgn_end, nos_start, nos_end)
                    warningz = (warning1 | warning2 | warning3)
                    snrs_z.append(max(snr_A1, snr_A2, snr_A3))
                except Exception as e:
                    snrs_z.append(0)
                    warningz = True
                findz = 1
            elif tr.meta.channel[-1] in ['N', '1'] :
                if findn == 1:
                    print(f" ********** Warining ****** Both N and 1 channels exist for station {tr.meta.station} on {tr.meta.starttime.date} ******* ")
                    continue
                try:
                    snr_A1, warning1 = get_snr(stream1[j], sgn_start, sgn_end, nos_start, nos_end)
                    snr_A2, warning2 = get_snr(stream2[j], sgn_start, sgn_end, nos_start, nos_end)
                    snr_A3, warning3 = get_snr(stream3[j], sgn_start, sgn_end, nos_start, nos_end)
                    warningn = (warning1 | warning2 | warning3)
                    snrs_n.append(max(snr_A1, snr_A2, snr_A3))
                except Exception as e:
                    snrs_n.append(0)
                    warningn = True
                findn = 1
            elif tr.meta.channel[-1] in ['E', '2']:
                if finde == 1:
                    print(f" ********** Warining ****** Both E and 2 channels exist for station {tr.meta.station} on {tr.meta.starttime.date} ******* ")
                    continue
                try:
                    snr_A1, warning1 = get_snr(stream1[j], sgn_start, sgn_end, nos_start, nos_end)
                    snr_A2, warning2 = get_snr(stream2[j], sgn_start, sgn_end, nos_start, nos_end)
                    snr_A3, warning3 = get_snr(stream3[j], sgn_start, sgn_end, nos_start, nos_end)
                    warninge = (warning1 | warning2 | warning3)
                    snrs_e.append(max(snr_A1, snr_A2, snr_A3))
                except Exception as e:
                    snrs_e.append(0)
                    warninge = True
                finde = 1
        if not findz:
            snrs_z.append(0)
            warningz = True
        if not findn:
            snrs_n.append(0)
            warningn = True
        if not finde:
            snrs_e.append(0)
            warninge = True
        
        warnings.append(warningn & warninge & warningz)


    # del stream1
    # del stream2
    # del stream3

    snrs_z = [float(f"{i:.{3}g}") for i in snrs_z]
    snrs_n = [float(f"{i:.{3}g}") for i in snrs_n]
    snrs_e = [float(f"{i:.{3}g}") for i in snrs_e]

    return snrs_z, snrs_n, snrs_e, warnings


def pick_maximum(picks1, picks2):
    if len(picks1.picks) > len(picks2.picks):
        return picks1
    else:
        return picks2
    
def round_sig_column(val, sig):
    if val == 0 or np.isnan(val):
        return val
    return round(val, sig - int(np.floor(np.log10(abs(val)))) - 1)


def run_and_save_picks(process_date, stream, stream2, stream_amp, stream_disp, streams_for_snr, model, SaveDir = 'Picks', label = 'diting'):


    total_phases = ClassifyOutput('picks', picks=PickList([]))

    # crr_date = stream[0].meta.starttime + 1
    net = stream[0].meta.network
    sta = stream[0].meta.station

    Dir_path = os.path.join(SaveDir, "%s/%s/%s/%s/" % (process_date.year, label, net, sta))
    if not os.path.exists(Dir_path):
        Path(Dir_path).mkdir(parents=True, exist_ok=True)

    csv_filename = Dir_path + '.'.join([net, sta, str(process_date.strftime(format='%Y%m%d')), 'csv'])


    pn_preds = model.annotate(stream)

    if label in ['diting']:

        if net != 'AM':
            pn_preds2 = model.annotate(stream2)
            for i in range(len(pn_preds)):
                pn_preds[i].data = np.max([pn_preds[i].data, pn_preds2[i].data], axis = 0)

            del pn_preds2

        #del stream2

        phases1 = model.classify_aggregate(pn_preds, {'P_threshold':0.1,  'S_threshold':0.1})
        phases2 = model.classify_aggregate(pn_preds, {'P_threshold':0.2,  'S_threshold':0.2})
        phases3 = model.classify_aggregate(pn_preds, {'P_threshold':0.3,  'S_threshold':0.3})
        phases4 = model.classify_aggregate(pn_preds, {'P_threshold':0.4,  'S_threshold':0.4})

        total_phases.picks = PickList(phases1.picks + phases2.picks + phases3.picks + phases4.picks)

    if label == 'skynet':
    
        phases1 = model.classify_aggregate(pn_preds, {'P_threshold':0.2,  'S_threshold':0.2})
        phases2 = model.classify_aggregate(pn_preds, {'P_threshold':0.3,  'S_threshold':0.3})
        phases3 = model.classify_aggregate(pn_preds, {'P_threshold':0.4,  'S_threshold':0.4})

        total_phases.picks = PickList(phases1.picks + phases2.picks + phases3.picks)

    else:

        phases1 = model.classify_aggregate(pn_preds, {'P_threshold':0.1,  'S_threshold':0.1})
        phases2 = model.classify_aggregate(pn_preds, {'P_threshold':0.2,  'S_threshold':0.2})
        phases3 = model.classify_aggregate(pn_preds, {'P_threshold':0.4,  'S_threshold':0.4})

        total_phases.picks = PickList(phases1.picks + phases2.picks + phases3.picks)

    del pn_preds

    gc.collect()

    if len(total_phases.picks) == 0:
        return 
    
    picks_df = total_phases.picks.to_dataframe()
    picks_df = picks_df.drop(columns = ['index'])
    try:
        picks_df = picks_df.drop(columns = ['polarity', 'polarity_probability'])
    except:
        pass
    picks_df['probability'] = picks_df['probability'].apply(lambda v: round_sig_column(v, 3))
    
    picks_df = picks_df.sort_values(by='time')
    picks_df = picks_df.reset_index(drop=True)
    picks_df_p = picks_df[picks_df['phase'] == 'P'].reset_index(drop=True)
    picks_df_s = picks_df[picks_df['phase'] == 'S'].reset_index(drop=True)
    picks_collapsed_p = collapse_phases(picks_df_p, 0.1)
    picks_collapsed_s = collapse_phases(picks_df_s, 0.1)
    picks_df = pd.concat([picks_collapsed_p, picks_collapsed_s], axis=0, ignore_index=True)

    for i in range(len(picks_df)):
        picks_df['station'][i] = '.'.join(picks_df['station'][i].split('.')[:2])

    [ampls_z, ampls_n, ampls_e, 
     ampdis_z, ampdis_n, ampdis_e, 
     nois_z, nois_n, nois_e, 
     amprms_z, amprms_n, amprms_e] = get_amplitudes(stream_amp, stream_disp, picks_df, label)
    
    snrs_z, snrs_n, snrs_e, warnings= get_snrs(streams_for_snr, picks_df, label)
    
    if len(ampls_z) != len(picks_df):
        print(f'Rank {rank}: **** station error {sta} on {process_date.date}*******')

    picks_df['ampdis_z'] = ampdis_z
    picks_df['ampdis_n'] = ampdis_n
    picks_df['ampdis_e'] = ampdis_e

    picks_df['amp_z'] = ampls_z
    picks_df['amp_n'] = ampls_n
    picks_df['amp_e'] = ampls_e

    picks_df['rms_z'] = amprms_z
    picks_df['rms_n'] = amprms_n
    picks_df['rms_e'] = amprms_e

    picks_df['nos_z'] = nois_z
    picks_df['nos_n'] = nois_n
    picks_df['nos_e'] = nois_e


    if len(snrs_z) != len(picks_df):
        print(f'Rank {rank}: **** station error {sta} on {process_date.date}*******')
    
    picks_df['snr_z'] = snrs_z
    picks_df['snr_n'] = snrs_n
    picks_df['snr_e'] = snrs_e
    picks_df['source'] = [label] * len(picks_df)
    picks_df['warning'] = warnings
    picks_df = picks_df.sort_values(by=['phase', 'time'])

    picks_df.to_csv(csv_filename, index=False)
    print(f'Rank {rank}: saved file {csv_filename} with model_{label} picked {len(picks_df)} phases')

    return


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



def process_single_day(curr_date, SaveDir):

    print(f'Rank {rank}: ============== processing date ' + str(curr_date.date) + ' =======================')

    for net in nets:

        filepath = os.path.join('../WF_CORR/', str(curr_date.year), str(curr_date.month), net)

        sta_paths = glob(filepath+'/*')

        for sta_path in sta_paths:

            sta_code = sta_path.split('/')[-1]

            # if sta_code != 'R5A68':
            #     continue

            # ======================= If you want to include more neural networks or less, change here =====================

            # models = [pn_model_diting, pn_model_original, pn_model_instance, skynet_model_o, eqt_model_stead]
            # labels = ['diting', 'original', 'instance', 'skynet', 'eqt']
            # models = [pn_model_diting, pn_model_original, skynet_model_o, eqt_model_stead]
            # labels = ['diting', 'original', 'skynet', 'eqt']
            models = [pn_model_diting, skynet_model_o, eqt_model_stead]
            labels = ['diting', 'skynet', 'eqt']
            # models = [pn_model_diting, skynet_model_o]
            # labels = ['diting', 'skynet',]

            # ======================= END =====================

            process_idx = []

            for i in range(len(labels)):

                Dir_path = os.path.join(SaveDir, "%s/%s/%s/%s/" % (curr_date.year, labels[i], net, sta_code))
                csv_filename = Dir_path + '.'.join([net, sta_code, str(curr_date.strftime(format='%Y%m%d')), 'csv'])

                # if os.path.exists(csv_filename):
                #    print(f'Rank {rank}: file {csv_filename} already exists')
                # else:
                #    process_idx.append(i)

                process_idx.append(i)


            if len(process_idx) == 0:
                continue

            filename = sta_path + '/' + "%s.%s.%s." % (net, sta_code, curr_date.strftime("%Y%m%d")) + '*.SAC'

            try:
                stream = read(filename)
                stream.merge()
            except:
                print(f'Rank {rank}: no data for {sta_code} on {curr_date.date}')
                continue

            for tr in stream:
                if tr.meta.channel[-1] == '1':
                    tr.meta.channel = tr.meta.channel[:-1] + 'N'
                if tr.meta.channel[-1] == '2':
                    tr.meta.channel = tr.meta.channel[:-1] + 'E'

            stream_high = stream.copy()
            stream_low = stream.copy()
            stream_amp = stream.copy()
            stream_disp = stream.copy()
            stream_disp.integrate(method='cumtrapz')
            # for tr in stream_disp:
            #     tr.data *= 1000
            #     # Update the stats header to reflect new units
            #     tr.stats.units = 'mm'

            if net == 'AM': # If the station is Accelermerter:
                stream_high.filter('bandpass', freqmin=1.5, freqmax=10, corners=4, zerophase=True)
                stream.filter('bandpass', freqmin=1.0, freqmax=10, corners=4, zerophase=True)
                stream_low.filter('bandpass', freqmin=0.3, freqmax=5, corners=4, zerophase=True)
            else:
                stream_high.filter('bandpass', freqmin=1, freqmax=10, corners=4, zerophase=True)
                stream.filter('bandpass', freqmin=0.5, freqmax=5, corners=4, zerophase=True)
                stream_low.filter('bandpass', freqmin=0.3, freqmax=5, corners=4, zerophase=True)

            stream_amp.filter('bandpass', freqmin=0.5, freqmax=2, corners=4, zerophase=True)
            stream_disp.filter('bandpass', freqmin=0.5, freqmax=2, corners=4, zerophase=True)

            # if stream_high[0].meta.delta < 0.02:
            #     stream_high.resample(50)
            #     stream_low.resample(50)
            #     stream.resample(50)

            if stream_amp[0].meta.delta < 0.049:
                stream_amp.resample(20)
                stream_disp.resample(20)

            for st in [stream_high, stream_low, stream, stream_amp, stream_disp]:
                for tr in st:
                    tr.data = tr.data.astype(np.float32)

            streams_for_snr = [stream_high, stream, stream_low]
            
            gc.collect()

            for i in process_idx:

                # try:
                #     signal.alarm(300)

                #     run_and_save_picks(stream_high, stream, stream_low, stream_amp, stream_disp, model=models[i], SaveDir=SaveDir, label=labels[i])

                # except TimeoutError as ex:
                #     print(f'***!!!!*** ML picker {labels[i]} timeout for station {sta_code} on {date} with error {ex}')
                #     continue # Skip the rest of the current iteration and move to the next
                # finally:
                #     signal.alarm(0)

                if labels[i] in ['diting']:

                    try:
                        signal.alarm(300)
                        run_and_save_picks(curr_date, stream_high, stream_low, stream_amp, stream_disp, streams_for_snr, model=models[i], SaveDir=SaveDir, label=labels[i])

                    except TimeoutError as ex:

                        print(f'***!!!!*** ML picker {labels[i]} timeout for station {sta_code} on {date} with error {ex}')
                        continue # Skip the rest of the current iteration and move to the next
                    finally:
                        signal.alarm(0)
                        
                elif labels[i] == 'skynet':

                    try:
                        signal.alarm(300)
                        if net == 'AM':
                            run_and_save_picks(curr_date, stream_high, None, stream_amp, stream_disp, streams_for_snr, model=models[i], SaveDir=SaveDir, label=labels[i])
                        else:
                            run_and_save_picks(curr_date, stream, None, stream_amp, stream_disp, streams_for_snr, model=models[i], SaveDir=SaveDir, label=labels[i])

                    except TimeoutError as ex:

                        print(f'***!!!!*** ML picker {labels[i]} timeout for station {sta_code} on {date} with error {ex}')
                        continue # Skip the rest of the current iteration and move to the next
                    finally:
                        signal.alarm(0)

                elif labels[i] == 'eqt':

                    try:
                        signal.alarm(300)
                        run_and_save_picks(curr_date, stream_high, None, stream_amp, stream_disp, streams_for_snr, model=models[i], SaveDir=SaveDir, label=labels[i])
                    except TimeoutError as ex:

                        print(f'***!!!!*** ML picker {labels[i]} timeout for station {sta_code} on {date} with error {ex}')
                        continue # Skip the rest of the current iteration and move to the next
                    finally:
                        signal.alarm(0)

                else:

                    try:
                        signal.alarm(300)
                        run_and_save_picks(curr_date, stream, None, stream_amp, stream_disp, streams_for_snr, model=models[i], SaveDir=SaveDir, label=labels[i])
                    except TimeoutError as ex:

                        print(f'***!!!!*** ML picker {labels[i]} timeout for station {sta_code} on {date} with error {ex}')
                        continue # Skip the rest of the current iteration and move to the next
                    finally:
                        signal.alarm(0)



            del stream
            del stream_high
            del stream_amp
            del stream_low
            del stream_disp
            del streams_for_snr

            gc.collect()

    return



if __name__ == '__main__':

    nets = sys.argv[1].split(',')
    startdate = UTCDateTime(sys.argv[2])
    enddate = UTCDateTime(sys.argv[3])

    SaveDir = 'Picks'

    # Various pre-trained weights for PhaseNet
    pn_model_diting = sbm.PhaseNet.from_pretrained("diting")
    #pn_model_original = sbm.PhaseNet.from_pretrained("original")
    #pn_model_instance = sbm.PhaseNet.from_pretrained("instance")

    # Various pre-trained weights for EQT
    eqt_model_stead = sbm.EQTransformer.from_pretrained("stead")

    # Various pre-trained weights for SkyNet
    skynet_model_o = sbm.Skynet.from_pretrained("original")

    # pn_model_diting.to_preferred_device()
    # pn_model_original.to_preferred_device()
    # #pn_model_instance.to_preferred_device()
    # eqt_model_stead.to_preferred_device()
    # skynet_model_o.to_preferred_device()

    pn_model_diting.to('cpu')
    #pn_model_original.to('cpu')
    #pn_model_instance.to('cpu')
    eqt_model_stead.to('cpu')
    skynet_model_o.to('cpu')

    # Generate list of all dates to process
    all_dates = []
    crr_date = startdate
    while crr_date < enddate:
        all_dates.append(crr_date)
        crr_date += 24 * 3600
    
    # Distribute dates across processes
    # Each rank processes every Nth date where N is the number of processes
    my_dates = all_dates[rank::size]
    
    if rank == 0:
        print(f'Total days to process: {len(all_dates)}')
        print(f'Number of MPI processes: {size}')
        print(f'Days per process: ~{len(all_dates)//size}')
        time_beg = time.time()
    
    # Process assigned dates
    for date in my_dates:
        process_single_day(date, SaveDir)

    # Wait for all processes to finish
    comm.Barrier()
    
    if rank == 0:
        time_end = time.time()
        print(f'OvO ... All processes in {time_end - time_beg} s')
