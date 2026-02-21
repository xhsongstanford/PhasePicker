import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy import read, read_inventory
import os, sys
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

warnings.filterwarnings("ignore")


class TimeoutError (RuntimeError):
    pass

def handler(signum, frame):
    raise TimeoutError()

signal.signal(signal.SIGALRM, handler)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_amplitudes(stream_amp, picks_df):
    ampls_z = []
    ampls_n = []
    ampls_e = []
    nois_z = []
    nois_n = []
    nois_e = []
    rms_z = []
    rms_n = []
    rms_e = []
    dt = stream_amp[0].meta.delta
    for i in range(len(picks_df)):
        pick_time = UTCDateTime(picks_df['time'][i])
        findz = 0
        findn = 0
        finde = 0
        for tr in stream_amp:
            if picks_df['phase'][i] == 'P':
                i_start = int(round((pick_time - tr.meta.starttime)/dt))-int(1/dt)
                i_end   = int(round((pick_time - tr.meta.starttime)/dt))+int(2/dt)
                nois_start = int(round((pick_time - tr.meta.starttime)/dt))-int(5/dt)
                nois_end   = int(round((pick_time - tr.meta.starttime)/dt))-int(1/dt)
                rms_start = int(round((pick_time - tr.meta.starttime)/dt))+int(0/dt)
                rms_end   = int(round((pick_time - tr.meta.starttime)/dt))+int(4/dt)
            else:
                i_start = int(round((pick_time - tr.meta.starttime)/dt))-int(1/dt)
                i_end   = int(round((pick_time - tr.meta.starttime)/dt))+int(5/dt)
                nois_start = int(round((pick_time - tr.meta.starttime)/dt))-int(6/dt)
                nois_end   = int(round((pick_time - tr.meta.starttime)/dt))-int(1/dt)
                rms_start = int(round((pick_time - tr.meta.starttime)/dt))+int(0/dt)
                rms_end   = int(round((pick_time - tr.meta.starttime)/dt))+int(5/dt)
            if tr.meta.channel[-1] == 'Z':
                try:
                    ampls_z.append( np.max(np.abs(tr.data[i_start: i_end])) )
                    nois_z.append( np.sqrt(np.sum(tr.data[nois_start: nois_end]**2)/(nois_end - nois_start)) )
                    rms_z.append( np.sqrt(np.sum(tr.data[rms_start: rms_end]**2)/(rms_end - rms_start)) )
                except:
                    ampls_z.append(0)
                    nois_z.append(0)
                    rms_z.append(0)
                findz = 1
            elif tr.meta.channel[-1] in ['N', '1'] :
                try:
                    ampls_n.append( np.max(np.abs(tr.data[i_start: i_end])) )
                    nois_n.append( np.sqrt(np.sum(tr.data[nois_start: nois_end]**2)/(nois_end - nois_start)) )
                    rms_n.append( np.sqrt(np.sum(tr.data[rms_start: rms_end]**2)/(rms_end - rms_start)) )
                except:
                    ampls_n.append(0)
                    nois_n.append(0)
                    rms_n.append(0)
                findn = 1
            elif tr.meta.channel[-1] in ['E', '2']:
                try:
                    ampls_e.append( np.max(np.abs(tr.data[i_start: i_end])) )
                    nois_e.append( np.sqrt(np.sum(tr.data[nois_start: nois_end]**2)/(nois_end - nois_start)) )
                    rms_e.append( np.sqrt(np.sum(tr.data[rms_start: rms_end]**2)/(rms_end - rms_start)) )
                except:
                    ampls_e.append(0)
                    nois_e.append(0)
                    rms_e.append(0)
                finde = 1
        if not findz:
            ampls_z.append(0)
            nois_z.append(0)
            rms_z.append(0)
        if not findn:
            ampls_n.append(0)
            nois_n.append(0)
            rms_n.append(0)
        if not finde:
            ampls_e.append(0)
            nois_e.append(0)
            rms_e.append(0)

    # print('***************', crr_date.date, stream_amp[0].meta.station, len(ampls_z), len(picks_df), len(stream_amp))
    # for tr in stream_amp:
    #     print('xxxxxx', stream_amp[0].meta.station, tr.meta.channel)
    
    return ampls_z, ampls_n, ampls_e, nois_z, nois_n, nois_e, rms_z, rms_n, rms_e


def get_snr(tr, sgn_start, sgn_end, nos_start, nos_end):
    sgn_A = np.sqrt(np.sum(tr.data[sgn_start: sgn_end] ** 2)/(sgn_end - sgn_start))
    nos_A = np.sqrt(np.sum(tr.data[nos_start: nos_end] ** 2)/(nos_end - nos_start))
    nos_Check = np.sqrt(np.sum(tr.data[nos_start: nos_start+10] ** 2)/(10))
    epsilon = 5e-10
    if tr.meta.network == 'AM':
        epsilon = 2e-8
    if nos_A < epsilon or nos_Check < epsilon:
        out_snr = 0
    else:
        out_snr = sgn_A/nos_A
    return out_snr


def get_snrs(stream0, picks_df):
    stream1 = stream0.copy()
    stream2 = stream0.copy()
    stream3 = stream0.copy()
    stream1.filter('bandpass', freqmin=0.3, freqmax=3, corners=4, zerophase=True)
    stream2.filter('bandpass', freqmin=0.5, freqmax=5, corners=4, zerophase=True)
    stream3.filter('bandpass', freqmin=1, freqmax=10, corners=4, zerophase=True)
    snrs_z = []
    snrs_n = []
    snrs_e = []
    dt = stream0[0].meta.delta
    for i in range(len(picks_df)):
        pick_time = UTCDateTime(picks_df['time'][i])
        findz = 0
        findn = 0
        finde = 0
        for j in range(len(stream0)):
            tr = stream0[j]
            if picks_df['phase'][i] == 'P':
                sgn_start = int(round((pick_time - tr.meta.starttime)/dt))
                sgn_end = int(round((pick_time - tr.meta.starttime)/dt))+int(1.5/dt)
                nos_start = int(round((pick_time - tr.meta.starttime)/dt))-int(10/dt)
                nos_end = int(round((pick_time - tr.meta.starttime)/dt))-int(0.7/dt)
            else:
                sgn_start = int(round((pick_time - tr.meta.starttime)/dt))
                sgn_end = int(round((pick_time - tr.meta.starttime)/dt))+int(3/dt)
                nos_start = int(round((pick_time - tr.meta.starttime)/dt))-int(5/dt)
                nos_end = int(round((pick_time - tr.meta.starttime)/dt))-int(0.7/dt)
            if tr.meta.channel[-1] == 'Z':
                try:
                    snr_A1 = get_snr(stream1[j], sgn_start, sgn_end, nos_start, nos_end)
                    snr_A2 = get_snr(stream2[j], sgn_start, sgn_end, nos_start, nos_end)
                    snr_A3 = get_snr(stream3[j], sgn_start, sgn_end, nos_start, nos_end)
                    snrs_z.append(max(snr_A1, snr_A2, snr_A3))
                except:
                    snrs_z.append(0)
                findz = 1
            elif tr.meta.channel[-1] in ['N', '1'] :
                try:
                    snr_A1 = get_snr(stream1[j], sgn_start, sgn_end, nos_start, nos_end)
                    snr_A2 = get_snr(stream2[j], sgn_start, sgn_end, nos_start, nos_end)
                    snr_A3 = get_snr(stream3[j], sgn_start, sgn_end, nos_start, nos_end)
                    snrs_n.append(max(snr_A1, snr_A2, snr_A3))
                except:
                    snrs_n.append(0)
                findn = 1
            elif tr.meta.channel[-1] in ['E', '2']:
                try:
                    snr_A1 = get_snr(stream1[j], sgn_start, sgn_end, nos_start, nos_end)
                    snr_A2 = get_snr(stream2[j], sgn_start, sgn_end, nos_start, nos_end)
                    snr_A3 = get_snr(stream3[j], sgn_start, sgn_end, nos_start, nos_end)
                    snrs_e.append(max(snr_A1, snr_A2, snr_A3))
                except:
                    snrs_e.append(0)
                finde = 1
        if not findz:
            snrs_z.append(0)
        if not findn:
            snrs_n.append(0)
        if not finde:
            snrs_e.append(0)
    
    return snrs_z, snrs_n, snrs_e


def run_and_save_picks(stream, stream_amp, model, SaveDir = 'Picks', label = 'diting'):


    total_phases = ClassifyOutput('picks', picks=PickList([]))

    crr_date = stream[0].meta.starttime + 1
    net = stream[0].meta.network
    sta = stream[0].meta.station

    Dir_path = os.path.join(SaveDir, "%s/%s/%s/%s/" % (crr_date.year, label, net, sta))
    if not os.path.exists(Dir_path):
        Path(Dir_path).mkdir(parents=True, exist_ok=True)

    csv_filename = Dir_path + '.'.join([net, sta, str(crr_date.strftime(format='%Y%m%d')), 'csv'])

    # if os.path.exists(csv_filename):
    #     print(f'Rank {rank}: file {csv_filename} already exists')
    #     return

    pn_preds = model.annotate(stream)
    if label == 'diting':
        phases1 = model.classify_aggregate(pn_preds, {'P_threshold':0.07, 'S_threshold':0.07})
        phases2 = model.classify_aggregate(pn_preds, {'P_threshold':0.1,  'S_threshold':0.1})
        phases3 = model.classify_aggregate(pn_preds, {'P_threshold':0.2,  'S_threshold':0.2})
        phases4 = model.classify_aggregate(pn_preds, {'P_threshold':0.3,  'S_threshold':0.3})
    else:
        phases1 = model.classify_aggregate(pn_preds, {'P_threshold':0.1, 'S_threshold':0.1})
        phases2 = model.classify_aggregate(pn_preds, {'P_threshold':0.2,  'S_threshold':0.2})
        phases3 = model.classify_aggregate(pn_preds, {'P_threshold':0.3,  'S_threshold':0.3})
        phases4 = model.classify_aggregate(pn_preds, {'P_threshold':0.4,  'S_threshold':0.4})

    total_phases.picks = PickList(phases1.picks + phases2.picks + phases3.picks + phases4.picks)

    if len(total_phases.picks) == 0:
        return 
    
    picks_df = total_phases.picks.to_dataframe()
    picks_df = picks_df.drop(columns = ['index'])
    picks_df = picks_df.sort_values(by='time')
    picks_df = picks_df.reset_index(drop=True)
    picks_df_p = picks_df[picks_df['phase'] == 'P'].reset_index(drop=True)
    picks_df_s = picks_df[picks_df['phase'] == 'S'].reset_index(drop=True)
    picks_collapsed_p = collapse_phases(picks_df_p, 0.1)
    picks_collapsed_s = collapse_phases(picks_df_s, 0.1)
    picks_df = pd.concat([picks_collapsed_p, picks_collapsed_s], axis=0, ignore_index=True)

    for i in range(len(picks_df)):
        picks_df['station'][i] = '.'.join(picks_df['station'][i].split('.')[:2])

    ampls_z, ampls_n, ampls_e, nois_z, nois_n, nois_e, rms_z, rms_n, rms_e = get_amplitudes(stream_amp, picks_df)
    stream.resample(20)
    snrs_z, snrs_n, snrs_e = get_snrs(stream, picks_df)
    picks_df['amp_z'] = ampls_z
    picks_df['amp_n'] = ampls_n
    picks_df['amp_e'] = ampls_e
    picks_df['nos_z'] = nois_z
    picks_df['nos_n'] = nois_n
    picks_df['nos_e'] = nois_e
    picks_df['rms_z'] = rms_z
    picks_df['rms_n'] = rms_n
    picks_df['rms_e'] = rms_e
    picks_df['snr_z'] = snrs_z
    picks_df['snr_n'] = snrs_n
    picks_df['snr_e'] = snrs_e
    picks_df['source'] = [label] * len(picks_df)
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



def process_single_day(curr_date):

    print(' ============== processing date ' + str(curr_date.date) + ' =======================')

    filepath = os.path.join('../WF_CORR/', str(curr_date.year), str(curr_date.month), net)

    sta_paths = glob(filepath+'/*')

    for sta_path in sta_paths:

        sta_code = sta_path.split('/')[-1]
        # if sta_code != 'R7912':
        #     continue
        #net = inv[0].code

        models = [pn_model_diting, pn_model_original, pn_model_instance, pn_model_neic, skynet_model_o, eqt_model_stead]
        labels = ['diting', 'original', 'instance', 'neic', 'skynet', 'eqt']

        process_idx = []

        for i in range(5):

            Dir_path = os.path.join('Picks', "%s/%s/%s/%s/" % (crr_date.year, labels[i], net, sta_code))
            csv_filename = Dir_path + '.'.join([net, sta_code, str(curr_date.strftime(format='%Y%m%d')), 'csv'])

            # if os.path.exists(csv_filename):
            #     print(f'Rank {rank}: file {csv_filename} already exists')
            # else:
            process_idx.append(i)

        if len(process_idx) == 0:
            continue

        #filepath = os.path.join('../WF_CORR/', str(curr_date.year), str(curr_date.month), net)
        #station_path = filepath + '/' + sta_code
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

        stream_amp = stream.copy()
        stream.resample(100)
        stream.filter('bandpass', freqmin=0.1, freqmax=20, corners=4, zerophase=True)
        stream_amp.filter('bandpass', freqmin=0.5, freqmax=2, corners=4, zerophase=True)

        #print('hhhhhhhhh', len(stream), len(stream_amp))

        for i in process_idx:
            
            try:
                signal.alarm(300)
                run_and_save_picks(stream, stream_amp, model=models[i], SaveDir='Picks', label=labels[i])

            except TimeoutError as ex:

                print(f'***!!!!*** ML picker {labels[i]} timeout for station {sta_code} on {date} with error {ex}')
                continue # Skip the rest of the current iteration and move to the next

            finally:
                signal.alarm(0)

    return



if __name__ == '__main__':

    net = sys.argv[1]
    startdate = UTCDateTime(sys.argv[2])
    enddate = UTCDateTime(sys.argv[3])

    #inv = read_inventory('../StationResp/' + net + '.*.xml')

    # Various pre-trained weights for PhaseNet
    pn_model_diting = sbm.PhaseNet.from_pretrained("diting")
    pn_model_original = sbm.PhaseNet.from_pretrained("original")
    pn_model_instance = sbm.PhaseNet.from_pretrained("instance")
    pn_model_neic = sbm.PhaseNet.from_pretrained("neic")

    # Various pre-trained weights for EQT
    eqt_model_stead = sbm.EQTransformer.from_pretrained("stead")

    # Various pre-trained weights for SkyNet
    skynet_model_o = sbm.Skynet.from_pretrained("original")

    # pn_model_diting.to_preferred_device()
    # pn_model_original.to_preferred_device()
    # pn_model_instance.to_preferred_device()
    # eqt_model_stead.to_preferred_device()
    # skynet_model_o.to_preferred_device()

    pn_model_diting.to('cpu')
    pn_model_original.to('cpu')
    pn_model_instance.to('cpu')
    pn_model_neic.to('cpu')
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
    
    # Process assigned dates
    for date in my_dates:
        process_single_day(date)

    # Wait for all processes to finish
    comm.Barrier()
    
    if rank == 0:
        print('All processes completed!')
