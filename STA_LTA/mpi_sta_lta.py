import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy import read
import os,sys
from pathlib import Path
from sta_lta_stream_utils import *
from glob import glob
from mpi4py import MPI
import warnings
warnings.filterwarnings("ignore")


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

####################################################################################
#####       STA_LTA DETECTION
####################################################################################

Dir = '../WF_CORR'



def process_single_day(curr_date):

    print('===========  processing date ' + str(curr_date.date) + ' ================================')
    #crr_date = UTCDateTime('2023-11-27')
    for net in nets:
        filepath = os.path.join(Dir, str(curr_date.year), str(curr_date.month), net)
        station_paths = glob(filepath + '/*')
        for station_path in station_paths:
            sta = station_path.split('/')[-1]

            # if sta != 'REDBD' and sta != 'ZHF':
            #     continue

            #print('process station ' + station_path)
            try:
                stream = read(station_path + '/' + "%s.%s.%s." % (net, sta, curr_date.strftime("%Y%m%d")) + '*.SAC')
            except:
                print('no data from station ' + station_path.split('/')[-1] + ' on ' + str(curr_date.strftime("%Y-%m-%d")) )
                continue

            #stream.trim(curr_date, curr_date + 24*3600, pad=True, fill_value=0)
            if stream[0].meta.delta < 0.05:
                stream.resample(20, window=None)
            # else:
            #     stream.resample(10)
                
            process_one_stream(stream, 1, (0.5, 5), (0.3, 3), (0.5, 2), wfBaseDir='Picks')
    
    # curr_date += 24*3600
    return


if __name__ == '__main__':

    nets = sys.argv[1].split(',')
    startdate = UTCDateTime(sys.argv[2])
    enddate = UTCDateTime(sys.argv[3])
    
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
