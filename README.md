STA-LTA & Machine-Learning Phase Picker
Xiaohan Song (xhsong@stanford.edu)

Welcome using my code. The code has two parts: the STA-LTA-based phase detection and the ML-Picker-based phase detection.

To run the code, you should have all the preprocessed daily waveforms stored in the following directory structure:

<img width="482" height="86" alt="image" src="https://github.com/user-attachments/assets/efb7ea69-1a61-4a01-8f50-8d62747288cf" />

Each trace is stored in the path:

```WF_CORR/YEAR/MONTH/NET/STA/NET.STA.YYYYMMDD.CHANNEL.SAC```

For example, the one-day continuous BHZ channel trace of station LSA in network IC in 2008/10/01 is stored in:

```WF_CORR/2008/10/IC/LSA/IC.LSA.20081001.BHZ.SAC```

For another example, the one-day continuous HHE channel trace of station EVN in network IO in 2015/05/12 is stored in:

```WF_CORR/2015/5/IC/LSA/IO.EVN.20081001.HHE.SAC```

IMPORTANT: Each daily continuous waveform file should be merged (with fill_value = 0), detrended, pre-filtered (e.g., 0.05-40 Hz), and station response removed (or divided by station sensitivity) already. Each trace should have its network code and station code stored in the SAC file header.

If you have any problem building a data structure like this, please contact me (xhsong@stanford.edu), and I’ll offer help based on your available database structure.



Install Environment

To run the code, you will need:

```
openmpi
Python=3.12
Numpy=1.26
Scipy
Pandas
Obspy
Seisbench
mpi4py
```

To solve the environment, try using the following command:

```
conda create -n phasepick python=3.12 mpi4py mpich
conda activate phasepick

pip3 install numpy==1.26
pip3 install pandas
pip3 install obspy
pip3 install seisbench
```


The STA-LTA Picker code

This code uses a rough-to-fine STA-LTA algorithm to pick the P and S phases.

To run the STA-LTA picker, enter the STA_LTA folder:
```
cd STA_LTA
```

Run code with:
```
mpirun -np NUMBER_OF_TASK python3 mpi_sta_lta.py NET STARTTIME ENDTIME
```
For example (and for debugging), if you want to run this code on network IC, and you want to detect phases between 2008-10-01T00:00:00 to 2008-11-01T00:00:00, and you can run the code on 8 cores (8 parallel tasks):
```
mpirun -np 8 python3 mpi_sta_lta.py IC 2008-10-01 2008-11-01
```
The picks will be stored in folder STA_LTA /Picks



The Machine-Learning Picker code

This code uses PhaseNet(Original, Diting, Neic, Instance), SkyNet, and EQTransformer to pick the P and S phases.

To run the ML Picker, enter the ML_Picker folder:
```
cd ML_Picker
``` 

Run code with:
```
mpirun -np NUMBER_OF_TASK python3 mpi_ml_picker.py NET STARTTIME ENDTIME
```

For example (and for debugging), if you want to run this code on station network IC, you want to detect phases between 2008-10-01T00:00:00 and 2008-11-01T00:00:00, and you want to run the code on 8 cores (8 parallel tasks), you can run:
```
mpirun -np 8 python3 mpi_ml_picker.py IC 2008-10-01 2008-11-01
```
The picks will be stored in folder ML_Picker/Picks
