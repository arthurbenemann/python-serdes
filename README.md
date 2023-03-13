# python-serdes

Python simulation of BER vs SNR for a lossy channel using various equalization schemes

Script test runs and output images can be found at:
[Github CI](https://github.com/arthurbenemann/python-serdes/actions/workflows/python-app.yml)

## Example run
Running the default config:
```
$ber.py 
isi                       100%|████████████████████████| 70.0/70.0 [00:00<00:00, 82218.22kbit/s]
ffe                       100%|████████████████████████| 70.0/70.0 [00:00<00:00, 42263.03kbit/s]
dfe                       100%|████████████████████████| 70.0/70.0 [00:01<00:00, 36.68kbit/s]
mlse                      100%|████████████████████████| 70.0/70.0 [00:03<00:00, 19.72kbit/s]
```

Running long channel with 10 SNR stops and 3e6 bits on a AMD 5500U:
```
$python3 ber.py --long -snr=10 -n=3e6 --multi-thread 
QUEUEING TASKS | isi      100%|████████████████████████| 33000.0/33000.0 [00:00<00:00, 632061.70kbit/s]
PROCESSING TASKS | isi    100%|████████████████████████| 33000.0/33000.0 [00:01<00:00, 23314.28kbit/s]
COLLECTING RESULTS | isi  100%|████████████████████████| 33000.0/33000.0 [00:00<00:00, 114864756.85kbit/s
QUEUEING TASKS | ffe      100%|████████████████████████| 33000.0/33000.0 [00:00<00:00, 518049.82kbit/s]
PROCESSING TASKS | ffe    100%|████████████████████████| 33000.0/33000.0 [00:01<00:00, 21449.44kbit/s]
COLLECTING RESULTS | ffe  100%|████████████████████████| 33000.0/33000.0 [00:00<00:00, 330338978.52kbit/s
QUEUEING TASKS | dfe      100%|████████████████████████| 33000.0/33000.0 [00:00<00:00, 631119.28kbit/s]
PROCESSING TASKS | dfe    100%|████████████████████████| 33000.0/33000.0 [05:05<00:00, 107.87kbit/s]
COLLECTING RESULTS | dfe  100%|████████████████████████| 33000.0/33000.0 [00:00<00:00, 245847303.73kbit/s
QUEUEING TASKS | mlse     100%|████████████████████████| 33000.0/33000.0 [00:00<00:00, 574767.38kbit/s]
PROCESSING TASKS | mlse   100%|████████████████████████| 33000.0/33000.0 [15:21<00:00, 35.81kbit/s] 
COLLECTING RESULTS | mlse 100%|████████████████████████| 33000.0/33000.0 [00:00<00:00, 193312893.85kbit/s 
```
![snr_vs_ber](https://user-images.githubusercontent.com/3289118/224606763-38c311da-201b-46f9-92c5-c5fdb40bf9e5.png)


