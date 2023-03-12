# python-serdes

Python simulation of BER vs SNR for a lossy channel using various equalization schemes

Script test runs and output images can be found at:
[Github CI](https://github.com/arthurbenemann/python-serdes/actions/workflows/python-app.yml)

## example run
Rnning `python ber.py`, for channel `([1., .6, .4, 0.2, 0.1, 0.0, -0.1, 0, 0, 0, 0, 0, 0, .3, -.2])` and `1e6` bits, on a 12600k:
```
$python ber.py
ideal: 100%|█████████████████████████████| 11000.0/11000.0 [00:00<00:00,  18715.77kbit/s] 
isi:   100%|█████████████████████████████| 11000.0/11000.0 [00:00<00:00, 845966.92kbit/s] 
ffe:   100%|█████████████████████████████| 11000.0/11000.0 [00:00<00:00,  59712.50kbit/s] 
dfe:   100%|█████████████████████████████| 11000.0/11000.0 [01:20<00:00,    136.19kbit/s] 
mlse:  100%|█████████████████████████████| 11000.0/11000.0 [23:58<00:00,      7.65kbit/s]
```
![snr_vs_ber](https://user-images.githubusercontent.com/3289118/224576588-876d92e4-a7e1-4892-87ba-3def970051d3.png)

