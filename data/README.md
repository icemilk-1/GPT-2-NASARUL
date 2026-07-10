# C-MAPSS Dataset

The NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset is publicly available from the NASA Prognostics Center of Excellence.

## Download

1. Go to the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/prognostics-center-of-excellence-data-set-repository/)
2. Download "Turbofan Engine Degradation Simulation Data Set"
3. Extract the following `.txt` files into `data/CMaps/`:

```
data/CMaps/
├── train_FD001.txt
├── train_FD002.txt
├── train_FD003.txt
├── train_FD004.txt
├── test_FD001.txt
├── test_FD002.txt
├── test_FD003.txt
├── test_FD004.txt
├── RUL_FD001.txt
├── RUL_FD002.txt
├── RUL_FD003.txt
└── RUL_FD004.txt
```

## Dataset Summary

| Dataset | Train engines | Test engines | Operating conditions | Fault modes |
|---------|-------------|-------------|---------------------|-------------|
| FD001 | 100 | 100 | 1 | 1 |
| FD002 | 260 | 259 | 6 | 1 |
| FD003 | 100 | 100 | 1 | 2 |
| FD004 | 249 | 248 | 6 | 2 |

Each engine has 21 sensor measurements and 3 operational settings, recorded per cycle until failure.
