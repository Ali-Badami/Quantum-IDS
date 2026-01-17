# SWaT Dataset

This directory should contain the raw SWaT (Secure Water Treatment) dataset files.

## Obtaining the Dataset

The SWaT dataset is provided by iTrust, Singapore University of Technology and Design. You need to request access through their official portal:

https://itrust.sutd.edu.sg/itrust-labs_datasets/

## Required Files

After obtaining access, place these files in this directory:

```
normal.csv    - Normal operation data (approx 495,000 samples)
attack.csv    - Attack scenario data (approx 449,000 samples)
```

## Dataset Description

The SWaT testbed is a scaled-down water treatment plant with six stages:

- P1: Raw water intake
- P2: Pre-treatment
- P3: Ultrafiltration
- P4: Dechlorination
- P5: Reverse osmosis
- P6: Backwash

The dataset includes 51 sensor and actuator readings sampled at 1 Hz.

## Preprocessing Notes

The data ingestion script will:

1. Remove the first 6 hours (21,600 rows) as warm-up period
2. Handle missing values with forward-fill
3. Normalize features using MinMaxScaler fitted on training data only

## Citation

If you use this dataset, please cite the original paper:

```
@inproceedings{goh2017dataset,
  title={A dataset to support research in the design of secure water treatment systems},
  author={Goh, Jonathan and others},
  booktitle={Critical Information Infrastructures Security},
  year={2017}
}
```
