# HAI Dataset

This directory should contain the raw HAI (Hardware-in-the-Loop Augmented ICS) dataset files.

## Obtaining the Dataset

The HAI dataset is provided by ETRI (Electronics and Telecommunications Research Institute) and is available on GitHub:

https://github.com/icsdataset/hai

Download the HAI 22.04 version for compatibility with this codebase.

## Required Files

After downloading, place all CSV files in this directory:

```
train1.csv
train2.csv
train3.csv
...
test1.csv
test2.csv
test3.csv
...
```

## Dataset Description

The HAI testbed simulates a thermal power generation and pumped-storage hydropower system. It includes:

- Boiler system
- Turbine system
- Water treatment system

The dataset contains sensor readings from various stages of the power generation process.

## Key Differences from SWaT

- HAI has a much more imbalanced class distribution (attacks are rarer)
- The attack patterns are more subtle and harder to detect
- Cross-domain validation from water treatment to power systems

## Preprocessing Notes

The HAI data ingestion script handles the multi-file structure and combines them appropriately for training and testing.

## Citation

If you use this dataset, please cite:

```
@article{shin2020hai,
  title={HAI 1.0: HIL-based augmented ICS security dataset},
  author={Shin, Hyeok-Ki and Lee, Woomyo and Yun, Jeong-Han and Kim, HyoungChun},
  year={2020}
}
```
