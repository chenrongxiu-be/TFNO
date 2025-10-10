# TFNO
This repository provides the official implementation of the paper:

* [TFNO: Real-time surrogate model of vehicle-bridge interaction based on a novel multiple-input neural operator](https://doi.org/10.1016/j.autcon.2025.106576)

In this work, we propose a novel multiple-input neural operator, TFNO, which integrates the Transformer architecture with the Fourier Neural Operator. TFNO is developed as a surrogate model for the vehicle–bridge interaction system, enabling accurate and real-time prediction of bridge dynamic deflection responses.

## Requirements
The implementation is based on Python with the following dependencies:
* numpy, pytorch, einops, matplotlib

## Usage
**1. Data preparation**

Download the datasets from the provided URL and place them in the ./data/ directory, e.g., ./data/continuous.

**2. Model training**

Take TFNO training as an example:

1. Configure model architecture, training setups, and dataset, etc. in ./config/config_tfno.ini*.
2. Configure already-provided node_num, target_datasets, target_node, and target_axis in train_tfno.py, . 
2. Run train_tfno.py

**The configuration files (.ini) included in this repository correspond to the experimental setups reported in the paper.*

**Optional: transfer learning**

To enable transfer learning after pre-training:

1. In ./config/config_tfno.ini, set transfer_learning = true and specify title_pretrain.
2. In train_tfno.py, configure already-provided node_num, target_datasets, target_node, and target_axis.
3. Run train_tfno.py

**3. Visualization (evaluation):**
1. Adjust evaluation configurations in plot.py.
2. Run plot.py

## Datasets
The datasets used in the paper are provided in .npy (NumPy array) format and are directly loadable within the scripts.

[Datasets](https://stkyotouac-my.sharepoint.com/:f:/g/personal/chen_rongxiu_72m_st_kyoto-u_ac_jp/EqVT_FXa-QFPlRYmqgdJ1LsB5lIo_yC6o6x9LH0S1hYSVQ?e=K0oXYH)*

**Shared via OneDrive. If you experience access issues, please try different devices or web browsers (Google Chrome is recommended).* 

## Models
We provide pre-trained models for the continuous bridge deck case, including TFNO and the baseline models (Fourier-MIONet, LSTM, and NARX).

[Pre-trained models](https://stkyotouac-my.sharepoint.com/:f:/g/personal/chen_rongxiu_72m_st_kyoto-u_ac_jp/EpS5Hb9EbhJBgeZTF_GAUrYBqf35rr4wR9d7Vg3mtSIYAw?e=p7ZNhZ)*

**Shared via OneDrive. If you experience access issues, please try different devices or web browsers (Google Chrome is recommended).* 

## Support
For questions regarding the datasets or code usage, please open a discussion in the **GitHub Issues** section.

## Citing

```
@article{CHEN2025106576,
  title = {Real-time surrogate model of vehicle-bridge interaction using a multiple-input neural operator},
  journal = {Automation in Construction},
  volume = {180},
  pages = {106576},
  year = {2025},
  issn = {0926-5805},
  doi = {https://doi.org/10.1016/j.autcon.2025.106576}
}
```

