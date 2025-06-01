LogiCP Implementation 
===============

This document serves as a step-by-step guide for running the LogiCP codebase. 


### 1. Dependencies
- Python 3.9 and python-dev packages are required. 
- The following additional packages are needed to run this repository: 
matplotlib==3.5.1, numpy==1.21.6, pandas==1.3.5, parsimonious==0.10.0, scikit_learn==1.2.0, scipy==1.10.0, singledispatch==3.7.0, torch==1.11.0, tqdm==4.64.0


### 2. Data Generation 
- TBD.

### 3. Data Preprocessing 
- The script `dataset.py` included in the `data_preprocessing` folder can be run for preprocessing the FHWA dataset. The user can use the command `python3.9 dataset.py` to generate the training dataset. 

### 4. Specification Inference 
- The folder telex includes the code needed for specification inference, where `scorer.py` includes STL metrics and the implementations of Equation 3 defined in the text. 
Additionally, `synth.py` includes the code for generating specifications from STL templates. 

### 5. Network Training and Evaluation 
- To train the FedAvg model on FHWA data, use the following command: 
    ```
    python3.9 main_fedavg.py --mode train --dataset fhwa --client 100 --cluster 0 --frac 0.1
    ```
    Optional parameters:\
    `--mode` Select the mode from these options: `train`, `train-logic`, `eval`\
    `--dataset` fhwa\
    `--client` The number of participating clients \
    `--cluster` The total number of clusters \
    `--frac` The client participation rate\
    `--model` RNN, GRU, or LSTM backbone\
    `--epoch` Number of total communication rounds\
    `--batch_size` The batch size\
    `--max_lr` Maximum learning rate\

- To run the FedProx framework on FHWA dataset, execute the following command: 
```
python3.9 main_fedprox.py --mode train --dataset fhwa --client 100 --cluster 0 --frac 0.1
```

Other framework-specific settings can be found in `options.py`.

### 6. Backbone models implementation
All backbone models, except for Transformer, were implemented in the file `network.py`. For implementations of the Transformer, please refer to `transformer.py`. We thank the following open source repositories for implementations of FL and backbone models:

- https://github.com/ki-ljl/FedProx-PyTorch/blob/main/client.py 
- https://github.com/susmitjha/TeLEX 
- https://github.com/KasperGroesLudvigsen/influenza_transformer 