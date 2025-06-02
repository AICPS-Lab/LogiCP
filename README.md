LogiCP Implementation 
===============

“This GitHub repository provides the implementation code for the LogiCP algorithm.”


### 1. Dependencies
- Python 3.9 and python-dev packages are required. 
- The following additional packages are needed to run this repository: 
matplotlib==3.5.1, numpy==1.21.6, pandas==1.3.5, parsimonious==0.10.0, scikit_learn==1.2.0, scipy==1.10.0, singledispatch==3.7.0, torch==1.11.0, tqdm==4.64.0

<!-- ### 2. Data Preprocessing 
- The script `dataset.py` included in the `data_preprocessing` folder can be run for preprocessing the FHWA dataset. The user can use the command `python dataset.py` to generate the training dataset.  -->

<!-- ### 3. Specification Inference 
- The folder telex includes the code needed for specification inference, where `scorer.py` includes STL metrics and the implementations of Equation 3 defined in the text. 
Additionally, `synth.py` includes the code for generating specifications from STL templates.  -->
### 2. Dataset 
For the original city temperature dataset used in the paper, please refer to the following citation and link.

University of Dayton, "Temperature Data from Around the World Attracts Web Visitors to University of Dayton Site" (2002). News
Releases. 9948.
https://ecommons.udayton.edu/news_rls/9948

### 2. Distributed CP Implementation

For the distributed CP implementation, specifically, the computation of quantiles at both the client and cluster levels, please refer to compute.py. 

The original source of this implementation is adapted from: https://github.com/pierreHmbt/FedCP-QQ

### 3. Network Training and Evaluation 
- To train the LogiCP model on FHWA data, use the following command: 

    For pretrain process, run: 
    ```
    python main.py --mode pretrain_calib --dataset fhwa --client 50 --cluster 5 --frac 1 --sep_type spec_m --cp_epoch 30
    ```

    For training process, run:
    ```
    python main.py --mode train_cp --dataset fhwa --client 50 --cluster 5 --frac 1 --sep_type spec_m --cp_epoch 30
    ```

    For evaluation, run: 
    ```
    python eval.py --mode eval --dataset fhwa --client 50 --cluster 5 --frac 1 --sep_type spec_m --cp_epoch 30
    ```

- To train the LogiCP model on CT data, use the following command: 

    For pretrain process, run: 
    ```
    python main.py --mode pretrain_calib --dataset ct --client 50 --cluster 5 --frac 1 --sep_type spec_m --cp_epoch 30
    ```

    For training process, run:
    ```
    python main.py --mode train_cp --dataset ct --client 50 --cluster 5 --frac 1 --sep_type spec_m --cp_epoch 30
    ```

    For evaluation, run: 
    ```
    python eval.py --mode eval --dataset ct --client 50 --cluster 5 --frac 1 --sep_type spec_m --cp_epoch 30
    ```
### 4. Parameter Descriptions
We provide a description of the parameters implemented in option.py below:
`--mode` Select the mode from these options: `pretrain_calib`, `train_cp`, `eval`\
`--sep_type` The way to determine cluster: `spec_m`, `value`\ 
`--dataset` `fhwa`, `ct`\
`--client` The number of participating clients \
`--cluster` The total number of clusters \
`--frac` The client participation rate\
`--model` RNN, GRU, LSTM or Transformer backbone\
`--cp_epoch` Number of total communication rounds\
`--batch_size` The batch size\
`--max_lr` Maximum learning rate\

For additional configuration details, please refer directly to option.py.

### 5. Backbone models implementation
RNN, LSTM, GRU, Transformer were implemented in the file `network.py` and `transform.py`. 


“We would like to thank the following open-source repositories for their implementations of FL frameworks, CP implementation and Backbone Models:”

- https://github.com/pierreHmbt/FedCP-QQ
- https://github.com/susmitjha/TeLEX 
- https://github.com/KasperGroesLudvigsen/influenza_transformer 