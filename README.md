# VAE image anomaly detection

----

This is the implementation of the VAE and AE part of the course project in CMU 10-708.



## Setup

1. Install requirements by
   ```
   pip3 install -r requirements.txt
   ```

2. install the PyTorch 1.11.0 and PyTroch Vision 0.12.0 from https://pytorch.org/

3. Download MvTec dataset from https://www.mvtec.com/company/research/datasets/mvtec-ad, and put the unzipped folder under `datasets/mvtec_anomaly_detection` (otherwise, you need to change `dataset.root` in config files to the of MvTec dataset.)


## Run

All the config files for training and testing are in `configs/*.yaml`.

To train a model:

```
python3 main.py -c <path_to_training_config_file>

# example 1: to train a VAE model with Normalizing flow on MNIST:
python3 main.py -c configs/mnist_train_vae_nf.yaml

# example 2: to train a VT-AE model on MvTec:
python3 main.py -c configs/mvtec_train_vtae.yaml
```

To test a model:

1. change `model.load_path` in `config/mvtec_test.yaml` or `config/mnist_test.yaml` to the checkpoint file (ends with `.pth`)
2. run 
   ```
   python3 -t -c config/mnist_test.yaml  
   ```
   
   or
	```
   python3 -t -c config/mvtec_test.yaml  
	```