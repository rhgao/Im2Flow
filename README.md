## Im2FLow: Motion Hallucination from Static Images for Action Recognition
Im2Flow: Motion Hallucination from Static Images for Action Recognition: http://vision.cs.utexas.edu/projects/im2flow
[[Project Page]](http://vision.cs.utexas.edu/projects/im2flow/)    [[arXiv]](https://arxiv.org/abs/1712.04109)<br/>

This repository contains the code for our [CVPR 2018 paper Im2Flow](http://vision.cs.utexas.edu/projects/im2flow/gao-cvpr2018.pdf). The code is heavily borrowed from [Phillip Isola's pix2pix Implementation](https://github.com/phillipi/pix2pix).

If you find our code or project useful in your research, please cite:

        @inproceedings{gao2018im2flow,
          title={Im2Flow: Motion Hallucination from Static Images for Action Recognition},
          author={Gao, Ruohan and Xiong, Bo and Grauman, Kristen},
          booktitle={CVPR},
          year={2018}
        }
        
### 1) Preparation
1. Install Torch: http://torch.ch/docs/getting-started.html

2. Clone the repository
  ```Shell
  git clone https://github.com/rhgao/Im2Flow.git
  ```
  
3. Download our pre-trained model using the following script. This model is trained on the UCF-101 dataset.
  ```Shell
  bash model/download_model.sh Im2Flow
  ```
  
### 2) Training
Put video frames under directory `/YOUR_TRAINING_DATA_ROOT/A` and the corresponding ground-truth flow images under directory `/YOUR_TRAINING_DATA_ROOT/B`. 
Once the data is formatted this way, use the following script to generate paired training data:
  ```bash
  python combine_A_and_B.py --fold_A /YOUR_TRAINING_DATA_ROOT/A --fold_B /YOUR_TRAINING_DATA_ROOT/B --fold_AB /YOUR_TRAINING_DATA_ROOT/AB
  ```
Download the pre-trained motion content loss network:
  ```Shell
  bash model/download_model.sh resnet-18_motion
  ```
Use the following command to train Im2Flow network:
  ```
  DATA_ROOT=/YOUR_TRAINING_DATA_ROOT/AB name=flow_UCF_train continue_train=0 save_display_freq=5000 which_direction=AtoB loadSize=286 fineSize=256 batchSize=32 lr=0.0002 print_freq=20 niter=30 save_epoch_freq=5 decay_epoch_freq=10 save_latest_freq=2000 use_GAN=0 lambda_L2=50 lambda_ContentLoss=1 th train.lua |& tee train.log
  ```
  
### 3) Flow Prediction
  ```Shell
  DATA_ROOT=demo_images model_path=model/Im2Flow.t7 th test.lua
  ```
  
### 4) Flow Visualization
  ```Shell
  python visualizeFlow.py --flowImgInputDir results/output/ --rgbImgDir results/input/ --arrowImgOutDir visualization
  ```
