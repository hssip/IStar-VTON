# IStar-VTON

Project and work for Virtual Try-on Task by Imagination Scenario and Target Reconstruction.
More codes are uploading ...

## Requirements

* The python modules (including the reproducing of the baseline methods) needed in this work are released in ``requirements.txt``
* The installation instructions for the [adan](https://github.com/sail-sg/adan) package can be found in the corresponding project.

## Pre-process

* All the pre-process codes are packaged and released in  `/pre-process`.
* The features of`agnostic-v3.2` in [VITON-HD](https://github.com/shadow2496/VITON-HD) dataset is same to `agnositc` in [DressCode](https://github.com/aimagelab/dress-code) and [StreetTryOn]([https://](https://cuiaiyu.github.io/StreetTryOn)) in this project.
* `cloth-mask`  codes and weights are released `/pre-process/garment_detection`
* `agnostic-mask` codes are released in `/pre-process/agnostic_mask`
* `scenario-description` code are released in `/pre-process/scenario_description`

## Training

* prepare used model weight and put the weight path to train bash `train.sh`


`bash train.sh`


## Inference(Generation)

`bash generate.sh`


## Metrics

* prepare used model weight in `eval_src`
```
cd eval_src
bash eval.sh
```

## Acknowledgements

Some components of pre-process codes in this project are adapted from other open-source projects with necessary modifications. The copyrights and licenses of the original projects remain with their respective authors. Please refer to and comply with the original licenses when using the corresponding code.
