# DSFM

This repository is the source code and experimental data for our ICSE'24 paper entitled
''DSFM: Enhancing Functional Code Clone Detection with Deep Subtree Interactions''.

## Requirements

- Python 3.8
- PyTorch 1.12.1
- javalang 0.13.0
- pycparser 2.20
- Other python requirements can be referred to `requirements.txt`
- Either CPU or GPU is allowed (when using GPU, the batch size should be adapted to the memory)

## Installation

1. Create a virtual python environment for DSFM.

```commandline
conda create -n DSFM python=3.8
```

```commandline
conda activate DSFM
```

2. Install dependencies.

```commandline
cd <dir_of_repo>/DSFM
```

```commandline
pip install -r requirements.txt
```

3. Train, validate and test DSFM on datasets.

```commandline
cd src
```
```commandline
python train_val_test.py --data_dir <dir_of_dataset>
                         --save_dir <dir_to_save_result>
```
where <dir_of_dataset> is the directory contains dataset and <dir_to_save_result> is the 
directory to save the best trained model based on the validation set.

## Datasets

The processed datasets are available at `<datasets>` and the corresponding pretrained models are in `<pretrained_models>`.

We also provide the original datasets for possible future work:

- GCJ: https://www.dropbox.com/s/z8j2dakentazr21/GoogleCodeJam.zip
- OJClone: https://www.dropbox.com/s/m70uwtiwiiy96dm/OnlineJudge.zip
- BigCloneBench: https://www.dropbox.com/s/bi12l22f7lo5ah6/BigCloneBench.zip

The processed datasets can be reproduced by:

```commandline
python prepare_data.py --source_data_dir <original_dir_of_data>
                       --target_data_dir <dir_to_save_data> 
                       --dataset_name <name_of_dataset>
```
where <original_dir_of_data> is the directory of the original dataset, <dir_to_save_data> 
is the directory to save the processed dataset, and <name_of_dataset> is the name of dataset 
which includes GCJ, OJClone, and BigCloneBench.

