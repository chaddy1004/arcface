# ArcFace Implementation
Implementation of [ArcFace](https://arxiv.org/abs/1801.07698) by me!

##  Project Tree
```bash
arface
├── base
│   ├── base_dataloader.py
│   ├── base_model.py
│   ├── base_trainer.py
├── config.yaml
├── dataloader
│   ├── dataloader.py
├── data_processing
│   ├── crop_celeb_a.py
│   └── README.MD
├── model_and_trainer_builder.py
├── network
│   ├── network.py
├── quantize.py
├── README.md
├── trainers
│   └── trainer.py
├── train.py
└── utils
    ├── layers.py
    ├── losses.py
    ├── process_config.py
    ├── quantization.py
    └── valid_helper.py
```

## Running Training Code
```bash
python train.py 
```
(May have to use python3 command depending on the environment)
The information that comes out from training (ex. Tensorboard logs, saved models) will be saved under directory
```bash
<config.experiment_dir>
```
To view Tensorboard,
```buildoutcfg
tensorboard --logdir <config.experiment_dir>/logs
```


## Running Quantization Code
```bash
python quantize.py
```
The quantized model will be saved under directory 
```buildoutcfg
<config.experiment_dir>/saved_model
```

## TODO: Things that I must do 
1. Train on Large Scale dataset and validate
    - So far, I only confirmed that my implementation works by overfitting on very small data. I did not have the resource/time to train it on full data yet.
    - Also would need to add support for other datasets such as CASIA or Trillion Pairs
2. Create separate Dataset for True Positive pairs and True Negative pairs
    - Currently, the dataset is implemented naively where the files are simply shuffled and zipped together. This creates a lot more True Negative samples
3. Try to recreate the results from the paper
4. Compare the performance of original model with quantized model