# ecog2txt-pytorch

## Initial Plan

### Data Loader
Use the following library for reading TFRecords as PyTorch dataloaders:
https://github.com/vahidk/tfrecord

### Trainer
Make a single subject trainer for ecog to word sequence/phoneme

### Model
Transformer: 
https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer

### Training Loop
Add support for batch loading in `EcogDataLoader` and parallelized(GPU/distributed) training.


## Usage Instructions

**#TODO**


