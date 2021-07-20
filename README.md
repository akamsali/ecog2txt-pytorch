# ecog2txt-pytorch
Implementation of transformer model for Neural data decoding into words.

## Model

Convolution layer(`Conv1d`) before passing to the Encoder Layer of the Transformer.

Transformer[1]:
https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer

### Masks

* Padding values are masked at both source and target to ensure no attention is paid to them by the model.
* To keep the auto-regressive property intact(not consider future elements of sequence to predict the current output),
  target is masked using a lower triangular mask.

#### Decoder mask while training

```
train_tgt_mask tensor(
        [[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0., 0., 0., 0.]], device='cuda:0')
train_tgt_padding_mask tensor(
        [[False, False, False, False, False, False,  True,  True],
        [False, False, False, False, False, False, False, False]],
        device='cuda:0')
train_tgt_mask tensor(
        [[0., -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf, -inf],
        [0., 0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0., 0.]], device='cuda:0')
train_tgt_padding_mask tensor(
        [[False, False, False, False, False, False],
        [False, False, False, False, False, False]], device='cuda:0')
```

#### Decoder mask while evaluating

```
evaluate_tgt_mask tensor(
        [[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0., 0., 0., 0.]], device='cuda:0')
evaluate_tgt_padding_mask tensor(
        [[False, False, False, False, False, False,  True,  True],
        [False, False, False, False, False, False, False, False]],
       device='cuda:0')
evaluate_tgt_mask tensor(
        [[0., -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf, -inf],
        [0., 0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0., 0.]], device='cuda:0')
evaluate_tgt_padding_mask tensor(
        [[False, False, False, False, False, False],
        [False, False, False, False, False, False]], device='cuda:0')
```

This ensures that the predictions for position i can depend only on the known outputs at positions less than i.

## Trainer

Single subject trainer for ecog to word sequence.

### Data Loader

Use the following library for reading TFRecords as PyTorch dataloaders:
https://github.com/jongwook/tfrecord_lite using Leave One Out Cross-Validation(LOOCV) block configuration.

The flattened ecog sequence data is reshaped based on the number of good ecog channels.

All ecog sequences in a batch are padded to have the same sequence length.

All text sequences in a batch are padded by `'<EOS>'` and then with
`'<pad>'` to have the same sequence length.

### Loss Function

Cross Entropy Loss

### Observed Metrics (per epoch)

* Training loss
* Validation loss
* Validation WER: average(word\_error\_rate) [taken from `utils_jgm`]

## Results
Cross-validation results by validation-block type 
[here](https://github.com/akamsali/ecog2txt-pytorch/blob/main/notebooks/test_pytorch_transformer.ipynb).

## References

1. [Vaswani, Ashish, et al. "Attention is all you need."
   Advances in neural information processing systems. 2017.](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

## Usage Instructions

**#TODO**


