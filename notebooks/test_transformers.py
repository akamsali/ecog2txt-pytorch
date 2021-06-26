#%%

import torch

from ecog2txt_pytorch.dataloaders import EcogDataLoader
import json

#%%

subject_id = "400"
tfrecord_path="/scratch/gilbreth/akamsali/Research/Makin/data/ecog2txt/word_sequence/tf_records/EFC400"
block_config_path="/scratch/gilbreth/akamsali/Research/Makin/ecog2txt-pytorch/conf/block_breakdowns.json"
manifest_path="/scratch/gilbreth/akamsali/Research/Makin/ecog2txt-pytorch/conf/mocha-1_word_sequence.yaml"

#%%

# get the number of channels
import yaml
import ecog2txt.data_generators
with open(manifest_path, "r") as f:
    manifest_file = yaml.load(f)
    manifest_obj = manifest_file[int(subject_id)]

_DG_kwargs = {}
json_dir = manifest_obj['json_dir']
DataGenerator = manifest_obj['DataGenerator']
data_generator = DataGenerator(manifest_obj, subject_id, **dict(_DG_kwargs))

num_channels = data_generator.num_ECoG_channels

#%%

from ecog2txt_pytorch.vocabulary import Vocabulary

word_seq_vocabulary = Vocabulary("/scratch/gilbreth/akamsali/Research/Makin/ecog2txt-pytorch/conf/vocab.mocha-timit.1806")
SRC_VOCAB_SIZE = num_channels
TGT_VOCAB_SIZE = len(word_seq_vocabulary.words_ind_map)
EMB_SIZE = num_channels
PAD_IDX = word_seq_vocabulary.words_ind_map['<pad>']
#print("pad_idx", PAD_IDX)
EOS_IDX = word_seq_vocabulary.words_ind_map['<EOS>']
BATCH_SIZE = 16

WIN_SIZE = 1

block_config_all = None
with open(block_config_path) as bf:
    block_config_all = json.load(bf)

description = {"audio_sequence": "float", "ecog_sequence": "float",
               "text_sequence": "byte", "phoneme_sequence": "byte"}
ecog = EcogDataLoader(tfrecord_path, block_config_all[subject_id],
                      subject_id, num_ECoG_channels=num_channels, description=description)
#print(ecog.get_data_loader_for_blocks())


train_dataloaders = ecog.get_data_loader_for_blocks(batch_size=BATCH_SIZE, partition_type='training')
test_dataloaders = ecog.get_data_loader_for_blocks(batch_size=BATCH_SIZE, partition_type='extra')
valid_dataloaders = ecog.get_data_loader_for_blocks(batch_size=BATCH_SIZE, partition_type='validation')

#%%

from ecog2txt_pytorch.models.single_subject_transformer import *
#from longformer.longformer import LongformerSelfAttention, LongformerConfig
import torch.nn.functional as F
#from jgm_utils.toolbox import wer


# longformer_config = LongformerConffg(attention_window=[WIN_SIZE] * NUM_ENCODER_LAYERS,
#  attention_dilation=[1] * NUM_ENCODER_LAYERS,
#  hidden_size=EMB_SIZE,
#  num_attention_heads=NHEAD)

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
 EMB_SIZE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
 FFN_HID_DIM)

# for i, layer in enumerate(transformer.transformer_encoder.layers):
#  layer.self_attn = LongformerSelfAttention(config=longformer_config, layer_id=i)


for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(device)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(
    transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
)


#%%

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
  src_seq_len = src.shape[0]
  tgt_seq_len = tgt.shape[0]

  tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
  src_mask = torch.zeros((src_seq_len, src_seq_len), device=device)
  #making a mask with sliding window centred around i
  # ind_src = torch.arange(src_seq_len+WIN_SIZE-1, dtype=torch.int64).unfold(0,WIN_SIZE,1) - WIN_SIZE/2
  # ind_src[ind_src<0] = 0
  # ind_src[ind_src>=src_seq_len] = src_seq_len - 1
  # ind_src = ind_src.type(torch.int64)
  # src_mask.scatter_(1,ind_src,1)
  # print('SRC_MASK', src_mask, 'SRC_mask_shape', src_mask.shape)
  # src_mask = src_mask.float().masked_fill(src_mask == 0, float('-inf')).masked_fill(src_mask == 1, float(0.0))

  #print("src shape", src.shape)
  src_padding_mask = torch.zeros(src.shape[:-1], device=device).transpose(0, 1)
  tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
  return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def train_epoch(model, train_dataloaders, optimizer):
  model.train()
  losses = 0
  cnt = 0
  train_acc = 0
  for train_dataloader in train_dataloaders:
      for idx, (src, tgt) in enumerate(train_dataloader):
          #print('src_shape', src.shape, 'tgt_shape', tgt.shape)
          cnt += 1
          src = src.to(device)
          tgt = tgt.to(device)

          tgt_input = tgt[:-1, :]

          src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

          logits = model(src, tgt_input, src_mask, tgt_mask,
                                    src_padding_mask, tgt_padding_mask, src_padding_mask)

          optimizer.zero_grad()
          tgt_out = tgt[1:,:].type(torch.LongTensor).to(device)

          loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
          loss.backward()
          optimizer.step()
          losses += loss.item()
      
  return losses / cnt


def evaluate(model, val_dataloaders):
  model.eval()
  losses = 0
  cnt = 0
  val_accuracy = 0
  for val_dataloader in val_dataloaders:
      for idx, (src, tgt) in enumerate(val_dataloader):
        cnt += 1
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                                  src_padding_mask, tgt_padding_mask, src_padding_mask)
        preds = torch.argmax(logits, dim=2)


        tgt_out = tgt[1:,:].type(torch.LongTensor).to(device)
        
        val_accuracy += ( torch.sum(tgt_out == preds) / (preds.shape[0] * preds.shape[1]) )
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()


  return losses / cnt, val_accuracy / cnt

#%%

import time
for epoch in range(1, NUM_EPOCHS+1):
  start_time = time.time()
  train_loss = train_epoch(transformer, train_dataloaders, optimizer)
  end_time = time.time()
  val_loss, val_acc = evaluate(transformer, valid_dataloaders)
  print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Val accuracy: {val_acc:.3f}"
          f"Epoch time = {(end_time - start_time):.3f}s"))


#%%


