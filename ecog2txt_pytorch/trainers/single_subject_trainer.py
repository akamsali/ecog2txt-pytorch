from ecog2txt_pytorch.models.transformer import *
from ecog2txt_pytorch.dataloaders import EcogDataLoader
from ecog2txt_pytorch.vocabulary import Vocabulary
from ecog2txt_pytorch.utils.mask import *
from ecog2txt.data_generators import ECoGDataGenerator

from utils_jgm.toolbox import wer_vector

import torch
import yaml
import json
import time
import wandb



class SingleSubjectTrainer:
    def __init__(self, subject_id, manifest_path):
        self.setup(subject_id, manifest_path)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        
        self.manifest['optimizer']['eps'] = float(self.manifest['optimizer']['eps'])
        self.manifest['optimizer']['betas'] = [float(i) for i in self.manifest['optimizer']['betas']]
        self.optimizer = torch.optim.Adam(self.model.parameters(), **self.manifest['optimizer'])

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def setup(self, subject_id, manifest_path):
        with open(manifest_path, "r") as f:
            manifest_file = yaml.load(f)
            self.manifest = manifest_file[int(subject_id)]            
        
        block_config_path = (self.manifest['data'])['block_config_path']
        data_path = (self.manifest['data'])['data_path']
        vocab_path = (self.manifest['data'])['vocab_path']

        with open(block_config_path) as bf:
            block_config_all = json.load(bf)
        
        vocabulary = Vocabulary(vocab_path)
        
        ecog = EcogDataLoader(data_path, block_config_all[subject_id],
                              subject_id, self.manifest, vocabulary, description=self.manifest['data']['description'])
        manifest_model = self.manifest['model']
        self.manifest_model_args = self.manifest['model_args'][manifest_model.__name__]
        
        self.vocab_params = {'src_vocab_size': ecog.num_ECoG_channels,
                             'tgt_vocab_size': len(vocabulary.words_ind_map),
                             'emb_size': self.manifest_model_args['num_out_channels']
                             }
        
        self.pad_idx = vocabulary.words_ind_map['<pad>']

        # no extra files present in EFC400. Change validation to extra if present 
        self.dataloaders = {'train_dataloader':
                                ecog.get_data_loader_for_blocks(batch_size=self.manifest['training']['batch_size'],
                                                                partition_type='training'),
                            'test_dataloader':
                                ecog.get_data_loader_for_blocks(batch_size=self.manifest['training']['batch_size'],
                                                                partition_type='validation'),
                            'val_dataloader':
                                ecog.get_data_loader_for_blocks(batch_size=self.manifest['training']['batch_size'],
                                                                partition_type='validation')
                            }
        self.model = manifest_model(**self.manifest_model_args, **self.vocab_params).to(device)

    def train_epoch(self):
        self.model.train()
        losses = 0
        cnt = 0
        
        for idx, (src, tgt) in enumerate(self.dataloaders['train_dataloader']):
            cnt += 1
            src = src.to(self.device)
            tgt = tgt.transpose(0,1).to(self.device)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, self.manifest_model_args['kernel_size'], tgt_input, self.pad_idx, self.device)

            logits = self.model(src, tgt_input, src_mask, tgt_mask,
                                src_padding_mask, tgt_padding_mask, src_padding_mask)

            self.optimizer.zero_grad()
            tgt_out = tgt[1:, :].type(torch.LongTensor).to(self.device)

            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            self.optimizer.step()
            losses += loss.item()
        return losses / cnt

    def evaluate(self):
        self.model.eval()
        losses = 0.0
        cnt = 0.0
        val_wer = 0.0
        for idx, (src, tgt) in enumerate(self.dataloaders['val_dataloader']):
            cnt += 1
            src = src.to(self.device)
            tgt = tgt.transpose(0,1).to(self.device)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, self.manifest_model_args['kernel_size'], tgt_input,self.pad_idx, self.device)

            logits = self.model(src, tgt_input, src_mask, tgt_mask,
                                src_padding_mask, tgt_padding_mask, src_padding_mask)
            preds = torch.argmax(logits, dim=2)

            tgt_out = tgt[1:, :].type(torch.LongTensor).to(device)

            pred_words = preds.transpose(0, 1).cpu().detach().numpy().astype(str).tolist()
            target_words = tgt_out.transpose(0, 1).cpu().detach().numpy().astype(str).tolist()
            
#             print("target: ", target_words)
#             print("pred: ", pred_words)
#             print("matched: ", target_words[target_words==pred_words])
#             print("matched_shape: ", (target_words[target_words==pred_words]).shape)

            target_words_zero_count = list(map(lambda y: y.count('0'), target_words))
            #print('tgt_zero_count', target_words_zero_count)
            target_zero_dropped = list(map(lambda x: x[1][:len(x[1])-x[0]], zip(target_words_zero_count, target_words)))
            pred_zero_dropped = list(map(lambda x: x[1][:len(x[1])-x[0]], zip(target_words_zero_count, pred_words)))
            wer_vec = wer_vector(target_zero_dropped, pred_zero_dropped)
            val_wer += (sum(wer_vec) / float(len(wer_vec)))

            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()
        return losses / cnt, val_wer / cnt

    def train_and_evaluate(self):
        wandb.init(project='ecog2txt-pytorch', entity='akamsali')

        # Magic
        wandb.watch(self.model, log_freq=1)

        best_val_loss = 10000
        output_dir = self.manifest['training']['output_dir']
        for epoch in range(1, self.manifest['training']['num_epochs'] + 1):
            start_time = time.time()
            train_loss = self.train_epoch()
            end_time = time.time()
            val_loss, val_wer = self.evaluate()
            # change this to val_acc
            if val_loss < best_val_loss:
                torch.save(self.model.state_dict(), output_dir + 'best_model.pt')

            print(
                (f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Val WER: {val_wer:.3f}"
                 f" Epoch time = {(end_time - start_time):.3f}s"))
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "Val WER": val_wer})
