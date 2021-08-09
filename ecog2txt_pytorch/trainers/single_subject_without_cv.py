from ecog2txt_pytorch.models.transformer import *
from ecog2txt_pytorch.dataloaders import EcogDataLoader
from ecog2txt_pytorch.vocabulary import Vocabulary
from ecog2txt_pytorch.utils.mask import *
from ecog2txt_pytorch.utils.training_utils import reset_weights

from utils_jgm.toolbox import wer_vector
from ecog2txt.data_generators import ECoGDataGenerator

import torch
import yaml
import json
import time
from sklearn.model_selection import LeaveOneOut
import wandb


class SingleSubjectTrainer:
    def __init__(self, subject_id, manifest_path):
        self.setup(subject_id, manifest_path)

        self.self.device = torch.self.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.pad_idx)

        self.manifest['optimizer']['eps'] = float(self.manifest['optimizer']['eps'])
        self.manifest['optimizer']['betas'] = [float(i) for i in self.manifest['optimizer']['betas']]
        self.optimizer = torch.optim.Adam(self.model.parameters(), **self.manifest['optimizer'])

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def setup(self, subject_id, manifest_path):
        self.subject_id = subject_id
        with open(manifest_path, "r") as f:
            manifest_file = yaml.load(f)
            self.manifest = manifest_file[int(self.subject_id)]

        block_config_path = (self.manifest['data'])['block_config_path']
        data_path = (self.manifest['data'])['data_path']
        vocab_path = (self.manifest['data'])['vocab_path']

        with open(block_config_path) as bf:
            block_config_all = json.load(bf)

        vocabulary = Vocabulary(vocab_path)

        self.block_config = block_config_all[self.subject_id]
        self.ecog = EcogDataLoader(data_path, self.block_config,
                                   subject_id, self.manifest, vocabulary,
                                   description=self.manifest['data']['description'])
        manifest_model = self.manifest['model']
        self.manifest_model_args = self.manifest['model_args'][manifest_model.__name__]

        self.vocab_params = {'src_vocab_size': self.ecog.num_ECoG_channels,
                             'tgt_vocab_size': len(vocabulary.words_ind_map),
                             'emb_size': self.manifest_model_args['num_out_channels']
                             }

        self.pad_idx = vocabulary.words_ind_map['<pad>']
        self.eos_idx = vocabulary.words_ind_map['<EOS>']
        self.sos_idx = vocabulary.words_ind_map['<SOS>']

        self.word_ind = vocabulary.words_ind_map
        self.ind_word_map_dict = vocabulary.ind_words_map
        # no extra files present in EFC400. Change validation to extra if present 
        self.dataloaders = {'train_dataloader':
                                self.ecog.get_data_loader_for_blocks(batch_size=self.manifest['training']['batch_size'],
                                                                partition_type='training'),
                            'test_dataloader':
                                self.ecog.get_data_loader_for_blocks(batch_size=self.manifest['training']['batch_size'],
                                                                partition_type='validation'),
                            'val_dataloader':
                                self.ecog.get_data_loader_for_blocks(batch_size=self.manifest['training']['batch_size'],
                                                                partition_type='validation')
                            }
        
        self.model = manifest_model(**self.manifest_model_args, **self.vocab_params).to(self.device)

    def train_epoch(self):
        self.model.train()
        losses = 0
        cnt = 0

        for idx, (src, tgt) in enumerate(self.dataloaders['train_dataloader']):
            print('src_shape',src.shape, 'tgt_shape', tgt.shape)
            cnt += 1
            src = src.to(self.self.device)
            tgt = tgt.transpose(0, 1).to(self.self.device)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, self.manifest_model_args[
                'kernel_size'], tgt_input, self.pad_idx, self.self.device)


            logits = self.model(src, tgt_input, src_mask, tgt_mask,
                                src_padding_mask, tgt_padding_mask, src_padding_mask)

            self.optimizer.zero_grad()
            tgt_out = tgt[1:, :].type(torch.LongTensor).to(self.self.device)

            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            self.optimizer.step()
            losses += loss.item()
        return losses / cnt

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            cnt = 0
            losses = 0.0
            for idx, (src, tgt) in enumerate(self.dataloaders['val_dataloader']):
                cnt += 1
                src = src.to(self.self.device)
                tgt = tgt.transpose(0, 1).to(self.self.device)

                tgt_input = tgt[:-1, :]

                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, self.manifest_model_args[
                    'kernel_size'], tgt_input, self.pad_idx, self.self.device)

                
                logits = self.model(src, tgt_input, src_mask, tgt_mask,
                                    src_padding_mask, tgt_padding_mask, src_padding_mask)
                preds = torch.argmax(logits, dim=2)

                tgt_out = tgt[1:, :].type(torch.LongTensor).to(self.device)

                pred_words = preds.transpose(0, 1).cpu().detach().numpy().astype(str).tolist()
                target_words = tgt_out.transpose(0, 1).cpu().detach().numpy().astype(str).tolist()

                # drop after '<EOS>' (='1')
                # print('tgt words: ', target_words)
                target_words_EOS_pos = list(map(lambda y: y.index('1'), target_words))
                # print('position: ', target_words_EOS_pos)
                target_pad_dropped = list(
                    map(lambda x: x[1][:x[0] + 1], zip(target_words_EOS_pos, target_words)))
                # print('tgt_dropped_ind: ', target_pad_dropped)

                # print('pred words: ', pred_words)
                pred_words_EOS_pos = list(map(lambda y: y.index('1') if '1' in y else -1, pred_words))
                # print('position: ', pred_words_EOS_pos)
                pred_pad_dropped = list(
                    map(lambda x: x[1][:x[0] + 1], zip(pred_words_EOS_pos, pred_words)))
                # print('pred_dropped: ', pred_pad_dropped)
                
                tgt_ind_word_map = [list(map(lambda t: self.ind_word_map_dict[int(t)], tgt_sentence)) for tgt_sentence in target_pad_dropped]
#                 print('tgt_dropped_words: ', tgt_ind_word_map)
                pred_ind_word_map = [list(map(lambda t: self.ind_word_map_dict[int(t)], pred_sentence)) for pred_sentence in pred_pad_dropped]
#                 print('pred_dropped_words: ', pred_ind_word_map)
                ind_to_words = {'tgt': [], 'pred': []}
                ind_to_words['tgt'].append(tgt_ind_word_map)
                ind_to_words['pred'].append(pred_ind_word_map)
                
                wer_vec = wer_vector(target_pad_dropped, pred_pad_dropped)
                # print('wer_vec: ', wer_vec)
                val_wer += (sum(wer_vec) / float(len(wer_vec)))

                loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
                losses += loss.item()
            
            return losses / cnt


    def greedy_decode(self, model, src, src_mask, max_len, start_symbol):
        src = src.to(self.device)
        src_mask = src_mask.to(self.device)


        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)
        for i in range(max_len-1):
            memory = memory.to(self.device)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(self.device)
            out = model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            print('next_word: ', next_word )
            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            print('ys: ', ys)
            if next_word == self.eos_idx:
                break

            memory = model.encode(src, src_mask)

    def inference(self):
        cnt = 0.0
        val_wer = 0.0
        ind_to_words = {}

        self.model.eval()
        
        src = EcogDataLoader.transform_fn((src_sentence).view(-1, 1))
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = greedy_decode(
            model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()

        return words, wer


    def train_and_evaluate(self):
        
        best_val_loss = 100000
        tgt_pred_words = {'tgt': [], 'pred': []}
        metrics = {'train_loss': [], 'val_loss': [], 'val_WER': []}
        
        for epoch in range(1, self.manifest['training']['num_epochs'] + 1):
            start_time = time.time()
            train_loss = self.train_epoch()
            end_time = time.time()

            val_loss = self.evaluate()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model, '/scratch/gilbreth/akamsali/Research/Makin/outputs/best_models/best_model_bothmocha_2' + '.pt') 
            
            
            
            tgt_pred_words['tgt'].append(ind_to_words['tgt'])
            tgt_pred_words['pred'].append(ind_to_words['pred'])
            metrics['val_loss'].append(val_loss)
            metrics['train_loss'].append(train_loss)
            metrics['val_WER'].append(val_wer)
            
            # wandb.log({'Train_loss': train_loss, 'Val_loss': val_loss, 'Val_WER': val_wer})
            if not epoch%10: 
                print(f"epoch_{epoch}: Train_loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Val WER: {val_wer:.3f}, Epoch time = {(end_time - start_time):.3f}s")
            #print('words: ', tgt_pred_words[block_left_out])
            if not epoch%100:
                print('tgt: ', tgt_pred_words['tgt'][-1])
                print('pred: ', tgt_pred_words['pred'][-1])
                    
        return tgt_pred_words, metrics