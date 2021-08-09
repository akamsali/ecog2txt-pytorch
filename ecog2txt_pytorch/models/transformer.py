import torch
from torch import nn, Tensor
from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)
import math



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Transformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 emb_size: int, src_vocab_size: int, tgt_vocab_size: int, num_out_channels: int, 
                 kernel_size: int, dim_feedforward:int = 512, dropout:float = 0.1, nhead: int = 8):
        super(Transformer, self).__init__()
        self.conv_layer = torch.nn.Conv1d(in_channels=src_vocab_size, 
                        out_channels=emb_size, kernel_size=kernel_size)
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=nhead,
                                                dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, 
                                    num_layers=num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=nhead,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, 
                                    num_layers=num_decoder_layers)

        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        # self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor,
                tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
#         print('src shape: ', src.shape)
        src = src.transpose(-2, -1)

        src_conv = self.conv_layer(src)
#         print('src_conv_to_encoder', src_conv.transpose(1, 2).transpose(0,1).shape)
        src_emb = self.positional_encoding(src_conv.transpose(1, 2).transpose(0,1))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))

        #print('src_emb_shape: ', src_emb.shape,'tgt_emb_shape: ',tgt_emb.shape ,'src_mask_shape: ',
        #      src_mask.shape, 'src_pad_shape: ',src_padding_mask.shape)
        memory = self.transformer_encoder(src_emb, src_mask)
        #print("here af tr enc")
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)
        #print("here af tr dec")

        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer_encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer_decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0),:])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        #print('tokens shape: ', tokens.shape)
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)



