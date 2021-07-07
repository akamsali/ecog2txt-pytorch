import torch

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, pad_idx, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device)
    # making a mask with sliding window centred around i
    # ind_src = torch.arange(src_seq_len+WIN_SIZE-1, dtype=torch.int64).unfold(0,WIN_SIZE,1) - WIN_SIZE/2
    # ind_src[ind_src<0] = 0
    # ind_src[ind_src>=src_seq_len] = src_seq_len - 1
    # ind_src = ind_src.type(torch.int64)
    # src_mask.scatter_(1,ind_src,1)
    # print('SRC_MASK', src_mask, 'SRC_mask_shape', src_mask.shape)
    # src_mask = src_mask.float().masked_fill(src_mask == 0, float('-inf')).masked_fill(src_mask == 1, float(0.0))

    # print("src shape", src.shape)
    src_padding_mask = torch.zeros(src.shape[:-1], device=device).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
