import torch
import torch.utils.data as tdata
from tfrecord.torch.dataset import MultiTFRecordDataset
from torch.nn.utils.rnn import pad_sequence


class EcogDataLoader:
    def __init__(self,
                 tfrecord_pattern,
                 block_config,
                 subject_id,
                 num_ECoG_channels,
                 index_pattern=None,
                 description=None
                 ):
        self.tfrecord_pattern = tfrecord_pattern
        self.block_config = block_config
        self.subject_id = subject_id
        self.num_ECoG_channels = num_ECoG_channels
        self.index_pattern = index_pattern
        self.description = description

    def transform_fn(self, record):
        # Reshapes the ecog sequence to (len_ecog_sequence , num_channels)
        record['ecog_sequence'] = record['ecog_sequence'].reshape(-1, self.num_ECoG_channels)
        return record

    def pad_collate(self, batch):
        x = (pad_sequence([torch.tensor(item[key]) for item in batch]) for key in ['ecog_sequence','text_sequence'])
        return x

    def get_data_loader_for_blocks(self, batch_size=1, partition_type='training'):
        filtered_blocks = list(map(lambda y: "EFC" + self.subject_id + "_B" + y[0],
                                   filter(lambda x: x[1]["default_dataset"] == partition_type,
                                          self.block_config.items())))

        splits = {block: 1.0 / len(filtered_blocks) for block in filtered_blocks}


        dataset = MultiTFRecordDataset(self.tfrecord_pattern,
                                       self.index_pattern, splits, self.description,
                                       transform=self.transform_fn)

        return tdata.DataLoader(dataset, batch_size=batch_size, collate_fn=self.pad_collate)
