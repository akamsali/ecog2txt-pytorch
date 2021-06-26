import torch
import torch.utils.data as tdata
from tfrecord.torch.dataset import TFRecordDataset
from torch.nn.utils.rnn import pad_sequence


class EcogDataLoader:
    def __init__(self,
                 tfrecord_path,
                 block_config,
                 subject_id,
                 num_ECoG_channels,
                 index_path=None,
                 description=None
                 ):
        self.tfrecord_path = tfrecord_path
        self.block_config = block_config
        self.subject_id = subject_id
        self.num_ECoG_channels = num_ECoG_channels
        self.index_path = index_path
        self.description = description

    def transform_fn(self, record):
        # Reshapes the ecog sequence to (len_ecog_sequence , num_channels)
        record['ecog_sequence'] = record['ecog_sequence'].reshape(-1, self.num_ECoG_channels)
        return record

    def pad_collate(self, batch):
        x = (pad_sequence([torch.tensor(item[key]) for item in batch]) for key in ['ecog_sequence','text_sequence'])
        return x

    def get_data_loader_for_blocks(self, batch_size=1, partition_type='training'):
        filtered_files = list(map(lambda y: self.tfrecord_path + "/EFC" + self.subject_id + "_B" + y[0] + ".tfrecord",
                                   filter(lambda x: x[1]["default_dataset"] == partition_type,
                                          self.block_config.items())))

        dataloaders = []
        for f in filtered_files:
          dataset = TFRecordDataset(f, self.index_path, self.description,
                                       transform=self.transform_fn)

          dataloaders.append(tdata.DataLoader(dataset, batch_size=batch_size, collate_fn=self.pad_collate))
        
        return dataloaders
