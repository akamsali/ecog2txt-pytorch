import torch
import torch.utils.data as tdata
from tfrecord.torch.dataset import TFRecordDataset
from torch.nn.utils.rnn import pad_sequence
from tfrecord_lite import tf_record_iterator


class EcogDataLoader:
    def __init__(self,
                 tfrecord_path,
                 block_config,
                 subject_id,
                 data_generator,
                 config_obj,
                 index_path=None,
                 description=None
                 ):
        self.tfrecord_path = tfrecord_path
        self.block_config = block_config
        self.subject_id = subject_id
        self.num_ECoG_channels = self.get_num_ecog_channels(config_obj, subject_id, data_generator)
        self.index_path = index_path
        self.description = description

    def get_num_ecog_channels(self, config_obj, subject_id, data_generator):
        data_generator_obj = data_generator(config_obj, subject_id)
        return data_generator_obj.num_ECoG_channels

    def transform_fn(self, record):
        # Reshapes the ecog sequence to (len_ecog_sequence , num_channels)
        record['ecog_sequence'] = record['ecog_sequence'].reshape(-1, self.num_ECoG_channels)
        return record

    def pad_collate(self, batch):
        x = (pad_sequence([torch.tensor(item[key]) for item in batch]) for key in ['ecog_sequence', 'text_sequence'])
        return x

    def convert_to_str(self, record):
        record['text_sequence'] = list(map(lambda y: y.decode(), record['text_sequence']))
        self.transform_fn(record)

        return record

    def get_data_loader_for_blocks(self, batch_size=1, partition_type='training', mode='mem'):
        filtered_files = list(map(lambda y: self.tfrecord_path + "/EFC" + self.subject_id + "_B" + y[0] + ".tfrecord",
                                  filter(lambda x: x[1]["default_dataset"] == partition_type,
                                         self.block_config.items())))
        if mode == 'disk':
            # print("partition_type", partition_type, "filtered_files", filtered_files)
            datasets = []
            for f in filtered_files:
                dataset = TFRecordDataset(f, self.index_path, self.description,
                                          transform=self.transform_fn)
                datasets.append(dataset)

            concat_dataset = tdata.ChainDataset(datasets)
            return tdata.DataLoader(concat_dataset, batch_size=batch_size, collate_fn=self.pad_collate, pin_memory=True)

        elif mode == 'mem':
            datasets = []
            for f in filtered_files:
                dataset_iterator = tf_record_iterator(f)
                datasets.extend([self.convert_to_str(d) for d in dataset_iterator])

            return tdata.DataLoader(datasets, batch_size=batch_size, collate_fn=self.pad_collate, pin_memory=True)
