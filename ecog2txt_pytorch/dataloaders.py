import torch
import torch.utils.data as tdata
import numpy as np
from tfrecord.torch.dataset import TFRecordDataset
from torch.nn.utils.rnn import pad_sequence
from tfrecord_lite import tf_record_iterator


class EcogDataLoader:
    def __init__(self,
                 tfrecord_path,
                 block_config,
                 subject_id,
                 config_obj,
                 vocabulary,
                 index_path=None,
                 description=None
                 ):
        self.tfrecord_path = tfrecord_path
        self.block_config = block_config
        self.subject_id = subject_id
        self.num_ECoG_channels = self.get_num_ecog_channels(config_obj, subject_id, config_obj['DataGenerator'])
        self.index_path = index_path
        self.description = description
        self.vocabulary = vocabulary

    def get_num_ecog_channels(self, config_obj, subject_id, data_generator):
        data_generator_obj = data_generator(config_obj, subject_id)
        return data_generator_obj.num_ECoG_channels

    def transform_fn(self, record):
        # Reshapes the ecog sequence to (len_ecog_sequence , num_channels) 
        record['ecog_sequence'] = record['ecog_sequence'].reshape(-1, self.num_ECoG_channels)
        return record

    def pad_collate(self, batch):
        # print([item['text_sequence'] for item in batch])
        # print("test", ([torch.tensor(item['text_sequence']) for item in batch]))
        return (pad_sequence([torch.tensor(item[key]) for item in batch], batch_first=True)
                for key in ['ecog_sequence', 'text_sequence'])

    def convert_to_str(self, record):
        
        word_ind_map_dict = self.vocabulary.words_ind_map
        record['text_sequence'] = list(
            map(lambda y: float(word_ind_map_dict[y.decode()]) if y.decode() in word_ind_map_dict else
            word_ind_map_dict['<OOV>'], record['text_sequence']))
        record['text_sequence'] = [3.0] + record['text_sequence'] + [1.0]

        return self.transform_fn(record)

    def get_data_loader_for_blocks(self, partition_type='training', batch_size=1, mode='mem'):
        keys = list(self.block_config.keys())
        # filtered_files = list(
            # map(lambda y: self.tfrecord_path + "/EFC" + self.subject_id + "_B" + keys[y] + ".tfrecord", split))
        #         print(filtered_files)

        filtered_files = list(
            map(lambda y: self.tfrecord_path + "/EFC" + self.subject_id + "_B" + y[0] + ".tfrecord", filter(lambda x: x[1]["default_dataset"] == partition_type, self.block_config.items())))

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
            
            #print(len(datasets))
            return tdata.DataLoader(datasets, batch_size=batch_size, collate_fn=self.pad_collate, pin_memory=True)
