import torch.utils.data as tdata
from tfrecord.torch.dataset import MultiTFRecordDataset


class EcogDataLoader:
    def __init__(self,
                 tfrecord_pattern,
                 block_config,
                 subject_id,
                 index_pattern=None,
                 description=None
                 ):
        self.tfrecord_pattern = tfrecord_pattern
        self.block_config = block_config
        self.subject_id = subject_id
        self.index_pattern = index_pattern
        self.description = description

    def get_data_loader_for_blocks(self, partition_type='training'):

        filtered_blocks = list(map(lambda y : "EFC" + self.subject_id + "_B" + y[0],
                               filter(lambda x : x[1]["default_dataset"] == partition_type,
                                      self.block_config.items())))

        splits = { block: 1.0/len(filtered_blocks) for block in filtered_blocks }

        print("splits", splits)

        dataset = MultiTFRecordDataset(self.tfrecord_pattern,
                                        self.index_pattern, splits, self.description)

        return tdata.DataLoader(dataset, batch_size=1)
