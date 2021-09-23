from torch.utils.data.dataset import Subset
import numpy as np
import json
import tqdm

# Thanks for StructTrans
# https://github.com/jingraham/neurips19-graph-protein-design
class StructureDataset():
    def __init__(self, jsonl_file, truncate=None, 
                max_length=100, alphabet='ACDEFGHIKLMNPQRSTVWY'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
        }
        with open(jsonl_file) as f:
            lines = f.readlines()
        self.data = []

        for line in tqdm.tqdm(lines):
            entry = json.loads(line)
            seq = entry['seq']

            for key, val in entry['coords'].items():
                entry['coords'][key] = np.asarray(val)

            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:  self.data.append(entry)
                else:  discard_count['too_long'] += 1
            else:  discard_count['bad_chars'] += 1

            if truncate is not None and len(self.data) == truncate:  return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class StructureLoader():
    def __init__(self, dataset, batch_size=100, **kwargs):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)[::-1]

        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}

        clusters, batch = [], []
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
            else:
                clusters.append(batch)
                batch = []
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch
        
class CATH_Loader:
    def __init__(self, json_path, split_path, batch_tokens, test_split_path=None, **kwargs):
        self.batch_tokens = batch_tokens
        dataset = StructureDataset(json_path, truncate=None, max_length=500)
        dataset_indices = {d['name']:i for i,d in enumerate(dataset)}
        with open(split_path) as f:
            dataset_splits = json.load(f)
        if test_split_path is not None:
            with open(test_split_path) as f:
                test_splits = json.load(f)
            dataset_splits['test'] = test_splits['test']
        self.trainset, self.valset, self.testset = [
            Subset(dataset, [
                dataset_indices[chain_name] for chain_name in dataset_splits[key] 
                if chain_name in dataset_indices
            ]) for key in ['train', 'validation', 'test']
        ]
    
    def get_loader(self):
        train_loader, val_loader, test_loader = [StructureLoader(
                d, batch_size=self.batch_tokens
            ) for d in [self.trainset, self.valset, self.testset]]
        return train_loader, val_loader, test_loader