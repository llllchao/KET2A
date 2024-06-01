import re

import torch

from model.graph.gat.kg_processor import KGProcessor
from model.seq2seq.seq2seq_processor import Seq2SeqProcessor


class KET2AProcessor:
    def __init__(self, raw_dir, seq_max_len=10, batch_size=16):
        self.raw_dir = raw_dir
        self.seq_max_len = seq_max_len
        self.batch_size = batch_size

        self.graph_dataset = None
        self.seq2seq_dataset = None

        self.all_entities = {}  # store both graph entities and seq2seq entities, {name: id}
        self.num_entities = 0

        self.process()

    def get_entities(self):
        sorted_entities = sorted(self.all_entities.keys(), key=lambda k: self.all_entities[k])
        return list(sorted_entities)

    def process(self):
        self.graph_dataset = KGProcessor("graph", self.raw_dir)
        self.seq2seq_dataset = Seq2SeqProcessor("seq2seq", self.raw_dir, max_len=self.seq_max_len,
                                                batch_size=self.batch_size)

        for entity in self.graph_dataset.entity_dict:
            i = self.all_entities.setdefault(entity, self.num_entities)
            if i == self.num_entities:
                self.num_entities += 1

        for entity in self.seq2seq_dataset.in_words.word2index:
            i = self.all_entities.setdefault(entity, self.num_entities)
            if i == self.num_entities:
                self.num_entities += 1

        for entity in self.seq2seq_dataset.out_words.word2index:
            i = self.all_entities.setdefault(entity, self.num_entities)
            if i == self.num_entities:
                self.num_entities += 1

    def global_2_in_seq_entity(self):
        global_idx = []
        seq_idx = []
        for entity, i in self.seq2seq_dataset.in_words.word2index.items():
            index = self.all_entities.get(entity)
            global_idx.append(index)
            seq_idx.append(i)
        return global_idx, seq_idx

    def global_2_out_seq_entity(self):
        global_idx = []
        seq_idx = []
        for entity, i in self.seq2seq_dataset.out_words.word2index.items():
            index = self.all_entities.get(entity)
            global_idx.append(index)
            seq_idx.append(i)
        return global_idx, seq_idx

    def global_2_graph_entity(self):
        global_idx = []
        graph_idx = []
        for entity, i in self.graph_dataset.entity_dict.items():
            index = self.all_entities.get(entity)
            global_idx.append(index)
            graph_idx.append(i)
        return global_idx, graph_idx

    def in_seq_entity_2_graph_entity(self):
        seq_idx = []
        graph_idx = []
        for entity, i in self.seq2seq_dataset.in_words.word2index.items():
            index = self.graph_dataset.entity_dict.get(entity)
            if index is not None:
                seq_idx.append(i)
                graph_idx.append(index)
        return seq_idx, graph_idx

    def out_seq_entity_2_graph_entity(self):
        seq_idx = []
        graph_idx = []
        for entity, i in self.seq2seq_dataset.out_words.word2index.items():
            index = self.graph_dataset.entity_dict.get(entity)
            if index is not None:
                seq_idx.append(i)
                graph_idx.append(index)
        return seq_idx, graph_idx

    def in_index_tensors_from_sentence(self, sentence, device='cpu'):
        in_indexes = [self.seq2seq_dataset.in_words.word2index[e] for e in re.split(r"\s+", sentence)]
        eos_i = self.seq2seq_dataset.in_words.word2index["EOS"]
        in_indexes.append(eos_i)

        tgt_indexes = [self.seq2seq_dataset.out_words.word2index["SOS"]]
        return torch.tensor(in_indexes, dtype=torch.long, device=device).view(1, -1), \
               torch.tensor(tgt_indexes, dtype=torch.long, device=device).view(1, -1)

    def out_index_tensors_2_words(self, out_index_tensors):
        decoded_words = []
        eos_out_idx = self.seq2seq_dataset.out_words.word2index["EOS"]
        for idx in out_index_tensors:
            if idx.item() == eos_out_idx:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(self.seq2seq_dataset.out_words.index2word[idx.item()])
        return decoded_words


if __name__ == "__main__":
    p = KET2AProcessor("../../datasets/datasets4")
    print(p.get_entities())
