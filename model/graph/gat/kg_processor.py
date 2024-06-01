import os
import re

import dgl
import numpy as np
import scipy.sparse as sp
import torch
from dgl.data import DGLDataset


class KGProcessor(DGLDataset):
    """
        refer to KnowledgeGraphDataset
        refer to https://docs.dgl.ai/guide_cn/data.html
        1、to build homogeneous graph
        2、add pos_g, neg_g
    """

    def __init__(self, name, raw_dir):
        self.g = None
        self.pos_g = None
        self.neg_g = None
        self.entity_dict = None  # {entity: id}
        self.relation_dict = None  # {relation: id}
        self.triplets = None  # [[int, int, int], [...]]
        self.origin_triplets = None  # [[str, str, str], [...]]

        super(KGProcessor, self).__init__(name, raw_dir=raw_dir, verbose=True)

    def process(self):
        triplet_path = os.path.join(self.raw_path, 'triplets.txt')  # self.raw_path = raw_dir + {name}
        self.origin_triplets, self.triplets, self.relation_dict, self.entity_dict = _read_and_parse_triplets(
            triplet_path)

        print("# loading homogeneous graph dataset...")
        print("# entities: {}".format(len(self.entity_dict)))
        print("# relations: {}".format(len(self.relation_dict)))
        print("# triplet: {}".format(len(self.triplets)))

        # build homogeneous graph
        self.g, data = _build_knowledge_graph(len(self.entity_dict), self.triplets)
        rel = data[0]
        self.g.edata['etype'] = rel

        # build neg/pos graph
        self.pos_g, self.neg_g = _build_pos_and_neg_graph(self.g)

    def __getitem__(self, idx):
        assert 0 <= idx <= 2, "This dataset has only 3 graph"

        g = None
        if idx == 0:
            g = self.g
        elif idx == 1:
            g = self.pos_g
        elif idx == 2:
            g = self.neg_g

        if self._transform is None:
            return g
        else:
            return self._transform(g)

    def __len__(self):
        return 3


def _read_and_parse_triplets(filename):
    triplets = []
    entities = {}
    rels = {}
    entity_count = 0
    rel_count = 0
    origin_triplets = []

    for triplet in _read_triplets(filename):
        origin_triplets.append(triplet)
        if triplet[0] not in entities.keys():
            entities.setdefault(triplet[0], entity_count)
            entity_count += 1
        if triplet[1] not in rels.keys():
            rels.setdefault(triplet[1], rel_count)
            rel_count += 1
        if triplet[2] not in entities.keys():
            entities.setdefault(triplet[2], entity_count)
            entity_count += 1

        s = entities[triplet[0]]
        r = rels[triplet[1]]
        o = entities[triplet[2]]
        triplets.append([s, r, o])

    return origin_triplets, triplets, rels, entities


def _read_triplets(filename):
    with open(filename, 'r+', encoding='utf-8') as f:
        for line in f:
            processed_line = line.strip()
            processed_line = re.split(r"\s+", processed_line)
            yield processed_line


def _build_knowledge_graph(num_nodes, triplets):
    src = []
    rel = []
    dst = []

    for triplet in triplets:
        s, r, d = triplet
        # src.append(s)
        # rel.append(r)
        # dst.append(d)
        src.append(d)  # d是属性
        rel.append(r)
        dst.append(s)  # s是零件

    src = torch.tensor(src, dtype=torch.int64)
    rel = torch.tensor(rel, dtype=torch.int64)
    dst = torch.tensor(dst, dtype=torch.int64)

    g = dgl.graph((src, dst), num_nodes=num_nodes)

    return g, (rel,)


def _build_pos_and_neg_graph(g: dgl.DGLGraph):
    src, dst = g.edges()
    num_nodes = g.num_nodes()

    # find all negative edges
    max_len = torch.cat([src, dst]).max().item() + 1
    adj = sp.coo_matrix((np.ones(len(src)), (src.numpy(), dst.numpy())), shape=(max_len, max_len))
    adj_neg = 1 - adj.todense()
    # exclude reverse edges from negative samples
    adj_neg[dst, src] = 0  # 反向边不要
    adj_neg[torch.arange(0, max_len), torch.arange(0, max_len)] = 0  # 自连边不要
    adj_neg[src.repeat(len(src)), src.repeat_interleave(len(src))] = 0  # 同类连边不要
    adj_neg[dst.repeat(len(dst)), dst.repeat_interleave(len(dst))] = 0  # 同类连边不要

    neg_src, neg_dst = np.where(adj_neg != 0)

    # TODO: 一个正样本对几个负样本
    pos_g = dgl.graph((src, dst), num_nodes=num_nodes)
    neg_g = dgl.graph((neg_src, neg_dst), num_nodes=num_nodes)

    return pos_g, neg_g


if __name__ == "__main__":
    p = KGProcessor("graph", "../../datasets/datasets1")
    print(p.g.ndata["h"])
