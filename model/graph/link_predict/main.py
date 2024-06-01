import itertools

import torch

from model.graph.gat.gat import GAT
from model.graph.gat.kg_processor import KGProcessor
from model.graph.gat.predictor import DotPredictor

"""
备注:
1、头节点是属性，尾节点是零件
"""

torch.device('cpu')

processor = KGProcessor("graph", "../../../datasets/datasets1")
g = processor.g
pos_g = processor.pos_g
neg_g = processor.neg_g

a = processor.entity_dict
b = pos_g.edges()
c = neg_g.edges()

num_nodes = g.num_nodes()
print(num_nodes)

entity_dim = 768
n_epochs = 500

h = torch.randn((num_nodes, entity_dim))
predictor = DotPredictor()

model = GAT(entity_dim, entity_dim, entity_dim, [2])

optimizer = torch.optim.Adam(
    itertools.chain(model.parameters()), lr=0.001
)

outputs = None
for e in range(n_epochs):
    # forward
    outputs, attentions = model(g, h)
    pos_score = predictor(pos_g, outputs)
    neg_score = predictor(neg_g, outputs)
    loss = predictor.compute_loss(pos_score, neg_score)
    # loss = predictor.compute_loss_only_pos(pos_score)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 5 == 0:
        print("In epoch {}, loss: {}".format(e, loss))

# ----------- check results ------------- #
with torch.no_grad():
    pos_score = predictor(pos_g, outputs)
    neg_score = predictor(neg_g, outputs)
    print("AUC", predictor.compute_auc(pos_score, neg_score))
