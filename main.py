import os

from torch import nn, optim

from model.graph.gat.predictor import DotPredictor
from model.loss.mtl import MTLoss
from utils import util
from utils.functions import *

import torch

from model.ket2a.ket2a_processor import KET2AProcessor
from model.ket2a.model import KET2A
from torch.summary import summary

config = {
    "seed": 0,
    "log": "log",
    "datasets": r"./datasets/datasets7",
    "entity_dim": 768,  # * "* means necessary"
    "batch_size": 64,  # *
    "pre_train": True,  # *
    "lang": "cn",  # * "lang of datasets"
    "gcn_type": "none",  # *
    "epochs": 120,  # *
    "seq2seq_type": "gru2gru",  # *
    "gcn_args": {  # *
        "device": "cpu",
        "hidden_dim": 128,
        "learning_rate": 0.001,
        "heads": [2, ]
    },
    "seq2seq_args": {  # *
        "device": "cpu",
        "max_length": 10,
        "encoder_hidden_dim": 128,
        "decoder_hidden_dim": 128,
        "learning_rate": 0.001
    },
}

torch.manual_seed(config["seed"])
processor = KET2AProcessor(
    config["datasets"],
    config["seq2seq_args"]["max_length"],
    config["batch_size"])
# model
ket2a = KET2A(processor, **config)
# datasets
train_dataloader = processor.seq2seq_dataset.train
test_dataloader = processor.seq2seq_dataset.test


def train(model: KET2A, train_data, test_data, n_epochs=200, print_every=3, save_every=50):
    log_dir = util.create_log_dir(config.get("log_dir"))

    # criterion
    seq_criterion = nn.NLLLoss()
    gcn_criterion = DotPredictor()
    combined_criterion = MTLoss()

    # optimizer
    if model.gcn is not None:
        gcn_optimizer = optim.Adam([
            {"params": model.gcn.parameters(), "lr": config["gcn_args"]["learning_rate"]},
        ])
    seq_optimizer = optim.Adam([
        {"params": model.seq2seq.parameters(), "lr": config["seq2seq_args"]["learning_rate"]},
    ])
    loss_optimizer = optim.Adam([
        {"params": combined_criterion.parameters(), "lr": 0.001},
    ])
    emb_optimizer = optim.Adam([
        {"params": model.embeddings.parameters(), "lr": 0.001},
    ])

    epoch_seq_loss = []
    epoch_gcn_loss = []
    epoch_train_acc = []
    epoch_test_acc = []
    epoch_perplexity = []
    start = time.time()

    def train_epoch():
        model.train()
        total_seq_loss = 0
        total_gcn_loss = 0
        total_train_correct_n = 0
        total_n = 0

        for data in train_data:
            inputs, outputs, targets = data
            output_embedded, data, decoder_outputs, decoder_hidden, seq2seq_attentions = model(
                inputs, targets, is_train=True
            )

            # calculate loss
            seq_loss = seq_criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                outputs.view(-1)
            )

            pos_scores = gcn_criterion(processor.graph_dataset.pos_g, output_embedded)
            neg_scores = gcn_criterion(processor.graph_dataset.neg_g, output_embedded)
            gcn_loss = gcn_criterion.compute_loss(pos_scores, neg_scores)

            loss = combined_criterion(gcn_loss, seq_loss)

            _, topi = decoder_outputs.topk(1)
            decoder_outputs = topi.squeeze()
            correct_num = torch.eq(decoder_outputs, outputs).all(dim=-1).sum()

            # backward
            loss_optimizer.zero_grad()
            seq_optimizer.zero_grad()
            emb_optimizer.zero_grad()
            if model.gcn is None:
                seq_loss.backward()
            else:
                gcn_optimizer.zero_grad()
                loss.backward()
                gcn_optimizer.step()

            seq_optimizer.step()
            loss_optimizer.step()
            emb_optimizer.step()

            total_seq_loss += seq_loss.item()
            total_gcn_loss += gcn_loss.item()
            total_train_correct_n += correct_num
            total_n += decoder_outputs.shape[0]

        return total_seq_loss / len(train_dataloader), total_gcn_loss / len(
            train_dataloader), total_train_correct_n / total_n

    for epoch in range(1, n_epochs + 1):
        seq_loss, gcn_loss, train_acc = train_epoch()
        test_acc, perplexity = test_epoch(test_data, model)

        epoch_test_acc.append(test_acc)
        epoch_train_acc.append(train_acc)
        epoch_seq_loss.append(seq_loss)
        epoch_gcn_loss.append(gcn_loss)
        epoch_perplexity.append(perplexity)

        if epoch % print_every == 0:
            print('%s (%d %d%%) seq_loss: %.5f gcn_loss: %.5f train_acc: %.5f test_acc: %.5f perplexity: %.5f' %
                  (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, seq_loss, gcn_loss,
                   train_acc * 100, test_acc * 100, perplexity))
        if epoch % save_every == 0:
            save_model(os.path.join(log_dir, 'model_weights-%d.pth' % epoch), model)

    save_log(log_dir, epoch_seq_loss, epoch_train_acc, epoch_test_acc, epoch_perplexity, config)


if __name__ == "__main__":
    summary(ket2a)
    train(ket2a, train_dataloader, test_dataloader, config["epochs"])
    # model.load_state_dict(torch.load('.\\log\\log2023-10-31-16-26-21\\model_weights-120.pth'))
    # evaluate_test(model, dataset)
    # predict_task2actions(model, dataset, "装配区 后盖3 中框 公母匹配")
    # predict_and_show_attn(model, "物料区 手机电池_1 手机电池 插槽固定", dataset)
