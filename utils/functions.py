import json
import math
import os
import time

import numpy as np
import torch
from torch import nn

from model.ket2a.model import KET2A


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# Calculating remaining time
def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def save_model(save_file, model):
    torch.save(model.state_dict(), save_file)


def test_epoch(dataloader, model: KET2A):
    model.eval()
    with torch.no_grad():
        total_correct_n = 0
        total_n = 0
        seq_criterion = nn.NLLLoss(reduction="sum")
        total_words = 0
        total_loss = 0

        for data in dataloader:
            inputs, outputs, targets = data

            output_embedded, data, decoder_outputs, decoder_hidden, seq2seq_attentions = model(
                inputs, targets, is_train=False
            )

            # compute perplexity
            # calculate loss
            seq_loss = seq_criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                outputs.view(-1)
            )
            total_loss += seq_loss
            total_words += outputs.numel()

            # calculate accuracy on the test dataset
            _, topi = decoder_outputs.topk(1)
            decoder_outputs = topi.squeeze()
            correct_num = torch.eq(decoder_outputs, outputs).all(dim=-1).sum()
            total_n += decoder_outputs.shape[0]
            total_correct_n += correct_num

    return total_correct_n / total_n, np.exp(total_loss / total_words)


def save_log(dir_path, loss: list, train_acc: list, test_acc: list, epochs_perplexity: list, config: dict = None):
    loss_file = os.path.join(dir_path, "train_loss.txt")
    train_acc_file = os.path.join(dir_path, "train_acc.txt")
    perplexity_file = os.path.join(dir_path, "perplexity.txt")
    test_acc_file = os.path.join(dir_path, "test_acc.txt")
    config_file = os.path.join(dir_path, "config.txt")
    with open(loss_file, 'a', encoding='utf-8') as f:
        for i, l in enumerate(loss):
            f.write("%d\t%f\n" % (i, l))

    with open(train_acc_file, 'a', encoding='utf-8') as f:
        for i, acc in enumerate(train_acc):
            f.write("%d\t%f\n" % (i, acc))

    with open(test_acc_file, 'a', encoding='utf-8') as f:
        for i, acc in enumerate(test_acc):
            f.write("%d\t%f\n" % (i, acc))
    with open(config_file, 'a', encoding='utf-8') as f:
        if config is not None:
            f.write(json.dumps(config, indent=2))

    with open(perplexity_file, 'a', encoding='utf-8') as f:
        for i, perplexity in enumerate(epochs_perplexity):
            f.write("%d\t%f\n" % (i, perplexity))
