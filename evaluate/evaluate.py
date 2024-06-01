import json
import os.path
import re

import torch

from model.ket2a.ket2a_processor import KET2AProcessor
from model.ket2a.model import KET2A


def load_model(model_dir):
    config_file = os.path.join(model_dir, "config.txt")
    model_file = os.path.join(model_dir, "model_weights-100.pth")
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    torch.manual_seed(config["seed"])
    processor = KET2AProcessor(
        config["datasets"],
        config["seq2seq_args"]["max_length"],
        config["batch_size"])
    # model
    ket2a = KET2A(processor, **config)
    ket2a.load_state_dict(torch.load(model_file))

    return ket2a


def predict(model: KET2A, input_desc):
    model.eval()
    with torch.no_grad():
        inputs, tgt = model.processor.in_index_tensors_from_sentence(input_desc)

        output_embedded, data, decoder_outputs, decoder_hidden, seq2seq_attentions = model(inputs, tgt, is_train=False)
        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoder_words = model.processor.out_index_tensors_2_words(decoded_ids)

    print("> %s" % input_desc)
    print("< %s" % " ".join(decoder_words))

    return decoder_words


def predict_task2actions(model: KET2A, task_desc):
    actions = predict(model, task_desc)[:-1]
    actions_params = []
    task_desc = " ".join(re.split(r"\s+", task_desc)[1:])

    for action in actions:
        action_desc = task_desc + " %s" % action
        action_params = predict(model, action_desc)[:-1]
        actions_params.append(action_params)

    return actions_params

def task_action():
    model = load_model("../log/log2023-11-21-10-21-04")
    predict(model, "装配区 sim卡托2 sim卡槽 公母匹配")
    predict_task2actions(model, "装配区 sim卡托2 sim卡槽 公母匹配")

if __name__ == "__main__":
    task_action()

