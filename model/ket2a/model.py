import torch
from torch import nn

from model.graph.gat.gat import GAT
from model.ket2a.ket2a_processor import KET2AProcessor
from model.pre_trained import cn_embedding, en_embedding
from model.seq2seq.gru2gru.gru2gru import GRU2GRU


class KET2A(nn.Module):
    def __init__(self, processor: KET2AProcessor, entity_dim, pre_train=True, gcn_type="gat", seq2seq_type="gru2gru",
                 **kwargs):
        super(KET2A, self).__init__()

        self.processor = processor
        self.gcn_type = gcn_type
        self.seq2seq_type = seq2seq_type
        self.kwargs = kwargs

        self.entity_dim = entity_dim
        self.seq2seq_output_dim = len(
            processor.global_2_out_seq_entity()[0])  # output dimension of seq2seq, equal to the number of output words

        self.gcn = None
        self.seq2seq = None

        if pre_train:
            if kwargs.get("lang") == "en":
                self.embeddings = nn.Embedding.from_pretrained(en_embedding.get_embeddings(processor.get_entities()))
            elif kwargs.get("lang") == "cn":
                self.embeddings = nn.Embedding.from_pretrained(cn_embedding.get_embeddings(processor.get_entities()))
            else:
                self.embeddings = nn.Embedding.from_pretrained(cn_embedding.get_embeddings(processor.get_entities()))
        else:
            self.embeddings = nn.Embedding(processor.num_entities, entity_dim)

        self.init_model()

    def init_model(self):
        gcn_args = self.kwargs.get("gcn_args")
        device = gcn_args.get("device") if gcn_args.get("device") is not None else "cpu"
        hidden_dim = gcn_args.get("hidden_dim")
        if hidden_dim is None:
            raise ValueError("miss the arg: 'hidden_dim'")

        if self.gcn_type == "gat":
            heads = gcn_args.get("heads")
            if heads is None:
                raise ValueError("'heads' should not be 'None' when gcn_type is 'gat'")
            self.gcn = GAT(self.entity_dim, hidden_dim, self.entity_dim, heads).to(device)
        elif self.gcn_type == "none":
            pass

        seq2seq_args = self.kwargs.get("seq2seq_args")
        device = seq2seq_args.get("device") if gcn_args.get("device") is not None else "cpu"
        encoder_hidden_dim = seq2seq_args.get("encoder_hidden_dim")
        if encoder_hidden_dim is None:
            raise ValueError("miss the arg: 'encoder_hidden_dim'")
        decoder_hidden_dim = seq2seq_args.get("decoder_hidden_dim")
        if decoder_hidden_dim is None:
            raise ValueError("miss the arg: 'decoder_hidden_dim'")
        max_length = seq2seq_args.get("max_length") if seq2seq_args.get("max_length") is not None else 10
        encoder_bidirectional = seq2seq_args.get("encoder_bidirectional") if seq2seq_args.get(
            "encoder_bidirectional") is not None else False

        if self.seq2seq_type == "gru2gru":
            self.seq2seq = GRU2GRU(self.entity_dim, encoder_hidden_dim, decoder_hidden_dim,
                                   self.seq2seq_output_dim, encoder_bidirectional=encoder_bidirectional,
                                   max_length=max_length).to(device)

    def get_init_graph_embedded(self):
        return self.embeddings(torch.tensor(self.processor.global_2_graph_entity()[0]))

    def get_init_in_seq_embedded(self):
        return self.embeddings(torch.tensor(self.processor.global_2_in_seq_entity()[0]))

    def get_init_out_seq_embedded(self):
        return self.embeddings(torch.tensor(self.processor.global_2_out_seq_entity()[0]))

    def forward(self, index_inputs, index_targets, is_train):
        # In GAT, data refers to attentions
        if self.gcn is not None:
            output_embedded, data = self.gcn(self.processor.graph_dataset.g, self.get_init_graph_embedded())
        else:
            output_embedded, data = self.get_init_graph_embedded(), None

        # replaces the embedding of the sequence input with the embedding of GCN output
        seq_idx, graph_idx = self.processor.in_seq_entity_2_graph_entity()
        in_seq_embedded = self.get_init_in_seq_embedded()
        in_seq_embedded[seq_idx, :] = output_embedded[graph_idx].clone()
        inputs = in_seq_embedded[index_inputs]

        # replaces the embedding of the sequence target with the embedding of GCN output
        seq_idx, graph_idx = self.processor.out_seq_entity_2_graph_entity()
        tgt_seq_embedded = self.get_init_out_seq_embedded()
        tgt_seq_embedded[seq_idx, :] = output_embedded[graph_idx].clone()
        targets = tgt_seq_embedded[index_targets]

        decoder_outputs, decoder_hidden, seq2seq_attentions = self.seq2seq(inputs, targets, tgt_seq_embedded, is_train=is_train)

        return output_embedded, data, decoder_outputs, decoder_hidden, seq2seq_attentions

