import dgl
from dgl.nn.pytorch.conv import gatconv
from torch import nn


class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super(GAT, self).__init__()
        self.gat_layers = nn.ModuleList()

        # one-layer GAT
        self.gat_layers.append(
            gatconv.GATConv(
                in_size,
                out_size,
                heads[0],
                feat_drop=0.1,
                attn_drop=0.1,
                activation=None,
                bias=True,
            )
        )

        # # two-layer GAT
        # self.gat_layers.append(
        #     gatconv.GATConv(
        #         in_size,
        #         hid_size,
        #         heads[0],
        #         feat_drop=0,
        #         attn_drop=0.1,
        #         activation=F.elu,
        #         bias=True,
        #     )
        # )
        # self.gat_layers.append(
        #     gatconv.GATConv(
        #         (in_size[0], hid_size * heads[0]),
        #         out_size,
        #         heads[1],
        #         feat_drop=0.1,
        #         attn_drop=0.1,
        #         activation=None,
        #         bias=True,
        #     )
        # )

    def forward(self, g, inputs):
        h = inputs
        g = dgl.add_self_loop(g)
        attn = None

        for i, layer in enumerate(self.gat_layers):
            h, attn = layer(g, h, get_attention=True)
            if i == len(self.gat_layers) - 1:  # last layer
                h = h.mean(1)
            else:
                h = h.flatten(1)
        return h, attn
