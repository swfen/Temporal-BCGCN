from typing import Callable, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset, zeros
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size


class TemporalConv(MessagePassing):
    r"""
        Args:
            in_channels (int or tuple): Size of each input sample, or : Node feature vector dimension
            out_channels (int): Size of each output sample. or : Node feature vector dimension
            map_size: Number of nodes
            aggr (string, optional): The aggregation scheme to use
                (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
                (default: :obj:`"add"`)
            root_weight (bool, optional): If set to :obj:`False`, the layer will
                not add the transformed root node features to the output.
                (default: :obj:`True`)
            bias (bool, optional): If set to :obj:`False`, the layer will not learn
                an additive bias. (default: :obj:`True`)
            **kwargs (optional): Additional arguments of
                :class:`torch_geometric.nn.conv.MessagePassing`.

        Shapes:
            - **input:**
              node features :math:`(|\mathcal{V}|, F_{in})` or
              :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
              if bipartite,
              edge indices :math:`(2, |\mathcal{E}|)`,
              edge features :math:`(|\mathcal{E}|, D)` *(optional)*
            - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
              :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
        """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 map_size: int,
                 batch_size: int,
                 aggr: str = 'add',
                 root_weight: bool = True, bias: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.map_size = map_size
        self.batch_size = batch_size
        #self.nn = nn
        self.root_weight = root_weight

        self.W = Parameter(torch.randn((batch_size*map_size*(map_size-1),in_channels), requires_grad= True ))  #48 the size of Brain map
        self.W = torch.nn.init.kaiming_normal_(self.W, a=0, mode='fan_in', nonlinearity='leaky_relu')

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.in_channels_l = in_channels[0]

        if root_weight:
            self.linW = Parameter(torch.randn((batch_size*map_size,out_channels), requires_grad= True ))
            self.linW = torch.nn.init.kaiming_normal_(self.linW, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        #reset(self.nn)
        #if self.root_weight:
        #    reset(self.linW)
        zeros(self.bias)
        #reset(self.W)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        device = torch.device('cuda:0')
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size).to(device)

        x_r = x[1]
        if x_r is not None and self.root_weight:
            out += torch.mul(self.linW, x_r)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:

        #weight = self.nn(edge_attr)
        #weight = weight.view(-1, self.in_channels_l, self.out_channels)
        #W = torch.randn((edge_attr.size()[0],edge_attr.size()[1]), requires_grad= True )
        device = torch.device('cuda:0')
        weight = torch.mul(self.W, edge_attr).to(device)
        sum = torch.mul(x_j, weight).to(device)

        return sum

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')


