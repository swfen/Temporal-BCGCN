from typing import Callable, Optional, Union

import torch
from torch.nn import Parameter
from torch_scatter import scatter_add, scatter_max

from torch_geometric.utils import softmax

from net.num_nodes import maybe_num_nodes
from net.inits import uniform


def topk(x, ratio, batch, min_score=None, tol=1e-7):
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter_max(x, batch)[0].index_select(0, batch) - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero(as_tuple=False).view(-1)
    else:
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes, ),
                             torch.finfo(x.dtype).min)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)
        perm_lab = perm.view(-1)

        if isinstance(ratio, int):
            k = num_nodes.new_full((num_nodes.size(0), ), ratio)
            k = torch.min(k, num_nodes)
        else:
            k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)

        mask = [
            torch.arange(k[i], dtype=torch.long, device=x.device) +
            i * max_num_nodes for i in range(batch_size)
        ]
        mask = torch.cat(mask, dim=0)

        perm = perm[mask]

    return perm,perm_lab,k


def filter_adj(edge_index, edge_attr, perm, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr

def Category_pool(perm,perm_lab,Tendency:[int, float] = None,k = None):
    if Tendency == None :
        perm = perm
    else:
        batch_size = k.size(0)
        max_num_nodes = int(perm_lab.shape[0]/batch_size)
        if isinstance(Tendency, int):
            Ten = k.new_full((k.size(0), ), Tendency)
            Ten_L = torch.min(Ten, k)
        else:
            Ten_L = (Tendency * k.to(torch.float)).ceil().to(torch.long)

        Ten_R = k - Ten_L
        mask = torch.empty(0,device=k.device,dtype = torch.long)
        for i in range(batch_size):
            permL_lab = torch.nonzero(perm_lab[i*90:(i*90+90)] % 2 == 0)
            permR_lab = torch.nonzero(perm_lab[i*90:(i*90+90)] % 2 == 1)
            L_Lab = [
                torch.arange(Ten_L[i], dtype=torch.long, device=k.device)
            ]
            R_Lab = [
                torch.arange(Ten_R[i], dtype=torch.long, device=k.device)
            ]
            permL = torch.squeeze(permL_lab[L_Lab], 1)
            permR = torch.squeeze(permR_lab[R_Lab], 1)
            permL = permL+(i*90)
            permR = permR+(i*90)
            Lab = torch.cat((permL,permR), dim=0)
            mask = torch.cat((mask,Lab), dim=0)
        perm = perm_lab[mask]
    return perm

class Category_TopK_Pooling(torch.nn.Module):
    r"""
        Args:
            in_channels (int): Size of each input sample.
            ratio (float or int): Graph pooling ratio, which is used to compute
                :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`, or the value
                of :math:`k` itself, depending on whether the type of :obj:`ratio`
                is :obj:`float` or :obj:`int`.
                This value is ignored if :obj:`min_score` is not :obj:`None`.
                (default: :obj:`0.5`)
            Tendency (float or int)：Ratio of the number of nodes to be retained after pooling of category A and B nodes
                or the number of nodes to be retained after pooling of category A nodes
            min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
                which is used to compute indices of pooled nodes
                :math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
                When this value is not :obj:`None`, the :obj:`ratio` argument is
                ignored. (default: :obj:`None`)
            multiplier (float, optional): Coefficient by which features gets
                multiplied after pooling. This can be useful for large graphs and
                when :obj:`min_score` is used. (default: :obj:`1`)
            nonlinearity (torch.nn.functional, optional): The nonlinearity to use.
                (default: :obj:`torch.tanh`)
        """
    def __init__(self, in_channels: int, ratio: Union[int, float] = 0.5,Tendency:[int, float] = None,
                 min_score: Optional[float] = None, multiplier: float = 1.,
                 nonlinearity: Callable = torch.tanh):

        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.Tendency =Tendency
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.weight = Parameter(torch.Tensor(1, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.weight)

    def forward(self, x, edge_index, edge_attr=None, batch=None, attn=None):
        """"""

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = (attn * self.weight).sum(dim=-1)

        if self.min_score is None:
            score = self.nonlinearity(score / self.weight.norm(p=2, dim=-1))
        else:
            score = softmax(score, batch)

        perm,perm_lab,k = topk(score, self.ratio, batch, self.min_score)

        perm = Category_pool(perm,perm_lab,self.Tendency,k)

        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))

        return x, edge_index, edge_attr,batch, perm, perm_lab

    def __repr__(self) -> str:
        if self.min_score is None:
            ratio = f'ratio={self.ratio}'
        else:
            ratio = f'min_score={self.min_score}'

        return (f'{self.__class__.__name__}({self.in_channels}, {ratio}, '
                f'multiplier={self.multiplier})')
