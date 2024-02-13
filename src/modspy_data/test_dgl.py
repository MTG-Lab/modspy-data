from dgl import heterograph
from dgl.sampling import random_walk
import torch

g2 = heterograph({
    ('user', 'follow', 'user'): ([0, 1, 1, 2, 3], [1, 2, 3, 0, 0]),
    ('user', 'view', 'item'): ([0, 0, 1, 2, 3, 3], [0, 1, 1, 2, 2, 1]),
    ('item', 'viewed-by', 'user'): ([0, 1, 1, 2, 2, 1], [0, 0, 1, 2, 3, 3])}, device=torch.device('cpu'))

print(random_walk(g2, [0, 1, 2, 0], metapath=['follow', 'view', 'viewed-by'] * 2))

