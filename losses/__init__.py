from __future__ import absolute_import

from losses.jaccard import *
from losses.dice import *
from losses.lovasz import *
from losses.focal import *
from losses.ohem_ce import *


if __name__ == "__main__":
    import numpy as np

    input = torch.from_numpy(np.array([[
        [
            [0.8, 0.0, 0.3],
            [0.6, 0.1, 0.2],
            [0.9, 0.1, 0.1],
        ],
        [
            [0.1, 0.8, 0.2],
            [0.3, 0.8, 0.1],
            [0.1, 0.8, 0.0],
        ],
        [
            [0.1, 0.2, 0.5],
            [0.1, 0.1, 0.7],
            [0.0, 0.1, 0.9],
        ],
    ]])).float()
    target = torch.from_numpy(np.array([[
        [
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
        ],
    ]])).long()

    input_bin = torch.from_numpy(np.array([[
        [
            [0.8, 0.2, 0.1],
            [0.2, 0.8, 0.2],
            [0.1, 0.2, 0.8],
        ]
    ]])).float()
    target_bin = torch.from_numpy(np.array([[
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    ]])).long()

    # Test jaccard loss
    # multiclass
    jaccard_loss = SoftJaccardLoss(num_classes=input.size()[1], from_logits=False, reduction='mean')
    print("Multiclass jaccard loss: %1.4f, should be: %1.4f" % (jaccard_loss(input, target.squeeze(dim=1)), 0.3933))
    # binary
    jaccard_loss = SoftJaccardLoss(num_classes=input_bin.size()[1], from_logits=False, reduction='none')
    print("Binary jaccard loss: %1.4f, should be: %1.4f" % (jaccard_loss(input_bin, target_bin.squeeze(dim=1)), 0.4))

