# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn

class CellEmbeddings(nn.Module):
    def __init__(self, max_2d_position_embeddings=501, hidden_size=1024, ccat=False):
        super(CellEmbeddings, self).__init__()
        self.ccat = ccat
        self.max_2d_position_embeddings = max_2d_position_embeddings
        if ccat:
            self.x_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size // 4)
            self.y_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size // 4)
        else:
            self.x_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size)
            self.y_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size)

    def forward(self, bbox):
        bbox = torch.clip(bbox, 0.0, 1.0)
        bbox = (bbox * (self.max_2d_position_embeddings-1)).long()
        left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        if self.ccat:
            embeddings = torch.cat(
                [
                    left_position_embeddings,
                    upper_position_embeddings,
                    right_position_embeddings,
                    lower_position_embeddings
                ],
                dim=-1)
        else:
            embeddings = (
                left_position_embeddings
                + upper_position_embeddings
                + right_position_embeddings
                + lower_position_embeddings
            )

        return embeddings
    