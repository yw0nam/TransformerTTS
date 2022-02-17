import torch
from torch import nn 


class Encoder(nn.Module):
    def __init__(self, embedding_size, num_hidden):
        """
        Transformer Encoder

        Args:
            embedding_size (_type_): _description_
            num_hidden (_type_): _description_
        """
        