import torch
import torch.nn as nn

from networks.imaging_decoders import Relu


class TabularDecoder(nn.Module):
    def __init__(self, input_dim, dim, out_dim, depth, **kwargs) -> None:
        super().__init__()

        self.decoder = nn.ModuleList([Relu(input_dim, dim // 2, dropout=0.1)])
        self.decoder.extend([
            Relu(dim // 2 ** i, dim // 2 ** (i + 1), dropout=0.1) for i in range(1, depth)])
        # self.out_dim = out_dim
        self.fc = nn.Linear(dim // 2 ** depth, out_dim, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.flatten(start_dim=1)
        for layer in self.decoder:
            x = layer(x)
        x = self.fc(x)
        return x


class NumericalReconHead(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs) -> None:
        super().__init__()
        self.head = Relu(in_size=input_dim, out_size=output_dim, dropout=0)
        
    def forward(self, x):
        x = self.head(x)
        return x
    
    
class SingleCategoricalReconHead(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs) -> None:
        super().__init__()
        self.head = torch.nn.Linear(in_features=input_dim, out_features=output_dim)
        
    def forward(self, x):
        x = self.head(x)
        x = x.view(x.shape[0], -1, 2)
        return x
    
    
class MultipleCategoricalReconHead(nn.Module):
    def __init__(self, input_dim, selected_features, **kwargs) -> None:
        super().__init__()
        self.head = torch.nn.ModuleList(
            [torch.nn.Linear(in_features=input_dim, out_features=class_num) 
             for class_num, _ in selected_features.values()])
        
    def forward(self, x):
        out_list = []
        for h in self.head:
            out = h(x)
            out_list.append(out)
        return out_list
