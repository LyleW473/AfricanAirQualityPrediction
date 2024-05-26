import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.layer1 = nn.Sequential(
                                    nn.Linear(74, 48),
                                    nn.BatchNorm1d(48),
                                    nn.ReLU()
                                    )
        self.downsample1 = nn.Sequential(
                                        nn.Linear(74, 48),
                                        nn.BatchNorm1d(48)                            
                                        )
        self.layer2 = nn.Sequential(
                                    nn.Linear(48, 24),
                                    nn.BatchNorm1d(24),
                                    nn.ReLU()
                                    )
        self.downsample2 = nn.Sequential(
                                        nn.Linear(48, 24),
                                        nn.BatchNorm1d(24)                            
                                        )
        self.layer3 = nn.Sequential(
                                    nn.Linear(24, 12),
                                    nn.BatchNorm1d(12),
                                    nn.ReLU()
                                    )
        self.downsample3 = nn.Sequential(
                                        nn.Linear(24, 12),
                                        nn.BatchNorm1d(12)                            
                                        )
                                    
        self.layer4 = nn.Linear(12, 1)
        self._init_weights()

    def forward(self, input_data):
        # (Add the skip connections here)
        x = self.layer1(input_data) + self.downsample1(input_data)
        x = self.layer2(x) + self.downsample2(x)
        x = self.layer3(x) + self.downsample3(x)
        return self.layer4(x)
    
    def _init_weights(self):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)