import torch


class Stage2(torch.nn.Module):
    def __init__(self):
        super(Stage2, self).__init__()
        self.layer1 = torch.nn.Linear(in_features=367, out_features=512, bias=True)
        self.layer2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Linear(in_features=512, out_features=256, bias=True)
        self.layer4 = torch.nn.ReLU()
        self.layer5 = torch.nn.Linear(in_features=256, out_features=1, bias=True)
        self.layer6 = torch.nn.Sigmoid()
        self._initialize_weights()

    

    def forward(self, input0):
        out0 = input0.clone()
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        return out6

    def _initialize_weights(self):
        import numpy as np
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                mean = 0.0  # std_dev = np.sqrt(variance)
                n, m = module.in_features, module.out_features
                std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
                W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
                std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
                bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
                # approach 1
                module.weight.data = torch.tensor(W, requires_grad=True)
                module.bias.data = torch.tensor(bt, requires_grad=True)
            elif isinstance(module, torch.nn.EmbeddingBag):
                n, m = module.num_embeddings, module.embedding_dim
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
                # approach 1
                module.weight.data = torch.tensor(W, requires_grad=True)
