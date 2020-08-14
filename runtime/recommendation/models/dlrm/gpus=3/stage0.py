import torch


class Stage0(torch.nn.Module):
    def __init__(self):
        super(Stage0, self).__init__()
        self.layer106 = torch.nn.Linear(in_features=13, out_features=512, bias=True)
        self.layer107 = torch.nn.ReLU()
        self.layer108 = torch.nn.Linear(in_features=512, out_features=256, bias=True)
        self.layer109 = torch.nn.ReLU()
        self.layer110 = torch.nn.Linear(in_features=256, out_features=64, bias=True)
        self.layer111 = torch.nn.ReLU()
        self.layer112 = torch.nn.Linear(in_features=64, out_features=16, bias=True)
        self._initialize_weights()

    

    def forward(self, input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12, input13, input14, input15, input16, input17, input18, input19, input20, input21, input22, input23, input24, input25, input26, input27, input28, input29, input30, input31, input32, input33, input34, input35, input36, input37, input38, input39, input40, input41, input42, input43, input44, input45, input46, input47, input48, input49, input50, input51, input52):
        out0 = input0.clone()
        out1 = input1.clone()
        out2 = input2.clone()
        out3 = input3.clone()
        out4 = input4.clone()
        out5 = input5.clone()
        out6 = input6.clone()
        out7 = input7.clone()
        out8 = input8.clone()
        out9 = input9.clone()
        out10 = input10.clone()
        out11 = input11.clone()
        out12 = input12.clone()
        out13 = input13.clone()
        out14 = input14.clone()
        out15 = input15.clone()
        out16 = input16.clone()
        out17 = input17.clone()
        out18 = input18.clone()
        out19 = input19.clone()
        out20 = input20.clone()
        out21 = input21.clone()
        out22 = input22.clone()
        out23 = input23.clone()
        out24 = input24.clone()
        out25 = input25.clone()
        out26 = input26.clone()
        out27 = input27.clone()
        out28 = input28.clone()
        out29 = input29.clone()
        out30 = input30.clone()
        out31 = input31.clone()
        out32 = input32.clone()
        out33 = input33.clone()
        out34 = input34.clone()
        out35 = input35.clone()
        out36 = input36.clone()
        out37 = input37.clone()
        out38 = input38.clone()
        out39 = input39.clone()
        out40 = input40.clone()
        out41 = input41.clone()
        out42 = input42.clone()
        out43 = input43.clone()
        out44 = input44.clone()
        out45 = input45.clone()
        out46 = input46.clone()
        out47 = input47.clone()
        out48 = input48.clone()
        out49 = input49.clone()
        out50 = input50.clone()
        out51 = input51.clone()
        out52 = input52.clone()
        out106 = self.layer106(out0)
        out107 = self.layer107(out106)
        out108 = self.layer108(out107)
        out109 = self.layer109(out108)
        out110 = self.layer110(out109)
        out111 = self.layer111(out110)
        out112 = self.layer112(out111)
        return (out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16, out17, out18, out19, out20, out21, out22, out23, out24, out25, out26, out27, out28, out29, out30, out31, out32, out33, out34, out35, out36, out37, out38, out39, out40, out41, out42, out43, out44, out45, out46, out47, out112, out48, out49, out50, out51, out52)

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
                

                            