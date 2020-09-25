import sys
import torch
from torch.utils.data import Dataset


def collate_wrapper_random(list_of_tuples):
    # where each tuple is (X, lS_o, lS_i, T)
    (X, lS_o, lS_i, T) = list_of_tuples[0]
    return (X,
            torch.stack(lS_o),
            lS_i,
            T)


class RandomDataset(Dataset):

    def __init__(
        self,
        mini_batch_size,
        nbatches=1,
#        write_data_folder="./synthetic_data/syn_data_bs65536/",
        write_data_folder="./synthetic_data/syn_data_bs65536_3M_emb_size/",
    ):
        self.write_data_folder = write_data_folder
        self.num_batches = nbatches
        self.mini_batch_size = mini_batch_size

        self.X = torch.load(f"{self.write_data_folder}/X_0.pt")
        self.lS_o = torch.load(f"{self.write_data_folder}/lS_o_0.pt")
        self.lS_i = torch.load(f"{self.write_data_folder}/lS_i_0.pt")
        self.T = torch.load(f"{self.write_data_folder}/T_0.pt")
        # print('data loader initiated ...')

    def __getitem__(self, index):
        sInd = index * self.mini_batch_size
        eInd = sInd + self.mini_batch_size
        if sInd >= len(self.X):
            sys.exit(f' mini_batch_size({self.mini_batch_size}) * '
                f'num_batches({self.num_batches}) has to be less'
                f' than size of data({len(self.X)})'
            )
        X = self.X[sInd:eInd]
        lS_o = [i[:][sInd:eInd] - i[:][sInd] for i in self.lS_o]

        if eInd < len(self.lS_o[0]):
            lS_i = [val[self.lS_o[ind][sInd]:self.lS_o[ind][eInd]] for ind, val in enumerate(self.lS_i)]
        elif sInd < len(self.lS_o[0]):
            lS_i = [val[self.lS_o[ind][sInd]:] for ind, val in enumerate(self.lS_i)]

        T = self.T[sInd:eInd]
        return (X, lS_o, lS_i, T)

    def __len__(self):
        return self.num_batches


def make_random_data_and_loader(args, ln_emb, m_den):

    train_data = RandomDataset(
        args.mini_batch_size,
        nbatches=args.num_batches
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_random,
        pin_memory=False,
        drop_last=False,
    )
    return train_data, train_loader

