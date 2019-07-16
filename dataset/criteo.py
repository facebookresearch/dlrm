from torch.utils.data import Dataset
from data_utils import *

class Criteo(Dataset):

    def __init__(self, dataset, randomize, split="train", df_path="", data=""):

        if dataset == "kaggle":
            df_exists = path.exists(str(data))
            if df_exists:
                print("Reading from pre-processed data=%s" % (str(data)))
                file = str(data)
            else:
                o_filename = "kaggleAdDisplayChallenge_processed"
                file = getKaggleCriteoAdData(df_path, o_filename)
        elif dataset == "terabyte":
            file = "./terbyte_data/tb_processed.npz"
            df_exists = path.exists(str(file))
            if df_exists:
                print("Reading Terabyte data-set processed data from %s" % file)
            else:
                raise (
                    ValueError(
                        "Terabyte data-set processed data file %s does not exist !!" % file
                    )
                )

        # load and preprocess data
        with np.load(file) as data:

            X_int = data["X_int"]
            X_cat = data["X_cat"]
            y = data["y"]
            self.counts = data["counts"]
        self.m_den = X_int.shape[1]
        self.n_emb = len(self.counts)
        print("Sparse features = %d, Dense features = %d" % (self.n_emb, self.m_den))

        indices = np.arange(len(y))
        indices = np.array_split(indices, 7)

        # randomize each day"s dataset
        if randomize == "day" or randomize == "total":
            for i in range(len(indices)):
                indices[i] = np.random.permutation(indices[i])

        train_indices = np.concatenate(indices[:-1])
        test_indices = indices[-1]
        val_indices, test_indices = np.array_split(test_indices, 2)

        print("Defined %s indices..." % (split))

        # randomize all data in training set
        if randomize == "total":
            train_indices = np.random.permutation(train_indices)
            print("Randomized indices...")

        # create training, validation, and test sets
        if split == 'train':
            self.samples_list = [(X_cat[i], X_int[i], y[i]) for i in train_indices]
        elif split == 'val':
            self.samples_list = [(X_cat[i], X_int[i], y[i]) for i in test_indices]
        elif split == 'test':
            self.samples_list = [(X_cat[i], X_int[i], y[i]) for i in val_indices]

        print("Split data according to indices...")



    def __getitem__(self, index):

        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]

        X_cat, X_int, y = self.samples_list[index]
        X_cat, X_int, y = self._default_preprocess(X_cat, X_int, y)

        return X_cat, X_int, y



    def _default_preprocess(self, X_cat, X_int, y):
        X_cat = torch.tensor(X_cat, dtype=torch.long).pin_memory()
        X_int = torch.log(
            torch.tensor(X_int, dtype=torch.float) + 1
        ).pin_memory()
        y = torch.tensor(y.astype(np.float32)).pin_memory()

        return X_int, X_cat, y

    def __len__(self):
        return len(self.samples_list)