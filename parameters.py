class benchmark:
    def __init__(self):
        self.workdir = ""
        self.datasets = {
            "20NG": [20, 1006],
            "Mediamill": [101, 120],
            "Scene": [6, 294],
            "Yeast": [14, 103],
            "synthetic_monolab": [4, 4],
            "synthetic_bilab": [4, 4],
            "synthetic_rand": [4, 4],
        }
        self.models = [
            "NN",
            "NN_TL",
            "NN_TLH",
            "NN_TLH_sampling",
            "NN_TLH_fifo",
            "NN_TLH_memories",
            "NN_TLH_mini_memories",
            "BR_HT",
            "LC_HT",
            "CC_HT",
            "BR_random_forest",
            "iSOUPtree",
        ]
