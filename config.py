class Config:
    def __init__(self):
        # Main arguments
        self.method = "FedSTL"
        self.model = "RNN"
        self.epoch = 30
        self.mode = "train_cp"
        self.dataset = "fhwa"
        self.client = 100
        self.cluster = 10
        self.frac = 0.1
        self.property_type = 'constraint'

        # Fine-tune arguments
        self.fine_tune_iter = 5
        self.cluster_fine_tune_iter = 2
        self.local_updates = 10
        self.client_iter = 10
        self.head_iter = 8

        # Training arguments
        self.batch_size = 64
        self.pretrain_iter = 30
        self.max_lr = 0.001
        self.grad_clip = 0.1
        self.weight_decay = 1e-4
