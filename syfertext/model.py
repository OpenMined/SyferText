class Model(nn.Module):
    """
    Abstract base class that defines a loss() function
    """
    def __init__(self, hparams=None):
        super(Model, self).__init__()
        
        if hparams is None:
            raise ValueError('Must provide hparams')
        
        self.hparams = hparams

        # Track total iterations
        self.iterations = nn.Parameter(torch.LongTensor([0]), requires_grad=False)

    def loss(self, batch, compute_predictions=False):
        """
        Called by train.Trainer to compute negative log likelihood
        Parameters:
            batch: A minibatch, instance of torchtext.data.Batch
            compute_predictions: If true compute and provide predictions, else None
        Returns:
            Tuple of loss and predictions
        """
        raise NotImplementedError("Must implement loss()")

    @classmethod
    def create(cls, task_name, hparams, overwrite=False, **kwargs):
        """
        Create a new instance of this class. Prepares the model directory
        and saves hyperparams. Derived classes should override this function
        to save other dependencies (e.g. vocabs)
        """
        logger.info(hparams)
        model_dir = gen_model_dir(task_name, cls)
        model = cls(hparams, **kwargs)
        # Xavier initialization
        model.apply(xavier_uniform_init)

        if torch.cuda.is_available():
            model = model.cuda()

        prepare_model_dir(model_dir, overwrite)

        #Save hyperparams
        torch.save(hparams, os.path.join(model_dir, HYPERPARAMS_FILE))

        return model

    @classmethod
    def load(cls, task_name, checkpoint, **kwargs):
        """
        Loads a model from a checkpoint. Also loads hyperparams

        Parameters:
            task_name: Name of the task.
            checkpoint: Number indicating the checkpoint. -1 to load latest
            **kwargs: Additional key-value args passed to constructor
        """
        model_dir = gen_model_dir(task_name, cls)
        hparams_path = os.path.join(model_dir, HYPERPARAMS_FILE)

        if not os.path.exists(hparams_path):
            raise OSError('HParams file not found')

        hparams = torch.load(hparams_path)
        logger.info('Hyperparameters: {}'.format(str(hparams)))

        model = cls(hparams, **kwargs)
        if torch.cuda.is_available():
            model = model.cuda()

        if checkpoint == -1:
            # Find latest checkpoint file
            files = glob.glob(os.path.join(model_dir, CHECKPOINT_GLOB))
            if not files:
                raise OSError('Checkpoint files not found')
            files.sort(key=os.path.getmtime, reverse=True)
            checkpoint_path = files[0]
        else:
            checkpoint_path = os.path.join(model_dir, CHECKPOINT_FILE.format(checkpoint))
            if not os.path.exists(checkpoint_path):
                raise OSError('File not found: {}'.format(checkpoint_path))

        logger.info('Loading from {}'.format(checkpoint_path))
        # Load the model
        model.load_state_dict(torch.load(checkpoint_path))

        return model, hparams

    def save(self, task_name):
        """
        Save the model. Directory is determined by the task name and model class name
        """
        model_dir = gen_model_dir(task_name, self.__class__)
        checkpoint_path = os.path.join(model_dir, 'checkpoint-{}.pt'.format(int(self.iterations)))
        torch.save(self.state_dict(), checkpoint_path)
        logger.info('-- Saved checkpoint {}'.format(int(self.iterations)))

