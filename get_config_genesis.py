def get_config():
    """Get the hyperparameter configuration."""
    config = {}
    config['mode'] = "train"
    config['use_wandb'] = True
    config['use_cuda'] = True
    config['log_dir'] = "../logs"
    # Hyperparameters for dataset.
    config['view'] = 'all'
    config['return_info'] = False
    config['flip_rate'] = 0.3
    config['label_scheme_name'] = 'all'
    # must be compatible with number of unique values in label scheme
    # will be automatic in future update
    config['num_classes'] = 4
    # Hyperparameters for models.
    config['model'] = "tvn"
    config['pretrained'] = False
    config['loss_type'] = "cross_entropy" # cross_entropy/evidential/laplace_cdf
    # Hyperparameters for training.
    config['batch_size'] = 4
    config['num_epochs'] = 1000
    config['lr'] = 1e-4  
    # Self supervised learning or tuning
    config['self_sup'] = True
    config['train_mode'] = "reconstruction"
    return config
