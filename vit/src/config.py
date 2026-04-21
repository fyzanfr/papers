import ml_collections

def get_miniVit_config():
    ''' Minimal Configuration for the ViT '''

    config = ml_collections.ConfigDict()
    
    # Model Hyperparameters
    config.model = model = ml_collections.ConfigDict()
    model.patch_size = 4
    model.hidden_size = 128
    model.num_layers = 6
    model.num_heads = 8
    model.mlp_dim = 512
    model.dropout_rate = 0.1
    model.attn_dropout_rate = 0.1
    
    # Dataset config
    model.num_classes = 10
    model.image_size = 32
    model.channels = 3

    # Training Hyperparameters
    config.training = training = ml_collections.ConfigDict()
    training.learning_rate = 1e-3
    training.weight_decay = 0.05
    training.batch_size = 128
    training.epochs = 100
    
    return config
