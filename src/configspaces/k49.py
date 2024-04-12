import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

K49CS = CS.ConfigurationSpace()
BASELINECNN_CFG = {
    'dataset': 'K49',
    'epochs': 10,
    'batch_size': 96,
    'model': 'BaselineCNN',
    'loss_fn': 'CrossEntropyLoss',
    'optimizer': 'Adam',
    'optim_args': {},
    'model_args': {
        'n_layers': 1
    },
    'transforms': 'NONE',
    'transform_args':
        {
            'degrees': 60.,
            'dist_scale': 0.2,
            'elastic_coeffs': [0.75, 0.4],
            'resize_target': 28
        },
    'autotune_lr': True,
    'warmstart': False,
}
RESNET18_CFG = {
    'dataset': 'K49',
    'epochs': 10,
    'batch_size': 96,
    'model': 'Resnet',
    'loss_fn': 'CrossEntropyLoss',
    'optimizer': 'AdamW',
    'optim_args': {
        # "amsgrad": True,
        # "nesterov": True,
        # "weight_decay": 0.138,
    },
    'model_args':
        {
            'block': 'BasicBlock',
            'layers': [2, 2, 2, 2],
            "zero_init_residual": False,
            "dropout": 0.,
        },
    'transforms': 'OCR',
    'transform_args':
        {
            'degrees': 60.,
            'dist_scale': 0.2,
            'elastic_coeffs': [0.75, 0.4],
            'resize_target': 64
        },
    'autotune_lr': True,
    'warmstart': False,
}

# ####################################################
#                     DataLoader
# ####################################################
# https://arxiv.org/pdf/1711.00489.pdf
batch_size = CSH.UniformIntegerHyperparameter(name='batch_size', lower=96, upper=1000)
K49CS.add_hyperparameter(batch_size)

# ####################################################
#                     Optimizers
# ####################################################
# https://arxiv.org/pdf/1506.01186.pdf
optim = CSH.CategoricalHyperparameter(name='optimizer', choices=['Adam', 'AdamW'])
K49CS.add_hyperparameter(optim)

amsgrad = CSH.CategoricalHyperparameter(
    name='optim_args.amsgrad',
    choices=[True, False],
)
K49CS.add_hyperparameter(amsgrad)

weight_decay = CSH.UniformFloatHyperparameter(
    name='optim_args.weight_decay',
    lower=1e-3,
    upper=0.999,
)
K49CS.add_hyperparameter(weight_decay)

nesterov = CSH.CategoricalHyperparameter(
    name='optim_args.nesterov',
    choices=[True, False],
)
K49CS.add_hyperparameter(nesterov)

# ####################################################
#                     Models
# ####################################################
model = CSH.CategoricalHyperparameter(name='model', choices=['BaselineCNN', 'Resnet'])
K49CS.add_hyperparameter(model)

warmstart = CSH.CategoricalHyperparameter(
    name='warmstart',
    choices=[True, False],
)
K49CS.add_hyperparameter(warmstart)

dropout = CSH.UniformFloatHyperparameter(
    name='model_args.dropout',
    lower=0.,
    upper=0.8,
)
K49CS.add_hyperparameter(dropout)

zero_init_res = CSH.CategoricalHyperparameter(
    name='model_args.zero_init_residual', choices=[True, False]
)
K49CS.add_hyperparameter(zero_init_res)

# ####################################################
#                     Transformations
# ####################################################
transforms = CSH.CategoricalHyperparameter(name='transforms', choices=['NONE', 'OCR'])
K49CS.add_hyperparameter(transforms)
