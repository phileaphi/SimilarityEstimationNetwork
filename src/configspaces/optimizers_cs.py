import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def add_Adam(parent_cs, cond_hp):
    # #################################################### Adam
    hparam_name = 'Adam'

    autotune_lr = CSH.CategoricalHyperparameter(
        name=hparam_name + '.' + 'optim_args.autotune_lr',
        choices=[True, False],
    )
    parent_cs.add_hyperparameter(autotune_lr)
    parent_cs.add_condition(CS.EqualsCondition(autotune_lr, cond_hp, hparam_name))

    lr = CSH.UniformFloatHyperparameter(
        name=hparam_name + '.' + 'optim_args.lr', lower=1e-9, upper=1e-1, log=True
    )
    parent_cs.add_hyperparameter(lr)
    parent_cs.add_condition(CS.EqualsCondition(lr, autotune_lr, False))

    amsgrad = CSH.CategoricalHyperparameter(
        name=hparam_name + '.' + 'optim_args.amsgrad',
        choices=[True, False],
    )
    parent_cs.add_hyperparameter(amsgrad)
    parent_cs.add_condition(CS.EqualsCondition(amsgrad, cond_hp, hparam_name))

    weight_decay = CSH.UniformFloatHyperparameter(
        name=hparam_name + '.' + 'optim_args.weight_decay',
        lower=1e-3,
        upper=0.999,
    )
    parent_cs.add_hyperparameter(weight_decay)
    parent_cs.add_condition(CS.EqualsCondition(weight_decay, cond_hp, hparam_name))


def add_AdamW(parent_cs, cond_hp):
    # #################################################### AdamW
    hparam_name = 'AdamW'

    autotune_lr = CSH.CategoricalHyperparameter(
        name=hparam_name + '.' + 'optim_args.autotune_lr',
        choices=[True, False],
    )
    parent_cs.add_hyperparameter(autotune_lr)
    parent_cs.add_condition(CS.EqualsCondition(autotune_lr, cond_hp, hparam_name))

    lr = CSH.UniformFloatHyperparameter(
        name=hparam_name + '.' + 'optim_args.lr', lower=1e-9, upper=1e-1, log=True
    )
    parent_cs.add_hyperparameter(lr)
    parent_cs.add_condition(CS.EqualsCondition(lr, autotune_lr, False))

    nesterov = CSH.CategoricalHyperparameter(
        name=hparam_name + '.' + 'optim_args.nesterov',
        choices=[True, False],
    )
    parent_cs.add_hyperparameter(nesterov)
    parent_cs.add_condition(CS.EqualsCondition(nesterov, cond_hp, hparam_name))

    amsgrad = CSH.CategoricalHyperparameter(
        name=hparam_name + '.' + 'optim_args.amsgrad',
        choices=[True, False],
    )
    parent_cs.add_hyperparameter(amsgrad)
    parent_cs.add_condition(CS.EqualsCondition(amsgrad, cond_hp, hparam_name))

    weight_decay = CSH.UniformFloatHyperparameter(
        name=hparam_name + '.' + 'optim_args.weight_decay',
        lower=1e-3,
        upper=0.999,
    )
    parent_cs.add_hyperparameter(weight_decay)
    parent_cs.add_condition(CS.EqualsCondition(weight_decay, cond_hp, hparam_name))
