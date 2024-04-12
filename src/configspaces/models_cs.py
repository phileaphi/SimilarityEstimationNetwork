import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def add_Resnet(parent_cs, cond_hp):
    # #################################################### Resnet
    hparam_name = 'Resnet'

    warmstart = CSH.CategoricalHyperparameter(
        name=hparam_name + '.' + 'warmstart', choices=[1, 0]
    )
    parent_cs.add_hyperparameter(warmstart)
    parent_cs.add_condition(CS.EqualsCondition(warmstart, cond_hp, hparam_name))

    block = CSH.CategoricalHyperparameter(
        name=hparam_name + '.' + 'model_args.block', choices=['BasicBlock', 'Bottleneck']
    )
    parent_cs.add_hyperparameter(block)
    parent_cs.add_condition(CS.EqualsCondition(block, cond_hp, hparam_name))

    layer1 = CSH.UniformIntegerHyperparameter(
        name=hparam_name + '.' + 'model_args.res_layer1', lower=2, upper=4
    )
    parent_cs.add_hyperparameter(layer1)
    parent_cs.add_condition(CS.EqualsCondition(layer1, cond_hp, hparam_name))

    layer2 = CSH.UniformIntegerHyperparameter(
        name=hparam_name + '.' + 'model_args.res_layer2', lower=2, upper=16
    )
    parent_cs.add_hyperparameter(layer2)
    parent_cs.add_condition(CS.EqualsCondition(layer2, cond_hp, hparam_name))

    layer3 = CSH.UniformIntegerHyperparameter(
        name=hparam_name + '.' + 'model_args.res_layer3', lower=2, upper=36
    )
    parent_cs.add_hyperparameter(layer3)
    parent_cs.add_condition(CS.EqualsCondition(layer3, cond_hp, hparam_name))

    layer4 = CSH.UniformIntegerHyperparameter(
        name=hparam_name + '.' + 'model_args.res_layer4', lower=2, upper=4
    )
    parent_cs.add_hyperparameter(layer4)
    parent_cs.add_condition(CS.EqualsCondition(layer4, cond_hp, hparam_name))

    dropout = CSH.UniformFloatHyperparameter(
        name=hparam_name + '.' + 'model_args.dropout', lower=0., upper=0.8
    )
    parent_cs.add_hyperparameter(dropout)
    parent_cs.add_condition(CS.EqualsCondition(dropout, cond_hp, hparam_name))

    zero_init_res = CSH.CategoricalHyperparameter(
        name=hparam_name + '.' + 'model_args.zero_init_residual', choices=[True, False]
    )
    parent_cs.add_hyperparameter(zero_init_res)
    parent_cs.add_condition(CS.EqualsCondition(zero_init_res, cond_hp, hparam_name))
