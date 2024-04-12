import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def add_OCR(parent_cs, cond_hp):
    # #################################################### OCR
    hparam_name = 'OCR'
    resize_target = CSH.UniformIntegerHyperparameter(
        name=hparam_name + '.' + 'transform_args.resize_target', lower=28, upper=128
    )
    parent_cs.add_hyperparameter(resize_target)
    parent_cs.add_condition(CS.EqualsCondition(resize_target, cond_hp, hparam_name))

    alpha = CSH.CategoricalHyperparameter(
        name=hparam_name + '.' + 'transform_args.elastic_coeffs',
        choices=[
            (0.75, 0.7),
            (0.5, 0.7),
            (0.25, 0.7),
            (0.75, 0.6),
            (0.5, 0.6),
            (0.25, 0.6),
            (0.75, 0.5),
            (0.5, 0.5),
            (0.25, 0.5),
            (0.75, 0.4),
            (0.5, 0.4),
            (0.25, 0.4),
        ]
    )
    parent_cs.add_hyperparameter(alpha)
    parent_cs.add_condition(CS.EqualsCondition(alpha, cond_hp, hparam_name))

    degrees = CSH.UniformFloatHyperparameter(
        name=hparam_name + '.' + 'transform_args.degrees', lower=0., upper=60.
    )
    parent_cs.add_hyperparameter(degrees)
    parent_cs.add_condition(CS.EqualsCondition(degrees, cond_hp, hparam_name))

    dist_scale = CSH.UniformFloatHyperparameter(
        name=hparam_name + '.' + 'transform_args.dist_scale', lower=0., upper=1.
    )
    parent_cs.add_hyperparameter(dist_scale)
    parent_cs.add_condition(CS.EqualsCondition(dist_scale, cond_hp, hparam_name))
