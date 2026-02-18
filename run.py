from importlib import import_module
from util.parser import parse_run_args, print_used_parameters


ALGORITHM_MAP = {
    "bc": ("algorithm.bc", "BC"),
    "bco": ("algorithm.bco", "BCO"),
    "q": ("algorithm.q", "Q"),
    "dqn": ("algorithm.dqn", "DQN"),
    "random": ("algorithm.random", "RandomPolicy"),
    "vpg": ("algorithm.vpg", "VPG"),
    "vpg_a2c": ("algorithm.vpg_a2c", "VPG"),
    "vpg_gae": ("algorithm.vpg_gae", "VPG"),
    "ppo": ("algorithm.ppo", "PPO"),
}


def get_algorithm_class(algorithm_name):
    try:
        module_name, class_name = ALGORITHM_MAP[algorithm_name]
    except KeyError as error:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}") from error

    module = import_module(module_name)
    return getattr(module, class_name)


if __name__ == "__main__":
    args, parser, default_args = parse_run_args()
    print_used_parameters(args, default_args)

    try:
        algo_class = get_algorithm_class(args.algorithm)
    except ValueError as error:
        parser.error(str(error))

    algo = algo_class(args)
    algo.process()
