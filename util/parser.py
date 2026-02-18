import argparse
from datetime import datetime
from pathlib import Path


def parse_bool(value):
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_yaml_scalar(value):
    value = value.split("#", 1)[0].strip()

    if value in {"", "null", "Null", "NULL", "~", "None"}:
        return None
    
    if (value.startswith('[') and value.endswith(']')):
        return [parse_yaml_scalar(v) for v in value[1:-1].split(',')]
    
    if (value.startswith('(') and value.endswith(')')):
        return (parse_yaml_scalar(v) for v in value[1:-1].split(','))

    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    lower_value = value.lower()
    if lower_value == "true":
        return True
    if lower_value == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def load_algorithm_defaults(algorithm_name):
    algorithm_name = algorithm_name.lower()
    config_path = (
        Path(__file__).resolve().parents[1]
        / "algorithm"
        / "config"
        / f"{algorithm_name}.yaml"
    )
    if not config_path.exists():
        raise ValueError(f"Algorithm config not found: {config_path}")

    defaults = {}
    with config_path.open("r", encoding="utf-8") as file:
        for line_no, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if ":" not in line:
                raise ValueError(
                    f"Invalid config format at {config_path}:{line_no} -> {raw_line.rstrip()}"
                )

            key, raw_value = line.split(":", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"Invalid empty key at {config_path}:{line_no}")

            defaults[key] = parse_yaml_scalar(raw_value)

    if not defaults:
        raise ValueError(f"No parameters found in config: {config_path}")

    defaults["algorithm"] = algorithm_name
    return defaults


def infer_arg_type(default_value):
    if isinstance(default_value, bool):
        return parse_bool
    if isinstance(default_value, int):
        return int
    if isinstance(default_value, float):
        return float
    return str


def build_parser(defaults):
    parser = argparse.ArgumentParser(description=None)
    for key, default_value in defaults.items():
        parser.add_argument(f"--{key}", default=default_value, type=infer_arg_type(default_value))
    return parser


def parse_run_args():
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--algorithm", default="bc", type=str)
    bootstrap_parser.add_argument("--mode", default="train", type=str)
    bootstrap_args, _ = bootstrap_parser.parse_known_args()

    selected_algorithm = bootstrap_args.algorithm.lower()
    selected_mode = bootstrap_args.mode.lower()
    try:
        default_args = load_algorithm_defaults(selected_algorithm)
    except ValueError as error:
        bootstrap_parser.error(str(error))

    # Ensure bootstrap options are always available in the final parser,
    # even when they are absent in algorithm config.
    default_args["algorithm"] = selected_algorithm
    default_args["mode"] = selected_mode

    parser = build_parser(default_args)
    args = parser.parse_args()
    if selected_mode in ["eval", "render"]:
        args = load_hparams_from_checkpoint(args)
    

    args.algorithm = args.algorithm.lower()
    args.mode = selected_mode.lower()
    args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    return args, parser, default_args


def print_used_parameters(args, defaults):
    print("\n" + "=" * 60)
    print("Parameters:")
    print("=" * 60)
    for key in defaults:
        print(f"  {key}: {getattr(args, key)}")
    print(f"  run_id: {args.run_id}")
    print("=" * 60 + "\n")

import json

def load_hparams_from_checkpoint(args):
    path = Path(args.checkpoint_path)

    # If a file is given, use it directly.
    if path.is_file():
        if path.suffix == ".json":
            json_path = path
        else:
            path = path.parent
    if path.is_dir():
        # If a directory is given, look for common json config names.
        file_name = "hparams.json"
        candidate = path / file_name
        if candidate.exists():
            json_path = candidate
        else:    
            raise ValueError(
                f"No json hyperparameter file found in: {path}. Checked: {candidate}"
            )

    # Load json
    with json_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)

    # Update args namespace in-place
    for k, v in loaded['exc_hparams'].items():
        setattr(args, k, v)

    # Ensure hparams_path remains what user passed
    args.hparams_path = str(path/args.hparams_path)
    args.checkpoint_path = str(path/args.checkpoint_path)
    args.results_path = str(path/args.results_path)

    return args
