import json
from typing import Any
import importlib

def resolve_function(fully_qualified_name: str):
    module_name, func_name = fully_qualified_name.rsplit('.', 1)
    module = importlib.import_module(f"code.{module_name}")
    return getattr(module, func_name)

def parse_config(file_path: str):
    with open(file_path) as file:
        config: dict[str, Any] = json.load(file)
        amount_of_chains: int = config.get("amount_of_chains", 3000)
        target_length: int = config.get("target_length", 1000)
        do_perm: bool = config.get("do_perm", False)
        w_low: float = config.get("w_low", 0.316)
        w_high: float = config.get("w_high", 3.16)
        dimension: int = config.get("dimension", 2)
        next_sides_function_str: str = config.get("next_sides_function", "simulate.get_allowed_sides_2d")
        next_sides_function = resolve_function(next_sides_function_str)
        return (
            amount_of_chains,
            target_length,
            do_perm,
            w_low,
            w_high,
            dimension,
            next_sides_function,
        )
