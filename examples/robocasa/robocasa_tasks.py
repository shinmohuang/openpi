import logging
import random
import re
from functools import lru_cache
from typing import Sequence

import robosuite
from robosuite.controllers import load_composite_controller_config

from robocasa.utils.dataset_registry import (
    MULTI_STAGE_TASK_DATASETS,
    SINGLE_STAGE_TASK_DATASETS,
)

from robocasa_args import Args

logger = logging.getLogger(__name__)

CAMELCASE_PATTERN = re.compile(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[0-9]+")
PNP_PATTERN = re.compile(r"PnP(?P<src>[A-Za-z0-9]+)To(?P<dst>[A-Za-z0-9]+)")

PNP_LOCATION_CANONICAL = {
    "counter": "counter",
    "cab": "cabinet",
    "cabinet": "cabinet",
    "sink": "sink",
    "microwave": "microwave",
    "stove": "stove",
    "drawer": "drawer",
    "pantry": "pantry shelves",
    "plate": "plate",
}

PNP_SOURCE_PHRASES = {
    "counter": "from the counter",
    "cabinet": "from inside the cabinet",
    "sink": "from the sink",
    "microwave": "from inside the microwave",
    "stove": "from the stove area",
    "drawer": "from the drawer",
    "pantry shelves": "from the pantry shelves",
    "plate": "from the plate",
}

PNP_DESTINATION_PHRASES = {
    "counter": "on the counter",
    "cabinet": "inside the cabinet",
    "sink": "into the sink",
    "microwave": "inside the microwave",
    "stove": "onto the stove area",
    "drawer": "into the drawer",
    "pantry shelves": "onto the pantry shelves",
    "plate": "onto the plate",
}

TARGET_ALIASES = {
    "SingleDoor": "single-door cabinet",
    "DoubleDoor": "double-door cabinet",
    "Drawer": "drawer",
    "SinkFaucet": "sink faucet",
    "SinkSpout": "sink spout",
    "Microwave": "microwave",
    "Stove": "stove burner",
    "MicrowaveDoor": "microwave door",
    "Door": "door",
}

TOKEN_ALIASES = {
    "PnP": "pick and place",
    "Counter": "counter",
    "Cab": "cabinet",
    "Cabinet": "cabinet",
    "Sink": "sink",
    "Microwave": "microwave",
    "Stove": "stove",
    "Open": "open",
    "Close": "close",
    "Drawer": "drawer",
    "Door": "door",
    "Turn": "turn",
    "On": "on",
    "Off": "off",
    "Faucet": "faucet",
    "Spout": "spout",
    "Navigate": "navigate",
    "Kitchen": "kitchen",
    "Coffee": "coffee",
    "Setup": "setup",
    "Serve": "serve",
    "Press": "press",
    "Button": "button",
    "Prepare": "prepare",
    "Vegetables": "vegetables",
    "Restock": "restock",
    "Pantry": "pantry",
    "Pre": "pre",
    "Soak": "soak",
    "Pan": "pan",
}

FIXTURE_PRIORITY: Sequence[str] = (
    "door_fxtr",
    "drawer",
    "coffee_machine",
    "microwave",
    "stove",
    "sink",
    "cab",
    "counter",
    "target_fixture",
    "src_fixture",
)

TASK_FRONT_APPROACH: set[str] = {
    "OpenSingleDoor",
    "CloseSingleDoor",
    "OpenDoubleDoor",
    "CloseDoubleDoor",
    "OpenDrawer",
    "CloseDrawer",
    "TurnOnSinkFaucet",
    "TurnOffSinkFaucet",
    "TurnSinkSpout",
    "TurnOnMicrowave",
    "TurnOffMicrowave",
    "TurnOnStove",
    "TurnOffStove",
    "CoffeePressButton",
}

TASK_SKIP_EE: set[str] = {
    "NavigateKitchen",
}

TASK_FRONT_DISTANCE: dict[str, float] = {
    "TurnOnMicrowave": 0.22,
    "TurnOffMicrowave": 0.22,
    "TurnOnStove": 0.20,
    "TurnOffStove": 0.20,
    "CoffeePressButton": 0.18,
}


def task_prompt(task_name: str, env=None) -> str:
    prompt = None
    if env is not None:
        prompt = _prompt_from_env(env)
    if not prompt:
        prompt = _env_task_prompt(task_name)
    if prompt:
        return prompt
    return _fallback_prompt(task_name)


def _pick_and_place_prompt(task_name: str) -> str | None:
    match = PNP_PATTERN.match(task_name)
    if not match:
        return None
    src_raw = match.group("src")
    dst_raw = match.group("dst")
    src = _canonical_location(src_raw)
    dst = _canonical_location(dst_raw)
    source_phrase = PNP_SOURCE_PHRASES.get(src, f"from the {src}")
    dest_phrase = PNP_DESTINATION_PHRASES.get(dst, f"onto the {dst}")
    return f"Pick an item {source_phrase} and place it {dest_phrase}."


def _canonical_location(name: str) -> str:
    key = name.lower()
    if key in PNP_LOCATION_CANONICAL:
        return PNP_LOCATION_CANONICAL[key]
    return key.replace("_", " ")


def _action_prompt(task_name: str, prefix: str, action: str) -> str | None:
    if not task_name.startswith(prefix):
        return None
    remainder = task_name[len(prefix) :]
    if not remainder:
        return f"{action} the target object."
    remainder = remainder.replace("_", "")
    target = TARGET_ALIASES.get(remainder)
    if target is None:
        tokens = CAMELCASE_PATTERN.findall(remainder)
        if not tokens:
            target = remainder.lower()
        else:
            words = [TOKEN_ALIASES.get(tok, tok.lower()) for tok in tokens]
            target = " ".join(words)
            target = target.replace("single door", "single-door").replace("double door", "double-door")
    article = "the" if not target.startswith(("a ", "an ")) else ""
    target_phrase = f"{article} {target}".strip()
    return f"{action} {target_phrase}."


def _prompt_from_env(env) -> str | None:
    get_meta = getattr(env, "get_ep_meta", None)
    if not callable(get_meta):
        return None
    try:
        meta = get_meta()
    except Exception:
        logger.debug("Failed to fetch episode metadata from env", exc_info=True)
        return None
    if isinstance(meta, dict):
        lang = meta.get("lang")
        if isinstance(lang, str):
            lang = lang.strip()
            if lang:
                return lang
    return None


@lru_cache(maxsize=None)
def _env_task_prompt(task_name: str) -> str | None:
    """Instantiate a light-weight environment and extract its language prompt."""
    try:
        controller_config = load_composite_controller_config(
            controller=None,
            robot="PandaOmron",
        )
        # Use minimal observation settings; cameras are not required for language generation.
        env = robosuite.make(
            env_name=task_name,
            robots="PandaOmron",
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=False,
            camera_depths=False,
            seed=0,
            obj_instance_split="B",
            translucent_robot=False,
        )
        try:
            env.reset()
            meta = env.get_ep_meta()
            if isinstance(meta, dict):
                lang = meta.get("lang")
                if isinstance(lang, str) and lang.strip():
                    return lang.strip()
        finally:
            try:
                env.close()
            except Exception:
                logger.debug("Failed to close environment for task %s", task_name, exc_info=True)
    except Exception:
        logger.warning("Falling back to heuristic prompt for task %s", task_name, exc_info=True)
    return None


def _fallback_prompt(task_name: str) -> str:
    pnp_prompt = _pick_and_place_prompt(task_name)
    if pnp_prompt:
        return pnp_prompt

    for prefix, action in (
        ("TurnOn", "Turn on"),
        ("TurnOff", "Turn off"),
        ("Turn", "Turn"),
        ("Open", "Open"),
        ("Close", "Close"),
    ):
        prompt = _action_prompt(task_name, prefix, action)
        if prompt:
            return prompt

    normalized = task_name.replace("_", " ")
    tokens = CAMELCASE_PATTERN.findall(normalized)
    if tokens:
        tokens = [TOKEN_ALIASES.get(tok, tok.lower()) for tok in tokens]
        return f"Please complete the task: {' '.join(tokens)}."
    return f"Please complete the task: {task_name}."


def pick_tasks(args: Args) -> list[str]:
    if args.eval_all and args.task_names:
        raise ValueError("Cannot specify both --all and explicit --task-names.")
    if args.eval_all:
        registry_all = MULTI_STAGE_TASK_DATASETS if args.use_multi_stage else SINGLE_STAGE_TASK_DATASETS
        return list(registry_all.keys())
    registry = MULTI_STAGE_TASK_DATASETS if args.use_multi_stage else SINGLE_STAGE_TASK_DATASETS
    all_names = list(registry.keys())
    if args.task_names:
        for name in args.task_names:
            if name not in registry:
                raise ValueError(f"Unknown task '{name}'. Available: {all_names[:8]} ... total={len(all_names)}")
        return list(args.task_names)
    random.shuffle(all_names)
    return all_names[:5]


def task_horizon(task_name: str, *, use_multi_stage: bool) -> int:
    registry = MULTI_STAGE_TASK_DATASETS if use_multi_stage else SINGLE_STAGE_TASK_DATASETS
    return int(registry[task_name].get("horizon", 500))
