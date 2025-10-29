import dataclasses
import logging
import multiprocessing as mp
import random
from typing import Iterable

import tyro

import numpy as np

from robocasa_args import Args
from robocasa_evaluator import eval_robocasa
from robocasa_tasks import pick_tasks


def _chunk_tasks(tasks: Iterable[str], num_workers: int) -> list[list[str]]:
    buckets: list[list[str]] = [[] for _ in range(max(1, num_workers))]
    for idx, task in enumerate(tasks):
        buckets[idx % len(buckets)].append(task)
    return [bucket for bucket in buckets if bucket]


def _prepare_worker_args(base: Args, task_subset: list[str], worker_id: int) -> Args:
    return dataclasses.replace(
        base,
        task_names=tuple(task_subset),
        seed=base.seed + worker_id,
        model_name=f"{base.model_name}_w{worker_id}",
        num_workers=1,
        eval_all=False,
    )


def _worker_main(worker_id: int, args: Args) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=f"[worker-{worker_id}] %(asctime)s %(levelname)s %(message)s",
    )
    logging.info("Starting evaluation for tasks: %s", list(args.task_names or []))
    eval_robocasa(args)
    logging.info("Worker finished.")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    mp.set_start_method("spawn", force=True)
    args = tyro.cli(Args)

    if args.num_workers <= 1:
        eval_robocasa(args)
        return

    # Determine task list once in the parent to avoid duplication.
    if args.task_names:
        base_tasks = list(args.task_names)
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        base_tasks = pick_tasks(args)

    if not base_tasks:
        logging.info("No tasks selected for evaluation. Exiting.")
        return

    task_chunks = _chunk_tasks(base_tasks, args.num_workers)
    logging.info("Dispatching %d tasks across %d workers.", len(base_tasks), len(task_chunks))

    processes: list[mp.Process] = []
    for worker_id, subset in enumerate(task_chunks):
        worker_args = _prepare_worker_args(args, subset, worker_id)
        proc = mp.Process(target=_worker_main, args=(worker_id, worker_args), daemon=False)
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"Worker process exited with code {proc.exitcode}")


if __name__ == "__main__":
    main()
