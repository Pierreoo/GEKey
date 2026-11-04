import os
import multiprocessing as mp

import torch


def get_data_wrapper(args):
    instance, idx = args
    data_name = instance.data_list[idx]
    return data_name, instance.get_data(idx)


def process_data_parallel(instance, logger):
    logger.info(f"Starting parallel parsing with {mp.cpu_count()} workers...")
    data_dict = {}
    total = len(instance.data_list)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        for i, (name, data) in enumerate(
                pool.imap_unordered(
                    get_data_wrapper,
                    [(instance, idx) for idx in range(total)]
                ), 1):
            data_dict[name] = data
            logger.info(f"[{i}/{total}] Processed: {name}")
    return data_dict


def save_frame_wrapper(args):
    """Build one frame's record and write it to its own .pth.

    Returns only the name and whether it was skipped, so the sample itself is
    never sent back through IPC.
    """
    instance, idx = args
    data_name = instance.data_list[idx]
    frame_path = instance._get_frame_path(data_name)

    if os.path.isfile(frame_path):
        return data_name, True

    frame_data = instance.get_data_single(data_name)

    os.makedirs(os.path.dirname(frame_path), exist_ok=True)
    # Write to a per-process temp file first so a crashed or concurrent worker
    # can never leave a half-written record behind.
    tmp_path = f"{frame_path}.tmp{os.getpid()}"
    torch.save(frame_data, tmp_path)
    os.replace(tmp_path, frame_path)
    return data_name, False


def process_frames_parallel(instance, logger, num_workers=None):
    """Parallel counterpart of process_data_parallel for datasets that cache one
    .pth per sample instead of a single record dict. Returns counts, not data."""
    num_workers = num_workers or mp.cpu_count()
    total = len(instance.data_list)
    logger.info(f"Starting parallel record build with {num_workers} workers...")

    processed = skipped = 0
    with mp.Pool(processes=num_workers) as pool:
        for i, (name, was_skipped) in enumerate(
                pool.imap_unordered(
                    save_frame_wrapper,
                    [(instance, idx) for idx in range(total)]
                ), 1):
            if was_skipped:
                skipped += 1
                logger.info(f"[{i}/{total}] Already exists: {name}")
            else:
                processed += 1
                logger.info(f"[{i}/{total}] Processed: {name}")
    logger.info(f"Record build complete: {processed} processed, {skipped} already existed")
    return processed, skipped
