"""Checkpoint writes that preserve the previous file when saving fails."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import tempfile

import torch


@contextmanager
def attribution_checkpoint_output(destination, *, train_augment, aug_policy, hierarchy):
    """Lock an attribution output for the run and reject incompatible saved modes."""
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lock file: unlinking it lets competing jobs lock different inodes.
    with destination.with_name(f".{destination.name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another training run is using checkpoint {destination}. "
                "Choose a different CKPT/--output."
            ) from exc
        if destination.exists():
            saved = torch.load(destination, map_location="cpu", weights_only=False)
            saved_mode = (saved.get("aug_policy", "corruption")
                          if saved.get("train_augment", False) else "none")
            requested_mode = aug_policy if train_augment else "none"
            saved_hierarchy = saved.get("hierarchy", "none")
            if (saved_mode, saved_hierarchy) != (requested_mode, hierarchy):
                raise ValueError(
                    f"Refusing to overwrite checkpoint {destination}: "
                    f"saved augmentation={saved_mode}, hierarchy={saved_hierarchy}; "
                    f"requested augmentation={requested_mode}, hierarchy={hierarchy}. "
                    "Choose a different CKPT/--output."
                )
            del saved
        yield


def atomic_torch_save(payload, destination):
    """Flush a complete checkpoint beside its destination, then replace it atomically.

    Needs space/quota for the new checkpoint in addition to any existing one. The
    temporary file must be on the same filesystem for os.replace to be atomic.
    """
    destination = Path(destination)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"Could not save checkpoint to {destination}. Any previous destination "
            "file was not replaced. Check free space, user/project quota, and filesystem "
            "health (on CINECA: cindata / cinQuota). Atomic saving requires space for "
            "one additional complete checkpoint."
        ) from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # A storage outage may also prevent cleanup; preserve the save error.
                pass
