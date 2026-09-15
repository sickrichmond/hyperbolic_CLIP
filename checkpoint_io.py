"""Checkpoint writes that preserve the previous file when saving fails."""
import os
from pathlib import Path
import tempfile

import torch


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
