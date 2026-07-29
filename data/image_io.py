"""Reading images off Leonardo's Lustre filesystem without dying.

Under heavy parallel I/O (several jobs, several dataloader workers each, all on
the same dataset) Lustre sporadically returns a transient PermissionError or
OSError on an individual file — a file that has been read successfully for a
whole epoch can suddenly fail once, and is perfectly readable a second later.
Three 20h jobs died ~40 minutes in for exactly this reason.

The policy is deliberate: retry the SAME file with escalating backoff, never skip
it and never substitute another sample. A loader that quietly drops unreadable
files trains on a different dataset than the one it reports, and nothing in the
results would show it. If a file is still unreadable after every attempt the
exception propagates and the job fails — which is what should happen when the
permission problem is real rather than transient.
"""
import time

from PIL import Image


def open_image_retry(path, retries: int = 8, backoff: float = 0.5) -> Image.Image:
    """Open `path` as RGB, retrying transient filesystem errors.

    Worst case waits backoff * (1 + 2 + ... + retries) ≈ 18s at the defaults
    before giving up.
    """
    last_err = None
    for attempt in range(retries):
        try:
            return Image.open(path).convert("RGB")
        except (PermissionError, OSError) as e:
            last_err = e
            time.sleep(backoff * (attempt + 1))
    raise last_err


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        good = Path(d) / "good.png"
        Image.new("RGB", (8, 8), (1, 2, 3)).save(good)
        assert open_image_retry(good).size == (8, 8)

        # A file that fails a few times then succeeds must be recovered, not lost:
        # this is the whole point of the retry.
        calls = {"n": 0}
        real_open = Image.open

        def flaky(p, *a, **kw):
            calls["n"] += 1
            if calls["n"] <= 3:
                raise PermissionError(13, "transient", str(p))
            return real_open(p, *a, **kw)

        Image.open = flaky
        try:
            assert open_image_retry(good, backoff=0.0).size == (8, 8)
            assert calls["n"] == 4, f"expected 3 failures then success, got {calls['n']}"

            # A file that never opens must still raise, after exactly `retries`
            # attempts — silently returning something would corrupt the dataset.
            calls["n"] = 0
            Image.open = lambda p, *a, **kw: (_ for _ in ()).throw(
                PermissionError(13, "always", str(p)))
            try:
                open_image_retry(good, retries=5, backoff=0.0)
            except PermissionError:
                pass
            else:
                raise AssertionError("a permanently unreadable file must raise")
        finally:
            Image.open = real_open

    print("ok — retries recover transient failures and still raise on real ones")
