from __future__ import annotations

import re
import threading

import numpy as np
import torch
import torch.nn.functional as F

from alphachess.config import Config
from alphachess.nn.model import AlphaChessNet
from alphachess.storage import Storage


_GENERATION_RE = re.compile(r"(\d+)\.pt$")


class InferenceModel:
    """Wraps a loaded AlphaChessNet with batched prediction and hot-reload.

    Caller calles maybe_reload(), if a new gen model found, model is swapped
    atomically under a lock so predict_batch is never called on a 
    half-swapped model.
    """

    def __init__(
        self,
        storage: Storage,
        device: str | torch.device,
        config: Config,
    ) -> None:
        self._storage = storage
        self._device = torch.device(device) if isinstance(device, str) else device
        self._config = config
        self._lock = threading.Lock()

        gen = self._newest_generation()
        if gen is None:
            raise FileNotFoundError(
                "No model checkpoints found in storage under 'models/'. "
                "Run pretrain or bootstrap first."
            )
        self._generation: int = gen
        self._model: AlphaChessNet = self._load_generation(gen)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def current_generation(self) -> int:
        return self._generation

    def maybe_reload(self) -> bool:
        """Check storage for a newer generation.

        If found, load and swap self._model under self._lock.
        Returns True if swap, False otherwise.
        """
        newest = self._newest_generation()
        if newest is None or newest <= self._generation:
            return False

        new_model = self._load_generation(newest)
        with self._lock:
            self._model = new_model
            self._generation = newest
        return True

    def predict_batch(self, encoded_boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run a forward pass on a batch of encoded board states.

        Args:
            encoded_boards (float32): shape of [B, planes, 8, 8]: 

        Returns
        -------
        policy_priors : np.ndarray
            Shape [B, 4672], softmax probabilities.
        values : np.ndarray
            Shape [B], scalar in [-1, 1].

        Internally moves tensors to self._device, runs under torch.no_grad(),
        and returns numpy arrays on CPU.
        """
        x = torch.from_numpy(np.ascontiguousarray(encoded_boards)).to(
            self._device, dtype=torch.float32, non_blocking=True
        )
        with self._lock:
            model = self._model
        with torch.no_grad():
            logits, value = model(x)
            priors = F.softmax(logits, dim=-1)

        priors_np = priors.detach().to("cpu").numpy().astype(np.float32, copy=False)
        values_np = (
            value.detach().to("cpu").numpy().reshape(-1).astype(np.float32, copy=False)
        )
        return priors_np, values_np

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_generation(self, generation: int) -> AlphaChessNet:
        model = AlphaChessNet.load_from(self._storage, generation)
        model.to(self._device)
        model.eval()
        return model

    def _newest_generation(self) -> int | None:
        names = [
            n for n in self._storage.list("models", ".pt")
            if _GENERATION_RE.search(n)
        ]
        if not names:
            return None
        match = _GENERATION_RE.search(names[-1])
        assert match is not None
        return int(match.group(1))
