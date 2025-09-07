import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value
from typing import Any, Dict, Optional, List, Iterable

import os
import csv



class Entry:
    def __init__(self, dt, output_dir, cfg):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)
    

    def _write_csv(self, filepath, data):
        """
        Write a heterogeneous logging dict to CSV, expanding vector-like values so that
        each element becomes its own column with an indexed suffix.

        Example:
        "root_pos" -> columns: root_pos_0, root_pos_1, root_pos_2
        "time"     -> column:  time

        Rules:
        - Scalars -> 1 column (the original key).
        - Vectors (list/tuple/np.ndarray/torch.Tensor) -> N columns named key_0..key_{N-1},
            where N is the maximum length for that key across all rows.
        - Shorter rows are padded with "" to match the maximum width for that key.
        """
        # Optional deps
        try:
            _HAS_NP = True
        except Exception:
            _HAS_NP = False

        try:
            import torch  # noqa: F401
            _HAS_TORCH = True
        except Exception:
            _HAS_TORCH = False

        def _to_1d_list(x: Any) -> List[Any]:
            """Flatten various types to a 1D Python list."""
            if x is None:
                return []
            if _HAS_TORCH and isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy().ravel().tolist()
            if _HAS_NP and hasattr(x, "shape"):
                try:
                    return np.asarray(x).ravel().tolist()
                except Exception:
                    pass
            if isinstance(x, (list, tuple)):
                if _HAS_NP:
                    try:
                        return np.asarray(x, dtype=object).ravel().tolist()
                    except Exception:
                        return _flatten_manual(x)
                return _flatten_manual(x)
            return [x]

        def _flatten_manual(seq: Iterable[Any]) -> List[Any]:
            out: List[Any] = []
            for el in seq:
                if isinstance(el, (list, tuple)):
                    out.extend(_flatten_manual(el))
                else:
                    if _HAS_TORCH and isinstance(el, torch.Tensor):
                        out.extend(el.detach().cpu().numpy().ravel().tolist())
                    elif _HAS_NP and hasattr(el, "shape"):
                        out.extend(np.asarray(el).ravel().tolist())
                    else:
                        out.append(el)
            return out

        keys = list(data.keys())  
        widths: Dict[str, int] = {}
        for k in keys:
            w = 1
            for v in data.get(k, []):
                w = max(w, len(_to_1d_list(v)))
            widths[k] = w

        headers: List[str] = []
        for k in keys:
            w = widths[k]
            if w == 1:
                headers.append(k)
            else:
                headers.extend([f"{k}_{i}" for i in range(w)])

        try:
            with open(filepath, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

                max_len = max((len(v) for v in data.values()), default=0)
                for i in range(max_len):
                    row: List[Optional[Any]] = []
                    for k in keys:
                        w = widths[k]
                        item = data[k][i] if i < len(data[k]) else None
                        flat = _to_1d_list(item)
                        if len(flat) < w:
                            flat = flat + [""] * (w - len(flat))
                        else:
                            flat = flat[:w]
                        row.extend(flat)
                    writer.writerow(row)
        except IOError as e:
            raise IOError(f"Failed to write CSV file {filepath}: {e}")


    def reset(self):
        self.state_log.clear()

