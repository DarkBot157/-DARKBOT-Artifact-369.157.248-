# benchmark/one_draw_vs_baseline.py

import time
import torch
import numpy as np
import faiss     # pip install faiss-cpu  or  faiss-gpu

from darkbot.core import DarkBot, DarkBotConfig

def brute_force_search(query: torch.Tensor, targets: torch.Tensor):
    dists = torch.norm(targets - query.unsqueeze(0), dim=1)
    return int(torch.argmin(dists))

def faiss_search(query: np.ndarray, index: faiss.Index):
    # returns the nearest neighbor id
    _, I = index.search(query[None, :].astype('float32'), 1)
    return int(I[0][0])

def benchmark(dim=512, n_targets=1024, iterations=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Prepare DARKBOT
    config = DarkBotConfig()
    db = DarkBot(config)
    db.device = device

    dtype = torch.complex64
    # Generate random targets on GPU
    targets = torch.randn(n_targets, dim, dtype=dtype, device=device)

    # Build FAISS index on CPU (float32 real part)
    targets_np = targets.real.cpu().numpy().astype('float32')
    index = faiss.IndexFlatL2(dim)
    index.add(targets_np)

    # Warm-up
    q = torch.randn(dim, dtype=dtype, device=device)
    db.one_draw_search(q, targets)
    if device.type=="cuda": torch.cuda.synchronize()

    # One-Draw Search
    t0 = time.perf_counter()
    for _ in range(iterations):
        q = torch.randn(dim, dtype=dtype, device=device)
        db.one_draw_search(q, targets)
    if device.type=="cuda": torch.cuda.synchronize()
    od_ms = (time.perf_counter() - t0)/iterations*1e3

    # Brute-Force Search
    if device.type=="cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iterations):
        q = torch.randn(dim, dtype=dtype, device=device)
        brute_force_search(q, targets)
    if device.type=="cuda": torch.cuda.synchronize()
    bf_ms = (time.perf_counter() - t0)/iterations*1e3

    # FAISS Search
    t0 = time.perf_counter()
    for _ in range(iterations):
        q_cpu = torch.randn(dim, dtype=dtype, device=device).real.cpu().numpy()
        faiss_search(q_cpu, index)
    faiss_ms = (time.perf_counter() - t0)/iterations*1e3

    print(f"Device         : {device}")
    print(f"One-Draw (𝒪₁)   : {od_ms:6.3f} ms")
    print(f"Brute-Force     : {bf_ms:6.3f} ms  → speedup {bf_ms/od_ms:4.1f}×")
    print(f"FAISS (L2)      : {faiss_ms:6.3f} ms  → speedup {faiss_ms/od_ms:4.1f}×")

if __name__ == "__main__":
    benchmark()
