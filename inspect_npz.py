import numpy as np

p = "data/features/vod_001.npz"
d = np.load(p)

print("keys:", d.files)
for k in d.files:
    a = d[k]
    if hasattr(a, "shape"):
        print(f"{k:10s} shape={a.shape} dtype={a.dtype}")
