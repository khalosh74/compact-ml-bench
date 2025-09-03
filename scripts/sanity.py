# quick forward/backward sanity on random data
import torch, time
from torch import nn

torch.manual_seed(42)
device = "cpu"  # Codespaces default; GPU not expected
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(32*32*3, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
).to(device)

x = torch.randn(128, 3, 32, 32, device=device)
y = torch.randint(0, 10, (128,), device=device)

opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()

# 3 tiny steps
model.train()
for step in range(3):
    opt.zero_grad(set_to_none=True)
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    opt.step()
    print(f"[SANITY] step={step} loss={loss.item():.4f}")

# latency smoke check (batch=1)
model.eval()
with torch.inference_mode():
    x1 = torch.randn(1, 3, 32, 32, device=device)
    for _ in range(20): model(x1)  # warmup
    t0 = time.perf_counter()
    N=100
    for _ in range(N): model(x1)
    dt = (time.perf_counter() - t0)/N
    print(f"[SANITY] ~{dt*1000:.2f} ms/inference (CPU)")
print("[SANITY] OK")
