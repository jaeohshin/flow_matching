# 🐍 Python for Flow Matching – Cheat Sheet

## 1. Tensors & Shapes
```python
x = torch.randn(32, 2)      # shape: [batch_size, dim]
t = torch.rand(32, 1)       # shape: [32, 1] ← enables broadcasting
x_t = (1 - t) * x0 + t * x1
```

## 2. Indexing & Slicing
```python
x[0]         # first element
x[:, 1]      # second column
x[:5]        # first 5 rows
```

## 3. Functions
```python
def interpolate(x0, x1, t):
    return (1 - t) * x0 + t * x1
```

## 4. Loop
```python
for epoch in range(100):
    ...
```

## 5. List & Dict
```python
layers = [nn.Linear(64, 64), nn.ReLU()]
params = {'lr': 1e-3, 'batch_size': 128}
```

## 6. Broadcasting
```python
# t: [32, 1], x: [32, 2] → (1 - t) * x works automatically
```

## 7. No Gradient / Eval Mode
```python
with torch.no_grad():
    sample = model(x)

model.eval()   # turn off dropout, batchnorm
```

## 8. Class & Model
```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)
```

## 9. Training Step
```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## 🔧 Pro Tips

| You see this | Do this |
|--------------|---------|
| `len(x)` | Get batch size |
| `x.view(-1, 1)` | Reshape tensor |
| `nn.Sequential(...)` | Quick network |
| `t = torch.rand(batch_size, 1)` | For per-sample interpolation |
| `x0 + t * (x1 - x0)` | Same as `(1 - t) * x0 + t * x1` |