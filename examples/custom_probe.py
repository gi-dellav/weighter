import torch
from weighter import Harness

net = torch.nn.Sequential(
    torch.nn.Linear(4, 16),
    torch.nn.Tanh(),
    torch.nn.Linear(16, 2),
)


def probe(module, payload):
    inputs = torch.tensor(payload["inputs"], dtype=torch.float32)
    with torch.no_grad():
        logits = module(inputs)
    return {
        "shape": list(logits.shape),
        "logits": logits.tolist(),
        "mean": float(logits.mean()),
    }


harness = Harness(
    net,
    llm="openai/gpt-4.1-mini",
    probe=probe,
    probe_budget=20,
)
result = harness.prompt(
    'Use run_probe with payload {"inputs": [[1, 2, 3, 4]]}, '
    "then explain which output unit dominates and why."
)
print(result.text)
