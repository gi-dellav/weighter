import torch
from weighter import Harness

net = torch.nn.Sequential(
    torch.nn.Linear(4, 16),
    torch.nn.Tanh(),
    torch.nn.Linear(16, 2),
)

harness = Harness(net, llm="openai/gpt-4.1-mini")
result = harness.prompt(
    "Inspect the weights and tell me what task this network most likely computes."
)
print(result.text)
