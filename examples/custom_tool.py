import torch
from weighter import Harness, ToolContext, tool

net = torch.nn.Sequential(
    torch.nn.Linear(4, 16),
    torch.nn.Tanh(),
    torch.nn.Linear(16, 2),
)


@tool
def count_large_weights(ctx: ToolContext, threshold: float = 0.5) -> dict:
    """Count parameters whose absolute value exceeds a threshold."""
    counts = {}
    with torch.no_grad():
        for name, tensor in ctx.store.module.named_parameters():
            counts[name] = int((tensor.abs() > threshold).sum())
    return {
        "threshold": threshold,
        "counts": counts,
        "run_label": ctx.extras.get("run_label"),
    }


harness = Harness(
    net,
    llm="openai/gpt-4.1-mini",
    tools=[count_large_weights],
    exclude_tools=["run_probe"],
    tool_extras={"run_label": "weight audit"},
)
result = harness.prompt(
    'Call count_large_weights with {"threshold": 0.5} and summarize the result.'
)
print(result.text)
