import torch

def apply_gaussian_noise(tensor: torch.Tensor, sigma: float):
    if sigma <= 0:
        return tensor
    noise = torch.randn_like(tensor) * sigma
    return tensor + noise


def clip_gradients(model: torch.nn.Module, max_norm: float):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def consume_epsilon(client, eps_cost: float):
    client.dp_epsilon_used += eps_cost
    client.dp_epsilon_remaining = max(0.0, client.dp_epsilon_remaining - eps_cost)
