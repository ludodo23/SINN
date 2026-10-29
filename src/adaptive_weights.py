
import torch


class GradientBalancedWeights:
    """Équilibrage global des pertes par EMA des normes de gradients."""

    def __init__(
        self,
        names,
        beta=0.95,
        min_weight=0.1,
        max_weight=10.0,
    ):
        self.names = tuple(names)
        self.beta = beta
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.ema = None

    def update(self, losses, model):
        params = [
            p for p in model.parameters() if p.requires_grad
        ]
        norms = []

        for name in self.names:
            grads = torch.autograd.grad(
                losses[name],
                params,
                retain_graph=True,
                allow_unused=True,
            )
            terms = [
                g.detach().square().sum()
                for g in grads if g is not None
            ]
            norm = (
                torch.stack(terms).sum().sqrt()
                if terms else losses[name].new_tensor(0.0)
            )
            norms.append(norm)

        norms = torch.stack(norms).clamp_min(1e-12)

        if self.ema is None:
            self.ema = norms.detach().clone()
        else:
            self.ema = (
                self.beta * self.ema
                + (1.0 - self.beta) * norms.detach()
            )

        target = torch.exp(torch.log(self.ema).mean())
        weights = target / self.ema
        weights = weights.clamp(self.min_weight, self.max_weight)
        weights = weights / weights.mean()

        return {
            name: weights[i].detach()
            for i, name in enumerate(self.names)
        }
