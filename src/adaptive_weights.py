
import torch


class GradNormWeights:
    """GradNorm-based adaptive weighting of multiple loss terms.

    Parameters
    ----------
    names : iterable of str
        Names of the loss terms to balance (must match keys of the
        ``losses`` mapping passed to :meth:`update`).
    alpha : float, optional
        Asymmetry parameter.  ``alpha = 0`` balances the gradient
        magnitudes only; larger values push more gradient budget towards
        the tasks that are learning the slowest.  Default is 0.5.
    lr : float, optional
        Learning rate used by the internal Adam optimiser that updates the
        (log-)weights.  Default is 1e-3.
    min_weight, max_weight : float, optional
        Clamping bounds applied to the weights after each update.
    """

    def __init__(
        self,
        names,
        alpha: float = 0.5,
        lr: float = 1e-3,
        min_weight: float = 0.1,
        max_weight: float = 10.0,
        shared_params=None,
    ):
        self.names = tuple(names)
        self.alpha = float(alpha)
        self.lr = float(lr)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.shared_params = shared_params
        self._resolved_params = None

        # Lazily created on the first call to ``update`` so that the
        # parameters live on the same device/dtype as the losses.
        self._weights = None
        self._optimizer = None
        self.initial_losses = None

    @property
    def weights(self):
        return self._weights

    def _init_weights(self, device, dtype):
        self._weights = {
            name: torch.nn.Parameter(
                torch.ones((), device=device, dtype=dtype)
            )
            for name in self.names
        }
        self._optimizer = torch.optim.Adam(
            list(self._weights.values()), lr=self.lr
        )

    def _resolve_shared_params(self, model):
        """Resolve shared_params into a concrete list of parameters.

        Parameters
        ----------
        model : torch.nn.Module
            Model whose parameters will be used for gradient norms.

        Returns
        -------
        list[torch.nn.Parameter]
            Parameters w.r.t. which gradient norms are computed.
        """
        if self._resolved_params is not None:
            return self._resolved_params

        sp = self.shared_params
        if sp is None:
            params = [p for p in model.parameters() if p.requires_grad]
        elif callable(sp):
            params = list(sp(model))
        else:
            params = list(sp)

        params = [p for p in params if p.requires_grad]
        self._resolved_params = params
        return params

    def update(self, losses, model):
        """Perform one GradNorm update and return the new weights.

        Parameters
        ----------
        losses : dict[str, torch.Tensor]
            Mapping from task name to (scalar) loss tensor.  The tensors
            must be differentiable w.r.t. ``model`` parameters.
        model : torch.nn.Module
            Model whose (shared) parameters are used to compute the
            per-task gradient norms.

        Returns
        -------
        dict[str, torch.Tensor]
            Detached weight for each task.
        """
        # Reference device/dtype from the first loss.
        ref = losses[self.names[0]]
        if self._weights is None:
            self._init_weights(ref.device, ref.dtype)

        params = self._resolve_shared_params(model)

        # Record baseline losses on the first update (for training rates).
        if self.initial_losses is None:
            self.initial_losses = {
                name: losses[name].detach().clamp_min(1e-12)
                for name in self.names
            }

        eps = 1e-12

        # Per-task gradient norms of the (unweighted) losses.  These are
        # detached: only the scalar weights are optimised by GradNorm.
        base_norms = []
        for name in self.names:
            loss = losses[name]
            grads = torch.autograd.grad(
                loss,
                params,
                retain_graph=True,
                allow_unused=True,
            )
            terms = [g.detach() for g in grads if g is not None]
            if terms:
                norm = torch.sqrt(
                    sum(g.pow(2).sum() for g in terms) + eps
                )
            else:
                norm = loss.detach().new_tensor(0.0)
            base_norms.append(norm.detach())

        # Weighted gradient norms (differentiable w.r.t. the weights).
        G = torch.stack(
            [self._weights[name] * base_norms[i]
             for i, name in enumerate(self.names)]
        )

        # Relative training rates (treated as constants).
        tilde = torch.stack(
            [
                losses[name].detach()
                / self.initial_losses[name].clamp_min(eps)
                for name in self.names
            ]
        )
        r = tilde / (tilde.mean().clamp_min(eps))

        # Target gradient norms (constant w.r.t. the weights).
        G_bar = G.detach().mean()
        target = (G_bar * (r.clamp_min(eps) ** self.alpha)).detach()

        # GradNorm loss: L1 distance between actual and target norms.
        gradnorm_loss = torch.abs(G - target).sum()

        self._optimizer.zero_grad(set_to_none=True)
        gradnorm_loss.backward()
        self._optimizer.step()

        # Clamp and renormalise so that the weights sum to the number of
        # tasks (preserves the overall scale of the total loss).
        with torch.no_grad():
            for name in self.names:
                self._weights[name].data.clamp_(
                    self.min_weight, self.max_weight
                )
            total = sum(
                self._weights[name].data for name in self.names
            ).clamp_min(eps)
            scale = len(self.names) / total
            for name in self.names:
                self._weights[name].data.mul_(scale)

        return {
            name: self._weights[name].detach().clone()
            for name in self.names
        }
