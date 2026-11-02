"""Deep muPC MLP with DKP direct-feedback, both toggleable, in plain numpy.

Mirrors FullPCLayer/FullPCNetwork's actual math (muPC's `a` per Table 1,
DKP's one-hop Psi feedback into W, and Psi's own training rule) closely
enough to test depth-scaling hypotheses for `fl` cheaply, before touching
the real C++ engine.

Psi's own training (psi_optimizer) mirrors FullPCLayer::UpdateWeights()'s
Psi branch exactly: grad_Psi = z_{l+1}^T @ e_terminal (a local, Hebbian
rule -- this layer's own belief correlated with the final error, no
backprop needed to train Psi itself), via SGD with `fl` as its own
learning rate (the real engine's default, and what the original CIFAR-10
investigation actually used), or via AdamW sharing the same `lmbda` decay
as W -- the actual Kolen-Pollack ingredient: matched local updates and
matched decay on both directions let Psi and W converge toward mirroring
each other, without ever copying values.
"""
import numpy as np


def mupc_scales(d_in: int, hidden_sizes: list[int], terminal_size: int) -> list[float]:
    L = len(hidden_sizes) + 1
    H = L - 1
    N = max(hidden_sizes) if H > 0 else terminal_size
    scales = []
    for l in range(1, L + 1):
        if l == 1:
            scales.append(d_in ** -0.5)
        elif l == L:
            scales.append(1.0 / N)
        else:
            scales.append((N * L) ** -0.5)
    return scales


class DeepPCMLP:
    def __init__(self, d_in: int, hidden_sizes: list[int], terminal_size: int,
                 lr: float, ir: float, fl0: float, psi_mode: str,
                 use_mu_pc_init: bool = True, use_residual: bool = True, seed: int = 0,
                 fl_decay_steps: int = 0, psi_optimizer: str = "fixed", lmbda: float = 0.0):
        # fl_decay_steps>0 linearly anneals fl to 0 over that many steps,
        # as a bootstrap: use DKP's O(1) feedback early, phase it out
        # before its fixed, never-updated Psi has time to accumulate the
        # drift that eventually destabilizes training (see pc_mlp.py's
        # module docstring -- Psi is fixed, so its contribution never
        # naturally shrinks the way the real local-path error does).
        self.fl_decay_steps = fl_decay_steps
        self.psi_optimizer = psi_optimizer
        self.lmbda = lmbda
        self.sizes = [d_in] + hidden_sizes + [terminal_size]
        self.L = len(self.sizes) - 1
        self.terminal_size = terminal_size
        self.lr = lr
        self.ir = ir
        self.a = mupc_scales(d_in, hidden_sizes, terminal_size)
        # muPC's "1-skip": mu_l = phi(preact_l) + z_{l-1}, hidden-to-hidden
        # transitions only (2<=l<=L-1, 1-indexed), where shapes match.
        self.residual = [use_residual and 2 <= l <= self.L - 1 for l in range(1, self.L + 1)]

        rng = np.random.default_rng(seed)
        self.W, self.b, self.Psi = [], [], []
        self.mW, self.vW, self.mb, self.vb = [], [], [], []
        for l in range(self.L):
            fan_in, fan_out = self.sizes[l], self.sizes[l + 1]
            w_std = 1.0 if use_mu_pc_init else np.sqrt(2.0 / (fan_in + fan_out))
            self.W.append(rng.normal(0.0, w_std, (fan_out, fan_in)).astype(np.float32))
            self.b.append(np.zeros(fan_out, dtype=np.float32))
            psi_std = np.sqrt(2.0 / (fan_out + terminal_size))
            self.Psi.append(rng.normal(0.0, psi_std, (terminal_size, fan_out)).astype(np.float32))
            self.mW.append(np.zeros_like(self.W[-1]))
            self.vW.append(np.zeros_like(self.W[-1]))
            self.mb.append(np.zeros_like(self.b[-1]))
            self.vb.append(np.zeros_like(self.b[-1]))
        self.mPsi = [np.zeros_like(p) for p in self.Psi]
        self.vPsi = [np.zeros_like(p) for p in self.Psi]
        self.adam_t = 0
        self.beta1, self.beta2, self.adam_eps = 0.9, 0.999, 1e-8

        if psi_mode == "off":
            self.fl = [0.0] * self.L
        elif psi_mode == "fixed":
            self.fl = [fl0] * self.L
        elif psi_mode == "scale_by_a":
            self.fl = [fl0 * self.a[l] for l in range(self.L)]
        elif psi_mode == "scale_by_a2":
            self.fl = [fl0 * self.a[l] ** 2 for l in range(self.L)]
        else:
            raise ValueError(f"unknown psi_mode {psi_mode!r}")

    @staticmethod
    def _softmax(x):
        x = x - x.max(axis=-1, keepdims=True)
        ex = np.exp(x)
        return ex / ex.sum(axis=-1, keepdims=True)

    def _phi(self, x, is_output):
        return self._softmax(x) if is_output else np.maximum(x, 0.0)

    @staticmethod
    def _phi_deriv(preact, is_output):
        return np.ones_like(preact) if is_output else (preact > 0.0).astype(preact.dtype)

    def _mu(self, l, z_prev, preact):
        mu = self._phi(preact, l == self.L - 1)
        return mu + z_prev if self.residual[l] else mu

    def _forward(self, z0):
        preacts, zs = [], [z0]
        z = z0
        for l in range(self.L):
            preact = self.a[l] * (z @ self.W[l].T) + self.b[l]
            z = self._mu(l, z, preact)
            preacts.append(preact)
            zs.append(z)
        return zs, preacts

    def train_step(self, X, Y, steps: int):
        zs, preacts = self._forward(X)
        zs[-1] = Y.copy()

        for _ in range(steps):
            preacts = [self.a[l] * (zs[l] @ self.W[l].T) + self.b[l] for l in range(self.L)]
            errs = [zs[l + 1] - self._mu(l, zs[l], preacts[l]) for l in range(self.L)]

            for l in range(1, self.L):
                deriv_above = self._phi_deriv(preacts[l], l == self.L - 1)
                feedback = (errs[l] * deriv_above) @ self.W[l]
                direct = errs[l] if self.residual[l] else 0.0
                zs[l] = zs[l] + self.ir * (-errs[l - 1] + self.a[l] * feedback + direct)

        preacts = [self.a[l] * (zs[l] @ self.W[l].T) + self.b[l] for l in range(self.L)]
        errs = [zs[l + 1] - self._mu(l, zs[l], preacts[l]) for l in range(self.L)]
        e_terminal = errs[-1]
        batch = X.shape[0]
        self.adam_t += 1
        bc1 = 1.0 - self.beta1 ** self.adam_t
        bc2 = 1.0 - self.beta2 ** self.adam_t

        for l in range(self.L):
            is_output = l == self.L - 1
            deriv = self._phi_deriv(preacts[l], is_output)
            local_signal = errs[l] * deriv
            gW = self.a[l] * (local_signal.T @ zs[l]) / batch
            gb = local_signal.mean(axis=0)

            self.mW[l] = self.beta1 * self.mW[l] + (1 - self.beta1) * gW
            self.vW[l] = self.beta2 * self.vW[l] + (1 - self.beta2) * gW ** 2
            self.W[l] += self.lr * (self.mW[l] / bc1) / (np.sqrt(self.vW[l] / bc2) + self.adam_eps)
            if self.lmbda != 0.0:
                self.W[l] -= self.lr * self.lmbda * self.W[l]

            self.mb[l] = self.beta1 * self.mb[l] + (1 - self.beta1) * gb
            self.vb[l] = self.beta2 * self.vb[l] + (1 - self.beta2) * gb ** 2
            self.b[l] += self.lr * (self.mb[l] / bc1) / (np.sqrt(self.vb[l] / bc2) + self.adam_eps)

            if self.fl[l] != 0.0:
                decay = 1.0
                if self.fl_decay_steps > 0:
                    decay = max(0.0, 1.0 - self.adam_t / self.fl_decay_steps)
                if decay > 0.0:
                    pseudo_err = e_terminal @ self.Psi[l]
                    dkp_signal = pseudo_err * deriv
                    self.W[l] += decay * self.fl[l] * (dkp_signal.T @ zs[l]) / batch

                if self.psi_optimizer == "sgd":
                    gPsi = e_terminal.T @ zs[l + 1] / batch
                    self.Psi[l] += decay * self.fl[l] * gPsi
                elif self.psi_optimizer == "adamw":
                    gPsi = e_terminal.T @ zs[l + 1] / batch
                    self.mPsi[l] = self.beta1 * self.mPsi[l] + (1 - self.beta1) * gPsi
                    self.vPsi[l] = self.beta2 * self.vPsi[l] + (1 - self.beta2) * gPsi ** 2
                    psi_lr = decay * self.fl[l]
                    self.Psi[l] += psi_lr * (self.mPsi[l] / bc1) / (np.sqrt(self.vPsi[l] / bc2) + self.adam_eps)
                    if self.lmbda != 0.0:
                        self.Psi[l] -= psi_lr * self.lmbda * self.Psi[l]

        return float(np.mean(errs[-1] ** 2)), zs, preacts

    def predict(self, X):
        zs, _ = self._forward(X)
        return zs[-1]

    def is_finite(self) -> bool:
        return (all(np.isfinite(w).all() for w in self.W)
                and all(np.isfinite(b).all() for b in self.b)
                and all(np.isfinite(p).all() for p in self.Psi))
