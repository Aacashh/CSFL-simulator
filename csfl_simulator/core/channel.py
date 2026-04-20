"""Massive MIMO channel noise simulation for Federated Distillation.

Implements the mMIMO uplink/downlink noise model from:
  Mu et al., "Federated Distillation in Massive MIMO Networks",
  IEEE TCCN, vol. 10, no. 4, Aug 2024.

ZF (zero-forcing) approach at the base station.  Noise variances are derived
from Eqs. 16, 22, 23d and Appendix A of the paper.
"""
from __future__ import annotations
import math
import torch


class MIMOChannel:
    """Simulates mMIMO uplink/downlink impairments on logit transmissions.

    Parameters:
        n_bs:      Number of base-station antennas  (N_BS, paper default 64)
        n_device:  Number of antennas per device     (N_D, paper default 1)
        ul_snr_db: Uplink   SNR in dB               (paper default -8)
        dl_snr_db: Downlink SNR in dB               (paper default -20)
        quantization_bits: Uniform quantization bits (paper default 8)
        combining: Combining / detection scheme at the BS. "zf" (zero-forcing,
            paper default) gives exact interference cancellation but amplifies noise
            as N_D/(2*SNR*N_BS). "mmse" (minimum-mean-squared-error) attenuates
            the same noise by a 1/(1 + 1/(SNR*N_BS)) factor (see paper §VI-C),
            becoming equivalent to ZF in the high-SNR regime.
    """

    def __init__(
        self,
        n_bs: int = 64,
        n_device: int = 1,
        ul_snr_db: float = -8.0,
        dl_snr_db: float = -20.0,
        quantization_bits: int = 8,
        combining: str = "zf",
    ):
        self.n_bs = n_bs
        self.n_device = n_device
        self.ul_snr_db = ul_snr_db
        self.dl_snr_db = dl_snr_db
        self.quantization_bits = quantization_bits
        self.combining = (combining or "zf").lower()
        if self.combining not in ("zf", "mmse"):
            raise ValueError(f"Unknown combining scheme: {combining!r}. Use 'zf' or 'mmse'.")

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------

    def quantize(self, logits: torch.Tensor) -> torch.Tensor:
        """Uniform mid-tread quantization to *quantization_bits* bits.

        Adds quantization error whose variance is approximately Delta^2 / 12
        where Delta = (max - min) / (2^bits - 1).
        """
        qmin = logits.min()
        qmax = logits.max()
        n_levels = 2 ** self.quantization_bits - 1
        if n_levels == 0 or (qmax - qmin).item() < 1e-12:
            return logits.clone()
        scale = (qmax - qmin) / n_levels
        return torch.round((logits - qmin) / scale) * scale + qmin

    # ------------------------------------------------------------------
    # Uplink noise  (Eq. 16)
    # ------------------------------------------------------------------

    def uplink_noise(self, logits: torch.Tensor) -> torch.Tensor:
        r"""Add ZF estimation noise for the uplink channel.

        From Eq. 16 of the paper, after ZF processing at the BS the
        estimated logits are:
            \hat{Z}_t = Z_t + \omega_{t,UL}
        where \omega_{t,UL} ~ N(0, \sigma_{UL}^2 I) with
            \sigma_{UL}^2 = N_D \sigma^2 / (2 P_{UL} N_{BS})

        We parameterise via SNR_{UL} = P_{UL} / \sigma^2, so:
            noise_var = N_D / (2 * SNR_UL_linear * N_BS)

        For MMSE combining, the receiver attenuates the noise by a
        1 / (1 + 1/(SNR_UL * N_BS)) factor, asymptotically matching ZF for
        SNR * N_BS -> inf (paper §VI-C, Fig. 7).
        """
        snr_lin = 10.0 ** (self.ul_snr_db / 10.0)
        noise_var = self.n_device / (2.0 * snr_lin * self.n_bs)
        if self.combining == "mmse":
            # MMSE attenuation factor: 1 / (1 + 1/(SNR*N_BS))
            mmse_att = 1.0 / (1.0 + 1.0 / max(snr_lin * self.n_bs, 1e-12))
            noise_var *= mmse_att
        noise = torch.randn_like(logits) * math.sqrt(max(noise_var, 0.0))
        return logits + noise

    # ------------------------------------------------------------------
    # Downlink noise  (Eq. 22)
    # ------------------------------------------------------------------

    def downlink_noise(self, logits: torch.Tensor) -> torch.Tensor:
        r"""Add ZF precoder noise for the downlink channel.

        From Eq. 22, the estimation error at the n-th user is approximately:
            \omega_{n,DL} ~ N(0, \hat{\sigma}_z^2 \sigma^2 / (2 P_{DL}) I)

        Using SNR_{DL} = P_{DL} / \sigma^2 and approximating \hat{\sigma}_z^2
        by the empirical variance of the aggregated logits:
            noise_var = var(logits) / (2 * SNR_DL_linear)
        """
        snr_lin = 10.0 ** (self.dl_snr_db / 10.0)
        logit_var = logits.var().item() + 1e-10
        noise_var = logit_var / (2.0 * snr_lin)
        noise = torch.randn_like(logits) * math.sqrt(max(noise_var, 0.0))
        return logits + noise

    # ------------------------------------------------------------------
    # Effective noise variance  (Eq. 23d / Appendix A)
    # ------------------------------------------------------------------

    def effective_noise_variance(self, n_selected: int, logit_var: float) -> float:
        r"""Compute \sigma_\omega^2 from the composite UL+DL mapping (Eq. 23d).

            \sigma_\omega^2 = \sigma_{UL}^2 / N + \sigma_{DL}^2

        where N is the number of selected (participating) clients.
        """
        snr_ul = 10.0 ** (self.ul_snr_db / 10.0)
        snr_dl = 10.0 ** (self.dl_snr_db / 10.0)
        sigma_ul_sq = self.n_device / (2.0 * snr_ul * self.n_bs)
        sigma_dl_sq = (logit_var + 1e-10) / (2.0 * snr_dl)
        return sigma_ul_sq / max(1, n_selected) + sigma_dl_sq
