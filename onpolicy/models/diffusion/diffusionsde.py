from typing import Optional, Union
import torch
from cleandiffuser.diffusion.diffusionsde import DiscreteDiffusionSDE as OrigDiffuser, xtheta_to_epstheta, epstheta_to_xtheta
from cleandiffuser.utils import (
    TensorDict,
    at_least_ndim,
    concat_zeros,
    dict_apply,
    get_noise_scheduler,
    get_sampling_scheduler,
)

SUPPORTED_SOLVERS = [
    "ddpm",
    "ddim",
    "ode_dpmsolver_1",
    "ode_dpmsolver++_1",
    "ode_dpmsolver++_2M",
    "sde_dpmsolver_1",
    "sde_dpmsolver++_1",
    "sde_dpmsolver++_2M",
]

class DiscreteDiffusionSDE(OrigDiffuser):
    def sample(
        self,
        # ---------- the known fixed portion ---------- #
        prior: torch.Tensor,
        fix_mask: torch.Tensor,
        # ----------------- sampling ----------------- #
        solver: str = "ddpm",
        sample_steps: int = 5,
        sampling_schedule: str = "linear",
        sampling_schedule_params: Optional[dict] = None,
        use_ema: bool = True,
        temperature: float = 1.0,
        # ------------------ guidance ------------------ #
        condition_cfg=None,
        mask_cfg=None,
        w_cfg: float = 0.0,
        condition_cg=None,
        w_cg: float = 0.0,
        # ----------- Diffusion-X sampling ----------
        diffusion_x_sampling_steps: int = 0,
        # ----------- Warm-Starting -----------
        warm_start_reference: Optional[torch.Tensor] = None,
        warm_start_forward_level: float = 0.3,
        # ------------------ others ------------------ #
        requires_grad: bool = False,
        preserve_history: bool = False,
        **kwargs,
    ):
        """Sampling.

        Inputs:
        - prior: torch.Tensor
            The known fixed portion of the input data. Should be in the shape of generated data.
            Use `torch.zeros((n_samples, *x_shape))` for non-prior sampling.

        - solver: str
            The solver for the reverse process. Check `supported_solvers` property for available solvers.
        - n_samples: int
            The number of samples to generate.
        - sample_steps: int
            The number of sampling steps. Should be greater than 1 and less than or equal to the number of diffusion steps.
        - sample_step_schedule: Union[str, Callable]
            The schedule for the sampling steps.
        - use_ema: bool
            Whether to use the exponential moving average model.
        - temperature: float
            The temperature for sampling.

        - condition_cfg: Optional
            Condition for Classifier-free-guidance.
        - mask_cfg: Optional
            Mask for Classifier-guidance.
        - w_cfg: float
            Weight for Classifier-free-guidance.
        - condition_cg: Optional
            Condition for Classifier-guidance.
        - w_cg: float
            Weight for Classifier-guidance.

        - diffusion_x_sampling_steps: int
            The number of diffusion steps for diffusion-x sampling.

        - warm_start_reference: Optional[torch.Tensor]
            Reference data for warm-starting sampling. `None` indicates no warm-starting.
        - warm_start_forward_level: float
            The forward noise level to perturb the reference data. Should be in the range of `[0., 1.]`, where `1` indicates pure noise.

        - requires_grad: bool
            Whether to preserve gradients.
        - preserve_history: bool
            Whether to preserve the sampling history.

        Outputs:
        - x0: torch.Tensor
            Generated samples. Be in the shape of `(n_samples, *x_shape)`.
        - log: dict
            The log dictionary.
        """
        assert solver in SUPPORTED_SOLVERS, f"Solver {solver} is not supported."

        # ===================== Initialization =====================
        n_samples = prior.shape[0]
        log = {"sample_history": []}

        model = self.model if not use_ema else self.model_ema

        if fix_mask == None:
            fix_mask = self.fix_mask

        sampling_schedule_params = sampling_schedule_params or {}
        sampling_schedule_params["T"] = self.diffusion_steps
        sampling_schedule_params["noise_scheduler"] = self.noise_scheduler

        prior = prior.to(self.device)
        if isinstance(warm_start_reference, torch.Tensor) and 0 < warm_start_forward_level < 1:
            warm_start_reference = warm_start_reference.to(self.device)
            diffusion_steps = int(warm_start_forward_level * self.diffusion_steps)
            fwd_alpha, fwd_sigma = self.alpha[diffusion_steps], self.sigma[diffusion_steps]
            xt = warm_start_reference * fwd_alpha + fwd_sigma * torch.randn_like(
                warm_start_reference
            )
            sampling_schedule_params["t_max"] = warm_start_forward_level
        else:
            xt = torch.randn_like(prior) * temperature
        xt = xt * (1.0 - fix_mask) + prior * fix_mask

        if preserve_history:
            log["sample_history"].append(xt.cpu().numpy())

        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = (
                model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            )
            condition_vec_cg = condition_cg

        sampling_scheduler = get_sampling_scheduler(sampling_schedule, **sampling_schedule_params)
        t_schedule = sampling_scheduler(
            sample_steps, device=self.device, **sampling_schedule_params
        )
        t_schedule[1:] = t_schedule[1:].clamp(1, None)

        alphas = self.alpha[t_schedule.long()]
        sigmas = self.sigma[t_schedule.long()]
        logSNRs = self.logSNR[t_schedule.long()]
        hs = torch.zeros_like(logSNRs)
        hs[1:] = (
            logSNRs[:-1] - logSNRs[1:]
        )  # hs[0] is not correctly calculated, but it will not be used.
        stds = torch.zeros((sample_steps + 1,), device=self.device)
        stds[1:] = sigmas[:-1] / sigmas[1:] * (1 - (alphas[1:] / alphas[:-1]) ** 2).sqrt()

        buffer = []

        # ===================== Denoising Loop ========================
        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))
        for i in reversed(loop_steps):
            t = torch.full((n_samples,), t_schedule[i], dtype=prior.dtype, device=prior.device)

            # guided sampling
            pred, logp = self.guided_sampling(
                xt,
                t,
                alphas[i],
                sigmas[i],
                model,
                condition_vec_cfg,
                w_cfg,
                condition_vec_cg,
                w_cg,
                requires_grad,
            )

            # clip the prediction
            pred = self.clip_prediction(pred, xt, alphas[i], sigmas[i])

            # noise & data prediction
            eps_theta = (
                pred if self.predict_noise else xtheta_to_epstheta(xt, alphas[i], sigmas[i], pred)
            )
            x_theta = (
                pred
                if not self.predict_noise
                else epstheta_to_xtheta(xt, alphas[i], sigmas[i], pred)
            )

            # one-step update
            if solver == "ddpm":
                xt = (alphas[i - 1] / alphas[i]) * (xt - sigmas[i] * eps_theta) + (
                    sigmas[i - 1] ** 2 - stds[i] ** 2 + 1e-8
                ).sqrt() * eps_theta
                if i > 1:
                    xt += stds[i] * torch.randn_like(xt)

            elif solver == "ddim":
                xt = (
                    alphas[i - 1] * ((xt - sigmas[i] * eps_theta) / alphas[i])
                    + sigmas[i - 1] * eps_theta
                )

            elif solver == "ode_dpmsolver_1":
                xt = (alphas[i - 1] / alphas[i]) * xt - sigmas[i - 1] * torch.expm1(
                    hs[i]
                ) * eps_theta

            elif solver == "ode_dpmsolver++_1":
                xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(
                    -hs[i]
                ) * x_theta

            elif solver == "ode_dpmsolver++_2M":
                buffer.append(x_theta)
                if i < sample_steps:
                    r = hs[i + 1] / hs[i]
                    D = (1 + 0.5 / r) * buffer[-1] - 0.5 / r * buffer[-2]
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * D
                else:
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(
                        -hs[i]
                    ) * x_theta

            elif solver == "sde_dpmsolver_1":
                xt = (
                    (alphas[i - 1] / alphas[i]) * xt
                    - 2 * sigmas[i - 1] * torch.expm1(hs[i]) * eps_theta
                    + sigmas[i - 1] * torch.expm1(2 * hs[i]).sqrt() * torch.randn_like(xt)
                )

            elif solver == "sde_dpmsolver++_1":
                xt = (
                    (sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt
                    - alphas[i - 1] * torch.expm1(-2 * hs[i]) * x_theta
                    + sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt)
                )

            elif solver == "sde_dpmsolver++_2M":
                buffer.append(x_theta)
                if i < sample_steps:
                    r = hs[i + 1] / hs[i]
                    D = (1 + 0.5 / r) * buffer[-1] - 0.5 / r * buffer[-2]
                    xt = (
                        (sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt
                        - alphas[i - 1] * torch.expm1(-2 * hs[i]) * D
                        + sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt)
                    )
                else:
                    xt = (
                        (sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt
                        - alphas[i - 1] * torch.expm1(-2 * hs[i]) * x_theta
                        + sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt)
                    )

            # fix the known portion, and preserve the sampling history
            if fix_mask is not None:
                xt = xt * (1.0 - fix_mask) + prior * fix_mask
            if preserve_history:
                log["sample_history"].append(xt.cpu().numpy())

        # ================= Post-processing =================
        if self.classifier is not None:
            with torch.no_grad():
                t = torch.zeros((n_samples,), dtype=prior.dtype, device=self.device)
                logp = self.classifier.logp(xt, t, condition_vec_cg)
            log["log_p"] = logp

        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        log["sampling_schedule"] = t_schedule
        log["alpha"] = alphas
        log["sigma"] = sigmas
        log["logSNR"] = logSNRs

        return xt, log

    def loss(self, x0: torch.Tensor, condition: Optional[Union[torch.Tensor, TensorDict]] = None, fix_mask: Optional[torch.Tensor] = None):
        xt, t, eps = self.add_noise(x0)

        if fix_mask == None:
            fix_mask = self.fix_mask

        condition = self.model["condition"](condition) if condition is not None else None

        if self.predict_noise:
            loss = (self.model["diffusion"](xt, t, condition) - eps) ** 2
        else:
            loss = (self.model["diffusion"](xt, t, condition) - x0) ** 2

        return (loss * self.loss_weight * (1 - fix_mask)).mean()