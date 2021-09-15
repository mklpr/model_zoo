from typing import Any, List, Optional, Union
from torch import Tensor
import torch

from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.distributions.multitask_multivariate_normal import MultitaskMultivariateNormal
from botorch import fit_gpytorch_model


MIN_INFERRED_NOISE_LEVEL = 1e-4


class IterativeHeteroskedasticSingleTaskGP(BatchedMultiOutputGPyTorchModel):
    def __init__(self, train_X: Tensor, train_Y: Tensor,
                 fit_use_predict_noise_var: bool = True,
                 max_iters: int = 10,
                 early_stop_eps: float = 1e-3) -> None:
        """
        fit_use_predict_noise_var: if True, use noise var predict from noise model as 
            train_Yvar to fit target model, else use observed noise var from observed data.
        """
        super().__init__()
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        self.base_model = SingleTaskGP(train_X, train_Y)
        self.train_X = train_X
        self.train_Y = train_Y
        self.fit_use_predict_noise_var = fit_use_predict_noise_var
        self.max_iters = max_iters
        self.early_stop_eps = early_stop_eps
        self.fitted_iters = 0
        self.is_fitted = False
        self._state_dict = {
            'base_model': None,
            'target_model': None,
            'noise_model': None
        }

    def state_dict(self) -> dict:
        return self._state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        self._state_dict = state_dict

    def fit(self) -> None:
        if self.fit_use_predict_noise_var:
            self._fit_use_predict_noise()
        else:
            self._fit_use_observed_noise()
        self.is_fitted = True

    def _fit_use_observed_noise(self) -> None:
        train_X, train_Y = self.train_X, self.train_Y
        mll_base_model = ExactMarginalLogLikelihood(
            self.base_model.likelihood, self.base_model)
        if self._state_dict['base_model'] is not None:
            self.base_model.load_state_dict(self._state_dict['base_model'])
        fit_gpytorch_model(mll_base_model)
        self._state_dict['base_model'] = self.base_model.state_dict()

        with torch.no_grad():
            observed_var = torch.pow(
                self.base_model.posterior(train_X).mean - train_Y, 2)

        for i in range(self.max_iters):
            self.target_model = FixedNoiseGP(
                train_X=train_X, train_Y=train_Y, train_Yvar=observed_var)
            mll_target_model = ExactMarginalLogLikelihood(self.target_model.likelihood,
                                                          self.target_model)
            if self._state_dict['target_model'] is not None:
                self.target_model.load_state_dict(
                    self._state_dict['target_model'])
            fit_gpytorch_model(mll_target_model)
            self._state_dict['target_model'] = self.target_model.state_dict()

            with torch.no_grad():
                observed_var_new = torch.pow(
                    self.target_model.posterior(train_X).mean - train_Y, 2)

            self.noise_model = SingleTaskGP(
                train_X=train_X, train_Y=observed_var_new)
            mll_noise_model = ExactMarginalLogLikelihood(self.noise_model.likelihood,
                                                         self.noise_model)
            if self._state_dict['noise_model'] is not None:
                self.noise_model.load_state_dict(
                    self._state_dict['noise_model'])
            fit_gpytorch_model(mll_noise_model)
            self._state_dict['noise_model'] = self.noise_model.state_dict()

            self.fitted_iters = i + 1

            max_diff = (observed_var_new - observed_var).abs().max().item()
            if max_diff < self.early_stop_eps:
                break
            observed_var = observed_var_new

    def _fit_use_predict_noise(self) -> None:
        train_X, train_Y = self.train_X, self.train_Y
        mll_base_model = ExactMarginalLogLikelihood(
            self.base_model.likelihood, self.base_model)
        if self._state_dict['base_model'] is not None:
            self.base_model.load_state_dict(self._state_dict['base_model'])
        fit_gpytorch_model(mll_base_model)
        self._state_dict['base_model'] = self.base_model.state_dict()

        with torch.no_grad():
            observed_var = torch.pow(
                self.base_model.posterior(train_X).mean - train_Y, 2)

        self.noise_model = SingleTaskGP(
            train_X=train_X, train_Y=observed_var)
        mll_noise_model = ExactMarginalLogLikelihood(self.noise_model.likelihood,
                                                     self.noise_model)
        if self._state_dict['noise_model'] is not None:
            self.noise_model.load_state_dict(
                self._state_dict['noise_model'])
        fit_gpytorch_model(mll_noise_model)
        self._state_dict['noise_model'] = self.noise_model.state_dict()

        with torch.no_grad():
            predict_var = self.noise_model.posterior(train_X).mean.detach().max(
                torch.tensor(MIN_INFERRED_NOISE_LEVEL))

        for i in range(self.max_iters):
            self.target_model = FixedNoiseGP(
                train_X=train_X, train_Y=train_Y, train_Yvar=predict_var)
            mll_target_model = ExactMarginalLogLikelihood(self.target_model.likelihood,
                                                          self.target_model)
            if self._state_dict['target_model'] is not None:
                self.target_model.load_state_dict(
                    self._state_dict['target_model'])
            fit_gpytorch_model(mll_target_model)
            self._state_dict['target_model'] = self.target_model.state_dict()

            with torch.no_grad():
                observed_var = torch.pow(
                    self.target_model.posterior(train_X).mean - train_Y, 2)

            self.noise_model = SingleTaskGP(
                train_X=train_X, train_Y=observed_var)
            mll_noise_model = ExactMarginalLogLikelihood(self.noise_model.likelihood,
                                                         self.noise_model)
            if self._state_dict['noise_model'] is not None:
                self.noise_model.load_state_dict(
                    self._state_dict['noise_model'])
            fit_gpytorch_model(mll_noise_model)
            self._state_dict['noise_model'] = self.noise_model.state_dict()

            with torch.no_grad():
                predict_var_new = self.noise_model.posterior(train_X).mean.detach().max(
                    torch.tensor(MIN_INFERRED_NOISE_LEVEL))

            self.fitted_iters = i + 1

            max_diff = (predict_var_new - predict_var).abs().max().item()
            if max_diff < self.early_stop_eps:
                break
            predict_var = predict_var_new

    def posterior(self,
                  X: Tensor,
                  output_indices: Optional[List[int]] = None,
                  observation_noise: Union[bool, Tensor] = False,
                  **kwargs: Any) -> GPyTorchPosterior:
        if not self.is_fitted:
            return self.base_model.posterior(X,
                                             output_indices,
                                             observation_noise,
                                             **kwargs)
        else:
            if not observation_noise:
                return self.target_model.posterior(X,
                                                   output_indices,
                                                   **kwargs)
            else:
                target_mvn = self.target_model.posterior(X,
                                                         output_indices,
                                                         **kwargs).mvn
                noise_mvn = self.noise_model.posterior(X,
                                                       output_indices,
                                                       **kwargs).mvn
                target_mean = target_mvn.mean
                target_covar = target_mvn.covariance_matrix
                noise_covar = torch.diag_embed(noise_mvn.mean.reshape(
                    target_covar.shape[:-1]).max(
                    torch.tensor(MIN_INFERRED_NOISE_LEVEL)))
                if self._num_outputs > 1:
                    mvn = MultitaskMultivariateNormal(
                        target_mean, target_covar + noise_covar)
                else:
                    mvn = MultivariateNormal(
                        target_mean, target_covar + noise_covar)

                return GPyTorchPosterior(mvn=mvn)
