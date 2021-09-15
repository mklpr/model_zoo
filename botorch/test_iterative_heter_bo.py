from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound
import torch
import matplotlib.pyplot as plt
import warnings
import numpy as np
from tqdm import tqdm

from iterative_heter_gp import IterativeHeteroskedasticSingleTaskGP
from botorch.optim import optimize_acqf

plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

warnings.filterwarnings('ignore')


def fun(x):
    x = x * 3.14
    y = np.random.normal(loc=(np.sin(2.5*x)*np.sin(1.5*x)),
                         scale=(0.01 + 0.25*(1-np.sin(2.5*x))**2))
    return torch.from_numpy(y)


def generate_initial_data(n, fun):
    train_x = torch.rand(n, 1)
    train_obj = fun(train_x)
    return train_x, train_obj


def initialize_model(train_x, train_obj, state_dict=None):
    model = IterativeHeteroskedasticSingleTaskGP(train_x, train_obj)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model


BATCH_SIZE = 1
NUM_RESTARTS = 10
RAW_SAMPLES = 512


def optimize_acqf_and_get_observation(acq_func, fun):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=torch.tensor([[0.0], [1.0]]),
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values
    new_x = candidates.detach()
    new_obj = fun(new_x)
    return new_x, new_obj


N_BATCH = 20
MC_SAMPLES = 256

torch.manual_seed(0)

train_x, train_obj = generate_initial_data(1, fun)
model = initialize_model(train_x, train_obj)
qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)


for iteration in tqdm(range(N_BATCH)):
    model.fit()

    acq_typ = 1
    if acq_typ == 1:
        acq_func = UpperConfidenceBound(
            model=model,
            beta=2
        )
    elif acq_typ == 2:
        acq_func = ExpectedImprovement(
            model=model,
            best_f=train_obj.max(),
        )
    elif acq_typ == 3:
        acq_func = qExpectedImprovement(
            model=model,
            best_f=train_obj.max(),
            sampler=qmc_sampler
        )
    elif acq_typ == 4:
        acq_func = qNoisyExpectedImprovement(
            model=model,
            X_baseline=train_x,
            sampler=qmc_sampler
        )

    new_x, new_obj = optimize_acqf_and_get_observation(acq_func, fun)
    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])

    model = initialize_model(
        train_x,
        train_obj,
        model.state_dict()
    )

plt.plot(train_obj.numpy(), '-*')
plt.grid()
plt.xlabel('step')
plt.ylabel('y')
plt.title('object value with noise')
plt.savefig('bayesian_optimization.png')
