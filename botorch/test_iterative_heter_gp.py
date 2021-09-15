import torch
import matplotlib.pyplot as plt
import warnings
import numpy as np

from iterative_heter_gp import IterativeHeteroskedasticSingleTaskGP

plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

warnings.filterwarnings('ignore')

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

size = 100
train_X = np.random.uniform(0, np.pi, size=size)
train_X = np.sort(train_X)
train_Y = np.random.normal(loc=(np.sin(2.5*train_X)*np.sin(1.5*train_X)),
                           scale=(0.01 + 0.25*(1-np.sin(2.5*train_X))**2),
                           size=size)

train_X = torch.tensor(train_X.reshape(-1, 1), dtype=torch.double)
train_Y = torch.tensor(train_Y.reshape(-1, 1), dtype=torch.double)

model = IterativeHeteroskedasticSingleTaskGP(
    train_X, train_Y, True)
model.fit()

scan_x = torch.linspace(0, np.pi, 500, dtype=torch.double).reshape(-1, 5, 1)

with torch.no_grad():
    scan_y = model.posterior(scan_x, observation_noise=False)
    plt.plot(scan_x.numpy().reshape(-1),
             scan_y.mean.reshape(-1), label='predict mean')

    lower, upper = scan_y.mvn.confidence_region()
    plt.fill_between(scan_x.numpy().reshape(-1),
                     lower.numpy().reshape(-1), upper.numpy().reshape(-1), alpha=0.2,
                     label='confidence region')

    scan_y_with_noise = model.posterior(scan_x, observation_noise=True)
    lower_with_noise, upper_with_noise = scan_y_with_noise.mvn.confidence_region()
    plt.fill_between(scan_x.numpy().reshape(-1), lower_with_noise.numpy().reshape(-1),
                     upper_with_noise.numpy().reshape(-1), alpha=0.2,
                     label='confidence region with noise')

    plt.scatter(train_X, train_Y, label='observed data')
    plt.plot(scan_x.reshape(-1), np.sin(2.5*scan_x.reshape(-1))*np.sin(
        1.5*scan_x.reshape(-1)), label='true mean')
    plt.plot(scan_x.reshape(-1), (0.01 + 0.25*(1-np.sin(2.5*scan_x.reshape(-1)))**2),
             label='true std')
    plt.plot(scan_x.reshape(-1), model.noise_model.posterior(scan_x).mean.sqrt().detach(
    ).reshape(-1), label='predict std')
    plt.legend()
    plt.title('fit use predict noise var' if model.fit_use_predict_noise_var else
              'fit use observed noise var')
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('gp_predict_noise_var.png' if model.fit_use_predict_noise_var else
                'gp_observed_noise_var.png')
