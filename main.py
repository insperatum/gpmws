import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
import gpytorch

n_history = 50 #How many GP observations to store in memory, per example
order = 3 #Polynomial order

n_dataset = 100
data_xs = torch.Tensor([-4, -2, 0,2,4]).cuda() #Evaluate polynomial at these points
observation_noise = 1


d_phi = (order+1)*2

class R(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(len(data_xs),32)
        self.lin2 = nn.Linear(32, d_phi*2)
        self.lin2.weight.data.zero_()
        self.lin2.bias.data.zero_()

    def forward(self, x, phi=None):
        _phi = self.lin2(torch.relu(self.lin1(x)))
        phi_mu = _phi[:, :d_phi]
        phi_sigma = F.softplus(_phi[:, d_phi:])
        dist = Normal(phi_mu, phi_sigma)
        if phi is None: phi = dist.sample()
        score = dist.log_prob(phi)
        return phi, score

class Q(nn.Module):
    def __init__(self, phi):
        super().__init__()
        self.mu = phi[:, :(order+1)]
        self.sigma = F.softplus(phi[:, (order+1):])
        self.dist = Normal(self.mu, self.sigma)

    def forward(self, z = None):
        if z is None: z = self.dist.sample()
        score = self.dist.log_prob(z).sum(dim=1)
        return z, score


class P(nn.Module):
    def __init__(self, sigma=1):
        super().__init__()
        powers = torch.arange(order+1).float().cuda()
        self.sigma=sigma
        self.inputs = data_xs[None, :] ** powers[:, None]

    def forward(self, z, x=None):
        coeffs=z
        mu = (coeffs[:, :, None] * self.inputs[None, :, :]).sum(dim=1)
        dist = Normal(mu, self.sigma)
        if x is None: x = dist.sample()
        score = dist.log_prob(x).sum(dim=1)
        return x, score

mean_module = gpytorch.means.ConstantMean()
covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
likelihood = gpytorch.likelihoods.GaussianLikelihood()
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

r = R().cuda()
p = P(observation_noise).cuda()

#Make Data
coeffs = torch.randn(n_dataset, 4).cuda()
coeffs[0, :3]=0
coeffs[0, 3]=2
data, _ = P(observation_noise).cuda()(coeffs)

#Make memory
n_observations = torch.zeros(len(data)).cuda().long()
observation_inputs = torch.zeros(len(data), n_history, d_phi).cuda()
observation_outputs = torch.zeros(len(data), n_history).cuda()
best_phi = torch.zeros(len(data), d_phi).cuda()

def getBatch():
    batch_size=50
    i = torch.randperm(len(data))[:batch_size].cuda()
    return i, data[i]

optimizer = torch.optim.Adam(
        list(mean_module.parameters()) +
        list(covar_module.parameters()) +
        list(r.parameters()))
avg_score = None
for iteration in range(2000):
    optimizer.zero_grad()

    i, x = getBatch()
    phi, rscore = r(x)
    q = Q(phi)
    z, qz = q()
    _, p_score = p(z, x)
    score = p_score - qz

    obs_idxs = i*n_history + n_observations[i]%n_history
    inputs_unrolled = observation_inputs.view(len(data) * n_history, *observation_inputs.size()[2:])
    inputs_unrolled.index_copy_(dim=0, index=obs_idxs, source=phi)
    outputs_unrolled = observation_outputs.view(len(data) * n_history)
    outputs_unrolled[obs_idxs] = score
    n_observations[i] += 1

    min_n_observations = min(n_history, min(n_observations[i]))
    if min_n_observations>=0:
        train_x = observation_inputs[i, :min_n_observations] #b * n * d
        train_y = observation_outputs[i, :min_n_observations] #b * n
        model = ExactGPModel(train_x, train_y, likelihood).cuda()
        
        model.train()
        likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        output = model(train_x)
        gp_score = mll(output, train_y)
        loss = -gp_score.sum()
        loss.backward()

        model.eval()
        likelihood.eval()
        m = model(train_x).mean
        best_score, best_idx = torch.max(m, dim=1)
        best_phi[i] = inputs_unrolled[i*n_history + best_idx]
        _, rscore = r(x, best_phi[i])
        (-rscore).mean().backward()

        optimizer.step()

        avg_score = best_score.mean().item() if avg_score is None else 0.9*avg_score + 0.1*best_score.mean().item() 
        if iteration%10==0:
            print("\nIteration", iteration, "score", avg_score)
            print("True:", coeffs[0])
            print("Mean:", Q(best_phi).mu[0])
            print("Std:", Q(best_phi).sigma[0])
