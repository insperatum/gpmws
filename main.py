import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
#from torch.distributions.cauchy import Cauchy
#from torch.distributions.studentT import StudentT
import gpytorch

n_history = 50 #How many GP observations to store in memory, per example
order = 3 #Polynomial order

n_dataset = 100
batch_size=50
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
        if phi is None: phi = dist.rsample()
        score = dist.log_prob(phi)
        return phi, score

class Q(nn.Module):
    def __init__(self, phi):
        super().__init__()
        self.mu = phi[:, :(order+1)]
        self.sigma = F.softplus(phi[:, (order+1):])
        self.dist = Normal(self.mu, self.sigma)

    def forward(self, z = None):
        if z is None: z = self.dist.rsample()
        score = self.dist.log_prob(z).sum(dim=1)
        return z, score

class Pz(nn.Module):
    def __init__(self, sigma=1):
        super().__init__()
        self.sigma=sigma
        self.mu = torch.zeros(batch_size, order+1).cuda()
        self.sigma = torch.ones(batch_size, order+1).cuda()
        self.dist = Normal(self.mu, self.sigma)

    def forward(self, z=None):
        if z is None: z = self.dist.rsample()
        score = self.dist.log_prob(z).sum(dim=1)*0
        return z, score

class Px(nn.Module):
    def __init__(self, sigma=1):
        super().__init__()
        powers = torch.arange(order+1).float().cuda()
        self.sigma=sigma
        self.inputs = data_xs[None, :] ** powers[:, None]

    def forward(self, z, x=None):
        coeffs=z
        mu = (coeffs[:, :, None] * self.inputs[None, :, :]).sum(dim=1)
        dist = Normal(mu, self.sigma)
        if x is None: x = dist.rsample()
        score = dist.log_prob(x)
        score=score.sum(dim=1)
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
pz = Pz().cuda()
px = Px(observation_noise).cuda()

#Make Data
coeffs = torch.randn(n_dataset, 4).cuda()
coeffs[0, :3]=0
coeffs[0, 3]=2
data, _ = Px(observation_noise).cuda()(coeffs)

#Make memory
n_observations = torch.zeros(len(data)).cuda().long()
observation_inputs = torch.zeros(len(data), n_history, d_phi).cuda()
inputs_unrolled = observation_inputs.view(len(data) * n_history, *observation_inputs.size()[2:])
observation_outputs = torch.zeros(len(data), n_history).cuda()
outputs_unrolled = observation_outputs.view(len(data) * n_history)
best_phi = torch.zeros(len(data), d_phi).cuda()

def getBatch():
    i = torch.randperm(len(data))[:batch_size].cuda()
    return i, data[i]

optimizer = torch.optim.Adam([
    {'params':mean_module.parameters(), 'lr':1},
    {'params':covar_module.parameters(), 'lr':1},
    {'params':likelihood.parameters(), 'lr':0.1},
    {'params':r.parameters(), 'lr':0.001}])

avg_score = None
avg_score_regression = None
avg_best_score = None
for iteration in range(2000):
    optimizer.zero_grad()

    i, x = getBatch()
    phi, rscore = r(x)
    q = Q(phi)
    z, qz = q()
    _, pz_score = pz(z)
    _, px_score = px(z, x)
    score = pz_score+px_score - qz
    if torch.isnan(score).any():
        print("Got nan!")
        raise Exception()
    #    keep_idxs = 1-torch.isnan(score)
    #    i, x = i[keep_idxs], x[keep_idxs]
    #    phi, rscore = phi[keep_idxs], rscore[keep_idxs]
    #    z, qz = z[keep_idxs], qz[keep_idxs]
    #    p_score = p_score[keep_idxs]
    #    score = score[keep_idxs]


    #z_sleep, _ = pz()
    #x_sleep, _ = px(z)
    #phi_sleep, _ = r(x_sleep)
    #q_sleep = Q(phi_sleep)
    #_, sleep_score = q_sleep(z_sleep)
    wake_score = score
    #r_score, _ = torch.cat([wake_score[None], sleep_score[None]]).max(dim=0)
    r_score=wake_score
    #r_score = sleep_score
    (-r_score.mean()).backward(retain_graph=True)
    if any(torch.isnan(param.grad).any() for param in r.parameters()):
        print("NaN in grad, oops!")
        optimizer.zero_grad()


    #inputs_unrolled.index_copy_(dim=0, index=obs_idxs, source=phi.data)
    #outputs_unrolled[obs_idxs] = score.data
    obs_idxs = i*n_history + n_observations[i]%n_history
    inputs_unrolled.index_copy_(dim=0, index=obs_idxs, source=observation_inputs[i, 0])
    outputs_unrolled[obs_idxs] = observation_outputs[i, 0]

    observation_inputs[i, 0] = phi.data
    observation_outputs[i, 0] = score.data
    n_observations[i] += 1

    min_n_observations = min(n_history, min(n_observations[i]))

    if min_n_observations>=2:
        train_x = observation_inputs[i, :min_n_observations] #b * n * d
        train_y = observation_outputs[i, :min_n_observations] #b * n
        model = ExactGPModel(train_x, train_y, likelihood).cuda()
        
        model.train()
        likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        output = model(train_x)
        gp_score = mll(output, train_y)
        loss = -gp_score.sum()
        loss.backward(retain_graph=True)

        model.eval()
        likelihood.eval()
        m = model(train_x).mean
        best_score, best_idx = torch.max(m, dim=1)
        best_phi[i] = inputs_unrolled[i*n_history + best_idx] #This is what would train p
        cur_score   = m[:, 0]

        optimizer.step()

        avg_score = score.mean().item() if avg_score is None else 0.9*avg_score + 0.1*score.mean().item() 
        avg_score_regression = cur_score.mean().item() if avg_score_regression is None else 0.9*avg_score_regression + 0.1*cur_score.mean().item() 
        avg_best_score = best_score.mean().item() if avg_best_score is None else 0.9*avg_best_score + 0.1*best_score.mean().item() 
        if iteration%10==0:
            print("\nIteration", iteration, "score", avg_score, "regression", avg_score_regression, "best", avg_best_score)
            print("True:", coeffs[0])
            print("Mean:", Q(best_phi).mu[0])
            print("Std:", Q(best_phi).sigma[0])
            print("gp noise", likelihood.noise.item())
