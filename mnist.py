import random
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
#from torch.distributions.cauchy import Cauchy
#from torch.distributions.studentT import StudentT
import gpytorch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-q", default="gp", choices=["r", "gp", "mem"])
parser.add_argument("-data", default="mnist", choices=["mnist", "omniglot"])
args = parser.parse_args()

n_history = 10 #How many GP observations to store in memory, per example
d_latent = 40 #Polynomial order

batch_size=100

d_phi = d_latent*2


if args.data == "omniglot":
    import scipy.io
    mat = scipy.io.loadmat("./chardata_omniglot.mat")
    d = np.concatenate([
        mat['data'].transpose(),
        mat['testdata'].transpose()])
    #d = d[:1000]
    np.random.seed(0)
    np.random.shuffle(d)
    data = (torch.Tensor(d)>0.5).reshape(-1,28,28).cuda()
elif args.data == "mnist":
    d = np.fromfile("./binarized_mnist_train.amat", dtype=np.int16).reshape(-1,28,28)
    np.random.seed(0)
    np.random.shuffle(d)
    #d = d[:1000]
    data = (torch.from_numpy(d)-48).byte().cuda()

class R(nn.Module):
    def __init__(self):
        super().__init__()
        #28x28
        self.conv1 = nn.Conv2d(1,32, 3)
        #26x26
        #13x13
        self.conv2 = nn.Conv2d(32,16, 3)
        #11x11
        #5x5
        self.fc3 = nn.Linear(400, 2*d_phi)

    def forward(self, x, phi=None):
        y = x[:, None, :, :].float()
        y = self.conv1(y)
        y = F.max_pool2d(F.relu(y), 2)
        y = self.conv2(y)
        y = F.max_pool2d(F.relu(y), 2)
        y = y.view(-1, 400)
        _phi = self.fc3(y)
        phi_mu = _phi[:, :d_phi]
        phi_sigma = F.softplus(_phi[:, d_phi:])
        dist = Normal(phi_mu, phi_sigma)
        if phi is None: phi = dist.rsample()
        score = dist.log_prob(phi)
        return phi, score

class Q(nn.Module):
    def __init__(self, phi):
        super().__init__()
        self.mu = phi[:, :d_latent]
        self.sigma = F.softplus(phi[:, d_latent:])
        self.dist = Normal(self.mu, self.sigma)

    def forward(self, z = None):
        if z is None: z = self.dist.rsample()
        score = self.dist.log_prob(z).sum(dim=1)
        return z, score

class Pz(nn.Module):
    def __init__(self, sigma=1):
        super().__init__()
        self.sigma=sigma
        self.mu = torch.zeros(batch_size, d_latent).cuda()
        self.sigma = torch.ones(batch_size, d_latent).cuda()
        self.dist = Normal(self.mu, self.sigma)

    def forward(self, z=None):
        if z is None: z = self.dist.rsample()
        score = self.dist.log_prob(z).sum(dim=1)
        return z, score

class Px(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(d_latent, 400)
        #32x5x5
        self.conv1 = nn.ConvTranspose2d(16, 32, 3, stride=2)
        #11x11
        self.conv2 = nn.ConvTranspose2d(32, 32, 3, stride=1)
        #13x13
        self.conv3 = nn.ConvTranspose2d(32, 32, 3, stride=2)
        #27x27
        self.conv4 = nn.ConvTranspose2d(32, 1, 3, stride=1)
        #29x29
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, z, x=None):
        y = z

        y = F.relu(self.fc1(y))
        y = y.view(-1, 16, 5, 5)
        y = F.relu(self.conv1(y))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = self.conv2(y)
        y = y[:, 0, 1:, 1:]
        dist = Bernoulli(logits = y)
        if x is None: x = dist.sample()
        score = dist.log_prob(x.float())
        score = score.sum(dim=2).sum(dim=1)
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
model = ExactGPModel(torch.zeros(100,1,d_phi), torch.zeros(100,1) , likelihood).cuda()
#model = ExactGPModel(None, None, likelihood).cuda()
model.eval()
likelihood.eval()

r = R().cuda()
pz = Pz().cuda()
px = Px().cuda()

#Make memory
n_observations = torch.zeros(len(data)).cuda().long()
observation_inputs = torch.zeros(len(data), n_history, d_phi).cuda()
inputs_unrolled = observation_inputs.view(len(data) * n_history, *observation_inputs.size()[2:])
observation_outputs = torch.zeros(len(data), n_history).cuda()
outputs_unrolled = observation_outputs.view(len(data) * n_history)
best_phi = torch.zeros(len(data), d_phi).cuda()

start_idx = 0
def getBatch():
    global start_idx
    #i = torch.randperm(len(data))[:batch_size].cuda()
    i = torch.arange(start_idx, start_idx+batch_size).cuda()
    d = data[i]

    start_idx += batch_size
    if start_idx+batch_size>=len(data): start_idx=0
    return i, d

def getGPModel(i):
    return model, train_x, train_y

def getProposalDistribution(i, x):
    if random.random()<0.5 or any(n_observations[i]==0):
        # Propose using recognition model
        phi, _ = r(x)
        return phi
    else:
        # Local proposal
        model, train_x, train_y = getGPModel(i)
        model.eval()
        likelihood.eval()
        v = Variable(train_x, requires_grad=True)
        m = model(v).mean
        best_score, best_idx = torch.max(m, dim=1)
        v_unrolled = v.data.view(-1, *v.size()[2:])
        best_phi = v_unrolled[torch.arange(0, len(v_unrolled), v.size(1)).cuda() + best_idx]
        #m.sum().backward()
        #grad_unrolled = v.grad.view(-1, *v.size()[2:])
        #best_grad = grad_unrolled[torch.arange(0, len(v_unrolled), v.size(1)).cuda() + best_idx]
        lr = 0.01#Variable(torch.Tensor([0.001]).cuda(), requires_grad=True)
        #step = best_grad*lr
        mu = best_phi# + step
        sigma = lr#step.abs()
        dist = Normal(mu, sigma)
        return dist.rsample() 


gp_optimizer = torch.optim.Adam([
    {'params':mean_module.parameters(), 'lr':1},
    {'params':covar_module.parameters(), 'lr':1},
    {'params':likelihood.parameters(), 'lr':0.1}])

r_optimizer = torch.optim.Adam([
    {'params':r.parameters(), 'lr':0.001}])

p_optimizer = torch.optim.Adam([
    {'params':pz.parameters(), 'lr':0.01},
    {'params':px.parameters(), 'lr':0.01}])

avg_score = None
avg_true_score = None
avg_score_regression = None
avg_best_score = None
for iteration in range(100000):
    i, x = getBatch()

    # 
    #phi = getProposalDistribution(i, x)
    phi, _ = r(x)
    q = Q(phi)
    z, qz = q()
    _, pz_score = pz(z)
    _, px_score = px(z, x)
    score = pz_score+px_score - qz
    if torch.isnan(score).any(): raise Exception("Score is NaN!")

    #Sleep
    #z_sleep, _ = pz()
    #x_sleep, _ = px(z)
    #phi_sleep, _ = r(x_sleep)
    #q_sleep = Q(phi_sleep)
    #_, sleep_score = q_sleep(z_sleep)
    #r_score = sleep_score

    r_optimizer.zero_grad()
    p_optimizer.zero_grad()
    if score.requires_grad:
        (-score.mean()).backward()
        if any(torch.isnan(param.grad).any() for param in r.parameters()):
            print("NaN in grad, oops!")
        else: r_optimizer.step()
        if args.q=="r": p_optimizer.step()

    if args.q=="gp" or args.q=="mem":
        obs_idxs = i*n_history + n_observations[i]%n_history
        inputs_unrolled.index_copy_(dim=0, index=obs_idxs, source=observation_inputs[i, 0])
        outputs_unrolled[obs_idxs] = observation_outputs[i, 0]

        observation_inputs[i, 0] = phi.data
        observation_outputs[i, 0] = score.data
        n_observations[i] += 1


        min_n_observations = min(n_history, min(n_observations[i]))
        train_x = observation_inputs[i, :min_n_observations] #b * n * d
        train_y = observation_outputs[i, :min_n_observations] #b * n
        if args.q=="gp":
            if min_n_observations==1:
                model.set_train_data(None, None, strict=False)
            else:
                model.set_train_data(train_x[:, 1:], train_y[:, 1:], strict=False)
            gp_optimizer.zero_grad()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            output = model(train_x[:,:1])
            gp_score = mll(output, train_y[:, :1])
            loss = -gp_score.sum()
            loss.backward()
            gp_optimizer.step()

            model.set_train_data(train_x, train_y, strict=False)
            p_optimizer.zero_grad()
            model.eval()
            likelihood.eval()
            m = model(train_x).mean
            best_score, best_idx = torch.max(m, dim=1)
            best_phi[i] = inputs_unrolled[i*n_history + best_idx]
            q = Q(best_phi[i])
            z, qz = q()
            _, pz_score = pz(z)
            _, px_score = px(z, x)
            true_score = (pz_score.mean() + px_score.mean()) - qz.mean()
            (-true_score).backward()
            p_optimizer.step()


        if args.q=="mem":
            p_optimizer.zero_grad()
            best_score, best_idx = torch.max(train_y, dim=1)
            best_phi[i] = inputs_unrolled[i*n_history + best_idx]
            q = Q(best_phi[i])
            z, qz = q()
            _, pz_score = pz(z)
            _, px_score = px(z, x)
            true_score = (pz_score.mean() + px_score.mean()) - qz.mean()
            (-true_score).backward()
            p_optimizer.step()

        avg_best_score = best_score.mean().item() if avg_best_score is None else 0.95*avg_best_score + 0.05*best_score.mean().item() 
        avg_true_score = true_score.item() if avg_true_score is None else 0.95*avg_true_score + 0.05*true_score.item() 
        #cur_score   = m[:, 0]

    avg_score = score.mean().item() if avg_score is None else 0.95*avg_score + 0.05*score.mean().item() 
    #avg_score_regression = cur_score.mean().item() if avg_score_regression is None else 0.9*avg_score_regression + 0.1*cur_score.mean().item() 
    if iteration%10==0:
    #    print("\nIteration", iteration, "score", avg_score, "regression", avg_score_regression, "best of %d" % train_x.size(1), avg_best_score)
        if args.q == "r":
            print("\nIteration", iteration, "score:", avg_score)
        else:
            print("\nIteration", iteration, "score:", avg_score, "true:", avg_true_score, "best of %d:" % train_x.size(1), avg_best_score)

    if iteration%100==0:
        z, _ = pz()
        x, _ = px(z)
        print("\n".join("|" + "".join("##" if xx else "  " for xx in row) + "|" for row in x[0]))


