import math
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

from collections import OrderedDict
from disentanglement.models.utils import reparametrize


class SSIDVAE(nn.Module):
    class Encoder(nn.Module):
        def __init__(self, num_channels=3, x_dim=32*32, hidden_dim=256, z_dim=10, u_dim=5): # changed
            super().__init__()
            self.conv_model = nn.Sequential(OrderedDict([
                ('conv_1', nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=4, stride=2, padding=1)), 
                ('relu_1', nn.ReLU()), 
                ('conv_2', nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)), 
                ('relu_2', nn.ReLU()), 
                ('conv_3', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)), 
                ('relu_3', nn.ReLU()), 
                ('conv_4', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)), 
                ('relu_4', nn.ReLU())   
            ]))
            self.linear_model = nn.Sequential(OrderedDict([
                ('line_1', nn.Linear(in_features=(64*4*4)+u_dim, out_features=hidden_dim, bias=True)), 
                ('relu_1', nn.ReLU())
            ]))
            self.mu = nn.Linear(in_features=hidden_dim, out_features=z_dim, bias=True)
            self.logvar = nn.Linear(in_features=hidden_dim, out_features=z_dim, bias=True)
        
        def forward(self, x, u):
            batch_size = x.size(0)
            out = self.conv_model(x)
            out = out.view((batch_size, -1))
            out = torch.cat((out, u), dim=1)
            out = self.linear_model(out)
            mu = self.mu(out)
            logvar = self.logvar(out)
            z = reparametrize(mu, logvar)
            return z, mu, logvar

    class Decoder(nn.Module):
        def __init__(self, num_channels=3, x_dim=32*32, hidden_dim=256, z_dim=10):
            super().__init__()
            self.linear_model = nn.Sequential(OrderedDict([
                ('line_1', nn.Linear(in_features=z_dim, out_features=hidden_dim, bias=True)), 
                ('relu_1', nn.ReLU()),  
                ('line_2', nn.Linear(in_features=hidden_dim, out_features=64*4*4, bias=True)), 
                ('relu_2', nn.ReLU())
            ]))
            self.convT_model = nn.Sequential(OrderedDict([
                ('convT_1', nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)), 
                ('relu_1', nn.ReLU()), 
                ('convT_2', nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)), 
                ('relu_2', nn.ReLU()), 
                ('convT_3', nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)), 
                ('relu_3', nn.ReLU()), 
                ('convT_4', nn.ConvTranspose2d(in_channels=32, out_channels=num_channels, kernel_size=4, stride=2, padding=1))
            ]))

        def forward(self, z):
            batch_size = z.size(0)
            out = self.linear_model(z)
            out = out.view(batch_size, 64, 4, 4)
            out = self.convT_model(out)
            return out
    
    class GroundTruthFactorLearner(nn.Module):
        def __init__(self, num_channels=3, x_dim=64*64, hidden_dim=256, u_dim=5):
            super().__init__()
            self.conv_model = nn.Sequential(OrderedDict([
                ('conv_1', nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=4, stride=2, padding=1)), 
                ('relu_1', nn.ReLU()), 
                ('conv_2', nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)), 
                ('relu_2', nn.ReLU()), 
                ('conv_3', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)), 
                ('relu_3', nn.ReLU()), 
                ('conv_4', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)), 
                ('relu_4', nn.ReLU())      
            ]))
            self.linear_model = nn.Sequential(OrderedDict([
                ('line_1', nn.Linear(in_features=(64*4*4), out_features=hidden_dim, bias=True)), 
                ('relu_1', nn.ReLU())
            ]))
            self.mu = nn.Linear(in_features=hidden_dim, out_features=u_dim, bias=True)
            self.logvar = nn.Linear(in_features=hidden_dim, out_features=u_dim, bias=True)
        
        def forward(self, x):
            batch_size = x.size(0)
            out = self.conv_model(x)
            out = out.view((batch_size, -1))
            out = self.linear_model(out)
            mu = self.mu(out)
            logvar = self.logvar(out)
            u = reparametrize(mu, logvar)
            return u, mu, logvar

    def __init__(self, num_channels, x_dim, hidden_dim, z_dim, u_dim):
        super().__init__()
        self.encoder = self.Encoder(num_channels, x_dim, hidden_dim, z_dim, u_dim)
        self.decoder = self.Decoder(num_channels, x_dim, hidden_dim, z_dim)
        self.groundtruthfactor_learner = self.GroundTruthFactorLearner(num_channels, x_dim, hidden_dim, u_dim)

    def encode(self, x, u):
        z, mu, logvar = self.encoder(x, u)
        return z, mu, logvar

    def decode(self, z):
        out = self.decoder(z)
        return out
    
    def groundtruthfactor_run(self, x):
        return self.groundtruthfactor_learner(x)

    def forward(self, x, u):
        z, mu, logvar = self.encode(x, u)
        z = reparametrize(mu, logvar)
        out = self.decode(z)
        return out, z, mu, logvar


class ConditionalPrior(nn.Module):
    class Encoder(nn.Module):
        def __init__(self, u_dim=10, hidden_dim=1000, z_dim=10):
            super().__init__()
            self.net = nn.Sequential(OrderedDict([
                ('linear_1', nn.Linear(in_features=u_dim, out_features=hidden_dim, bias=True)), 
                ('lrelu_1', nn.LeakyReLU(negative_slope=0.2, inplace=True)), 
                ('linear_2', nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)), 
                ('lrelu_2', nn.LeakyReLU(negative_slope=0.2, inplace=True)), 
                ('linear_3', nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)), 
                ('lrelu_3', nn.LeakyReLU(negative_slope=0.2, inplace=True))
                #('relu_3', nn.ReLU(inplace=True))
            ]))
            self.mu = nn.Linear(in_features=hidden_dim, out_features=z_dim, bias=True)
            self.logvar = nn.Linear(in_features=hidden_dim, out_features=z_dim, bias=True)

        def forward(self, u):
            output = self.net(u)
            mu = self.mu(output)
            logvar = self.logvar(output)
            z = reparametrize(mu, logvar)
            return z, mu, logvar

    class Decoder(nn.Module):
        def __init__(self, u_dim=10, hidden_dim=1000, z_dim=10):
            super().__init__()
            self.net = nn.Sequential(OrderedDict([
                ('linear_1', nn.Linear(in_features=z_dim, out_features=hidden_dim, bias=True)), 
                ('lrelu_1', nn.LeakyReLU(negative_slope=0.2, inplace=True)), 
                ('linear_2', nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)), 
                ('lrelu_2', nn.LeakyReLU(negative_slope=0.2, inplace=True)), 
                ('linear_3', nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)), 
                ('lrelu_3', nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                ('linear_4', nn.Linear(in_features=hidden_dim, out_features=u_dim, bias=True))
            ]))

        def forward(self, z):
            output = self.net(z)
            return output

    def __init__(self, u_dim, hidden_dim, z_dim):
        super().__init__()
        self.encoder = self.Encoder(u_dim, hidden_dim, z_dim)
        self.decoder = self.Decoder(u_dim, hidden_dim, z_dim)

    def encode(self, u):
        z, mu, logvar = self.encoder(u)
        return z, mu, logvar

    def decode(self, z):
        out = self.decoder(z)
        return out

    def forward(self, u):
        z, mu, logvar = self.encode(u)
        z = reparametrize(mu, logvar)
        out = self.decode(z)
        return out, z, mu, logvar


def _compute_labeled_loss(model, conditional_prior, x, u, beta, alpha, batch_size):
    # Forward pass
    x_, z, mu, logvar = model(x, u)
    _, _, c_mu, c_logvar = conditional_prior(u)
    u_, _, _ = model.groundtruthfactor_run(x)
    # Compute x reconstruction loss
    rec_loss = F.binary_cross_entropy_with_logits(
        x_.view(batch_size, -1), 
        x.view(batch_size, -1), 
        reduction='sum'
    ).div(batch_size) # 1-dimensional tensor
    # Compute u reconstruction loss, cross entropy
    u_rec_loss = F.binary_cross_entropy_with_logits(
        u_.view(batch_size, -1), 
        u.view(batch_size, -1), 
        reduction='sum'
    ).div(batch_size)
    # Compute kld loss
    kld = -0.5 * (1. + logvar - c_logvar  - (logvar.exp() + (mu - c_mu)**2)/c_logvar.exp())
    kld_loss = kld.sum(1).mean(0, True) # 1-dimensional tensor
    return rec_loss + beta * kld_loss + alpha * u_rec_loss

def _compute_unlabeled_loss(model, conditional_prior, x, beta, batch_size):
    # Estimate u from x
    u_, u_mu, u_logvar = model.groundtruthfactor_run(x)
    u_ = torch.sigmoid(u_)
    # Forward pass using u_ as input
    x_, z, mu, logvar = model(x, u_)
    _, _, c_mu, c_logvar = conditional_prior(u_)
    # Compute reconstruction loss
    rec_loss = F.binary_cross_entropy_with_logits(
        x_.view(batch_size, -1), 
        x.view(batch_size, -1), 
        reduction='sum'
    ).div(batch_size) # 1-dimensional tensor
    # Compute kld loss
    kld = -0.5 * (1. + logvar - c_logvar  - (logvar.exp() + (mu - c_mu)**2)/c_logvar.exp())
    kld_loss = kld.sum(1).mean(0, True) # 1-dimensional tensor
    # Compute entropy of q(u|x)
    entropy = .5 * u_logvar.sum(1).mean(0, True)
    return rec_loss + beta * kld_loss - entropy

def ssidvae_train(model, m_optimizer, conditional_prior, c_optimizer, data_iterator, u_idx, device, beta=1, gamma=1, alpha=.1, labeled_percentage=.01, training_steps=300000, print_every=1000):
    model = model.to(device=device)
    conditional_prior = conditional_prior.to(device=device)
    model.train()
    conditional_prior.train()
    m_loss_list = []
    l_loss_list = []
    u_loss_list = []
    c_loss_list = []
    iteration = 0
    done = False
    isnan = True    
    while not done:
        for i, (x, u) in enumerate(data_iterator): 
            batch_size = x.size(0)
            labeled_instances_within_batch = int(batch_size * labeled_percentage)+1
            unlabeled_instances_within_batch = batch_size - labeled_instances_within_batch
            l_x = x[:labeled_instances_within_batch,:].to(device=device).float()
            u_x = x[labeled_instances_within_batch:,:].to(device=device).float()
            l_u = u[:labeled_instances_within_batch, u_idx].to(device=device).float()
            u_u = u[labeled_instances_within_batch:, u_idx].to(device=device).float()
            # IVAE forward pass
            # 1. Compute labeled loss
            l_loss = _compute_labeled_loss(model, conditional_prior, l_x, l_u, beta, alpha, labeled_instances_within_batch)
            # 2. Compute unlabeled loss
            u_loss = _compute_unlabeled_loss(model, conditional_prior, u_x, beta, unlabeled_instances_within_batch)
            m_loss = l_loss + u_loss
            m_loss_list.append(m_loss.item())
            l_loss_list.append(l_loss.item())
            u_loss_list.append(u_loss.item())
            # Conditional prior pass
            u_, c_z, c_mu, c_logvar = conditional_prior(l_u)
            # Optimize the model parameters.
            m_optimizer.zero_grad()
            m_loss.backward(retain_graph=True)
            m_optimizer.step()
            # Compute the conditional prior reconstruction loss
            c_rec_loss = F.mse_loss(
                u_, 
                l_u, 
                reduction='sum'
            ).div(batch_size) # 1-dimensional tensor
            # Compute the conditional prior kl loss
            c_kld = -0.5 * (1. + c_logvar - c_mu**2 - c_logvar.exp())
            c_kld_loss = c_kld.sum(1).mean(0, True) # 1-dimensional tensor
            # Compute the full conditional prior loss
            c_loss = c_rec_loss + gamma * c_kld_loss # 1-dimensional tensor
            c_loss_list.append(c_loss.item())
            # Optimize the conditional_prior parameters.
            c_optimizer.zero_grad()
            c_loss.backward()
            c_optimizer.step()
            if iteration % print_every == 0:
                print("[Train] Iteration: {:5d}\t l_loss: {:.2f}\t u_loss: {:.2f}\t m_loss: {:.2f}\t c_loss: {:.2f}".format(iteration, l_loss.item(), u_loss.item(), m_loss.item(), c_loss.item()))
                if math.isnan(m_loss.item()):
                    isnan = True
                    break
            iteration += 1
            if iteration >= training_steps:
                done = True
                break
    return m_loss_list, l_loss_list, u_loss_list, c_loss_list