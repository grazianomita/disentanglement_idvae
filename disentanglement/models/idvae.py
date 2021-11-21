import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

from collections import OrderedDict
from disentanglement.models.utils import reparametrize


class IDVAE(nn.Module):
    class Encoder(nn.Module):
        def __init__(self, num_channels=3, x_dim=32*32, hidden_dim=256, z_dim=10, u_dim=5):
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

    def __init__(self, num_channels, x_dim, hidden_dim, z_dim, u_dim):
        super().__init__()
        self.encoder = self.Encoder(num_channels, x_dim, hidden_dim, z_dim, u_dim)
        self.decoder = self.Decoder(num_channels, x_dim, hidden_dim, z_dim)

    def encode(self, x, u):
        z, mu, logvar = self.encoder(x, u)
        return z, mu, logvar

    def decode(self, z):
        out = self.decoder(z)
        return out

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


def idvae_train(model, m_optimizer, conditional_prior, c_optimizer, data_iterator, u_idx, device, beta=1, gamma=1, training_steps=100000, print_every=1000):
    model = model.to(device=device)
    conditional_prior = conditional_prior.to(device=device)
    model.train()
    conditional_prior.train()
    rec_loss_list = []
    kld_loss_list = []
    kld_dim_loss_list = []
    train_loss_list = []  
    c_rec_loss_list = []
    c_kld_loss_list = []
    c_loss_list = []
    iteration = 0
    done = False
    while not done:
        for i, (x, u) in enumerate(data_iterator):
            batch_size = x.size(0)
            x = x.to(device=device).float()
            u = u[:, u_idx].to(device=device).float()
            # IVAE forward pass
            x_, z, mu, logvar = model(x, u)
            # Conditional prior pass
            u_, c_z, c_mu, c_logvar = conditional_prior(u)
            # Compute the IVAE reconstruction loss
            rec_loss = F.binary_cross_entropy_with_logits(
                x_.view(batch_size, -1), 
                x.view(batch_size, -1), 
                reduction='sum'
            ).div(batch_size) # 1-dimensional tensor
            # Compute kld between p(z|x,u) and p(z|u)
            kld = -0.5 * (1. + logvar - c_logvar  - (logvar.exp() + (mu - c_mu)**2)/c_logvar.exp())
            kld_loss = kld.sum(1).mean(0, True) # 1-dimensional tensor
            kld_dim_loss = kld.mean(0)
            # Compute the Full IVAE loss
            m_loss = rec_loss + beta * kld_loss
            rec_loss_list.append(rec_loss.item())
            kld_loss_list.append(kld_loss.item())
            kld_dim_loss_list.append(kld_dim_loss.data)
            train_loss_list.append(m_loss.item())
            # Optimize the model parameters
            m_optimizer.zero_grad()
            m_loss.backward(retain_graph=True)
            m_optimizer.step()
            # Compute the conditional prior reconstruction loss
            c_rec_loss = F.mse_loss(
                u_, 
                u, 
                reduction='sum'
            ).div(batch_size) # 1-dimensional tensor
            # Compute the conditional prior kl loss
            c_kld = -0.5 * (1. + c_logvar - c_mu**2 - c_logvar.exp())
            c_kld_loss = c_kld.sum(1).mean(0, True) # 1-dimensional tensor
            c_kld_dim_loss = c_kld.mean(0)
            # Compute the full conditional prior loss
            c_loss = c_rec_loss + gamma * c_kld_loss # 1-dimensional tensor
            c_rec_loss_list.append(c_rec_loss.item())
            c_kld_loss_list.append(c_kld_loss.item())
            c_loss_list.append(c_loss.item())
            # Optimize the conditional_prior parameters.
            c_optimizer.zero_grad()
            c_loss.backward()
            c_optimizer.step()
            if iteration % print_every == 0:
                print("[Train] Iteration: {:5d}\nrec_loss: {:.2f}\t kl_loss: {:.2f}\t m_loss: {:.2f}\nc_rec_loss: {:.2f}\t c_kld_loss: {:.2f}\t c_loss: {:.2f}".format(iteration, rec_loss.item(), kld_loss.item(), m_loss.item(), c_rec_loss.item(), c_kld_loss.item(), c_loss.item()))
            iteration += 1
            if iteration >= training_steps:
                done = True
                break
    return train_loss_list, rec_loss_list, kld_loss_list, kld_dim_loss_list, c_loss_list, c_rec_loss_list, c_kld_loss_list