#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


#=======================================================================================================================
# WEIGHTS INITS
#=======================================================================================================================
def xavier_init(m):
    s =  np.sqrt( 2. / (m.in_features + m.out_features) )
    m.weight.data.normal_(0, s)

#=======================================================================================================================
def he_init(m):
    s =  np.sqrt( 2. / m.in_features )
    m.weight.data.normal_(0, s)

################################################################################################
## Gated Dense structure
################################################################################################

class GatedDense(nn.Module):
    def __init__(self, input_size, output_size, activation=None):
        super(GatedDense, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.Linear(input_size, output_size)
        self.g = nn.Linear(input_size, output_size)

    def forward(self, x):
        h = self.h(x)
        if self.activation is not None:
            h = self.activation( self.h( x ) )

        g = self.sigmoid( self.g( x ) )

        return h * g

################################################################################################
## Nonlinear structure
################################################################################################

class NonLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation=None):
        super(NonLinear, self).__init__()

        self.activation = activation
        self.linear = nn.Linear(int(input_size), int(output_size), bias=bias)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation( h )

        return h
#=======================================================================================================================
class GatedConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, activation=None):
        super(GatedConv2d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)
        self.g = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation( self.h( x ) )

        g = self.sigmoid( self.g( x ) )

        return h * g

#=======================================================================================================================
class Conv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, activation=None, bias=True):
        super(Conv2d, self).__init__()

        self.activation = activation
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation, bias=bias)

    def forward(self, x):
        h = self.conv(x)
        if self.activation is None:
            out = h
        else:
            out = self.activation(h)

        return out



################################################################################################
## Encoder structure
################################################################################################
    
class Encoder(torch.nn.Module):
    def __init__(self, X_dim, h_dim, Z_dim , n_hidden=2):
        ## n_hidden must be larger than 0!
        super(Encoder, self).__init__()
        # encoder: q(z | x)
        self.q_z_layers = nn.Sequential(
            GatedConv2d(1, 32, 7, 1, 3),
            GatedConv2d(32, 32, 3, 2, 1),
            GatedConv2d(32, 64, 5, 1, 2),
            GatedConv2d(64, 64, 3, 2, 1),
            GatedConv2d(64, 6, 3, 1, 1)
        )
        
        self.q_z_mean = nn.Linear(h_dim, Z_dim)
        self.q_z_logvar = NonLinear(h_dim, Z_dim, activation=nn.Hardtanh(min_val=-6.,max_val=2.))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)
        
    def Sample_Z(self , z_mu , z_log_var):
        
        eps = torch.randn(z_log_var.shape).normal_().type(z_mu.type()) 
                
        sam_z = z_mu + (torch.exp(z_log_var / 2)) * eps    
        
        return sam_z , eps
    
    #def forward(self, x,option):
    def forward(self , x):
        x = x.view(-1, 1, 28, 28)
        h = self.q_z_layers(x)
        h = h.view(x.size(0),-1)

        z_q_mean = self.q_z_mean(h)
        z_q_logvar = self.q_z_logvar(h)

        return z_q_mean, z_q_logvar
               
################################################################################################
## Decoder structure
################################################################################################

class Decoder(torch.nn.Module):
    def __init__(self, Z_dim, h_dim, X_dim , n_hidden=2):
        ## n_hidden must be larger than 0!
        super(Decoder, self).__init__()
        # decoder: p(x | z)
        self.p_x_layers = nn.Sequential(
            GatedDense(Z_dim, 300),
            GatedDense(300, 784)
        )
        
        # decoder: p(x | z)
        act = nn.ReLU(True)
        # joint
        self.p_x_layers_joint = nn.Sequential(
            GatedConv2d(1, 64, 3, 1, 1),
            GatedConv2d(64, 64, 3, 1, 1),
            GatedConv2d(64, 64, 3, 1, 1),
            GatedConv2d(64, 64, 3, 1, 1),
        )
        self.p_x_mean = Conv2d(64, 1, 1, 1, 0, activation=nn.Sigmoid())
                
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)
        
    def forward(self, z):
        z = self.p_x_layers(z)

        z = z.view(-1, 1, 28, 28)
        z = self.p_x_layers_joint(z)
        
        
        x_mean = self.p_x_mean(z).view(-1,784)

        return x_mean
