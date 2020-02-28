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
    def __init__(self, input_size, output_size, bias=True, activation=F.tanh):
        super(NonLinear, self).__init__()

        self.activation = activation
        self.linear = nn.Linear(int(input_size), int(output_size), bias=bias)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation( h )

        return h

################################################################################################
## Encoder structure
################################################################################################
    
class Encoder(torch.nn.Module):
    def __init__(self, X_dim, h_dim, Z_dim , n_hidden=2):
        ## n_hidden must be larger than 0!
        super(Encoder, self).__init__()
        modules = []
        modules.append(NonLinear(np.prod(X_dim), h_dim))
        if n_hidden >= 2:
            for i in range(n_hidden-1):
                modules.append(NonLinear(h_dim, h_dim))
        
        self.q_z_layers = nn.Sequential(*modules)
        self.q_z_mean = nn.Linear(h_dim, Z_dim)
        self.q_z_logvar = nn.Linear(h_dim, Z_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m)
        
    def Sample_Z(self , z_mu , z_log_var):
        
        eps = torch.randn(z_log_var.shape).normal_().type(z_mu.type()) 
                
        sam_z = z_mu + (torch.exp(z_log_var / 2)) * eps    
        
        return sam_z , eps
    
    #def forward(self, x,option):
    def forward(self , x):
        x = self.q_z_layers(x)

        z_q_mean = self.q_z_mean(x)
        z_q_logvar = self.q_z_logvar(x)

        return z_q_mean, z_q_logvar
               
################################################################################################
## Decoder structure
################################################################################################

class Decoder(torch.nn.Module):
    def __init__(self, Z_dim, h_dim, X_dim , n_hidden=2):
        ## n_hidden must be larger than 0!
        super(Decoder, self).__init__()
        modules = []
        modules.append(NonLinear(Z_dim, h_dim))
        if n_hidden >= 2:
            for i in range(n_hidden-1):
                modules.append(NonLinear(h_dim, h_dim))
        
        self.p_x_layers = nn.Sequential(*modules)
        self.p_x_mean = NonLinear(h_dim, np.prod(X_dim), activation=nn.Sigmoid())
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m)
        
    def forward(self, z):
        z = self.p_x_layers(z)

        x_mean = self.p_x_mean(z)
        return x_mean

