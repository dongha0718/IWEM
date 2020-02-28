#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np


################################################################################################
## Calculate log pdf of normal distribution
################################################################################################

def logp_z(z , z_mu , z_log_var):
    ## z : mini_batch * z_dim
    ## z_mu , z_log_var : mini_batch * z_dim
    
    logp = -z_log_var/2-torch.pow((z-z_mu) , 2)/(2*torch.exp(z_log_var))
    
    return logp.sum(1)  ## (mini_batch,) vector

################################################################################################
## Calculate log pdf of normal distribution
## z ~ N(0,1)
################################################################################################

def logp_z_std_mvn(z):
    ## z : mini_batch * z_dim
    ## z_mu , z_log_var : mini_batch * z_dim
    
    logp = -torch.pow(z , 2)/2
    
    return logp.sum(1)  ## (mini_batch,) vector

################################################################################################
## Calculate log p(x|z;\theta)
################################################################################################
   
def logp_x_given_z(x , x_mu):
    ## x : mini_batch * x_dim
    ## x_mu : mini_batch * x_dim
    eps = 1e-5
    logp = (x * torch.log(x_mu+eps) + (1. - x) * torch.log(1.-x_mu+eps))
    
    return logp


################################################################################################
## Calculate log pi
##
## pi = p(z|x;\theta)/q(z|x;\phi) \approx p(x,z;\theta)/p(z|x;\phi)
###################################################
## Note that the pi is not normalized version!!
## (We normalize the pi at log likelihood function with logp_x)
################################################################################################
def calculate_log_pi(x , z_mu , z_log_var , decoder , prop_dist):
    
    #####################
    ## Sample z from the proposal distribution
    sam_z , _ = prop_dist.Sample_Z(z_mu , z_log_var)
    
    ## calculate the denominator
    log_pi_den = logp_z(sam_z , z_mu , z_log_var)
    ## calculate the numerator 1
    log_pi_num1 = logp_z_std_mvn(sam_z)
    ## calculate the numerator 2
    x_mu = decoder(sam_z)
        
    log_pi_num2 = logp_x_given_z(x , x_mu).sum(1)

    ## calculate the log pi
    log_pi = (log_pi_num1 + log_pi_num2) - log_pi_den
    
    return log_pi.data , sam_z.data
    
################################################################################################
## Calculate E step
################################################################################################

def calculate_Estep(x , decoder , prop_dist , num_sam, num_aggr,add_var):
    ## calculate mu and log var using the proposal distribution
    x=x.repeat(num_aggr,1)
    num_iter = int(num_sam / num_aggr)
    z_mu , z_log_var = prop_dist(x)
    log2 = torch.log(torch.ones_like(z_log_var)*add_var)
    
    z_log_var = z_log_var + log2

    x_nll , z_nll , pi = [] , [] , []
    #comp_nll = []
    for h in range(num_iter):
        h_log_pi , h_sam_z = calculate_log_pi(x , z_mu , z_log_var , decoder , prop_dist)
        
        h_logp_z = logp_z_std_mvn(h_sam_z)
        h_x_mu = decoder(h_sam_z)
        
        h_logp_x_given_z = logp_x_given_z(x , h_x_mu).sum(1)
        
        ## reshape the terms
        h_log_pi = h_log_pi.reshape(num_aggr , -1).t()
        h_logp_z = h_logp_z.reshape(num_aggr , -1).t()
        h_logp_x_given_z = h_logp_x_given_z.reshape(num_aggr , -1).t()
                
        x_nll.append(-(h_logp_x_given_z))
        z_nll.append(-(h_logp_z))
        #comp_nll.append((-(h_logp_x_given_z+h_logp_z)).unsqueeze(1))
        pi.append((h_log_pi))

    x_nll , z_nll , pi = torch.cat(x_nll , 1) , torch.cat(z_nll , 1) , torch.cat(pi , 1)
    #comp_nll , pi = torch.cat(comp_nll , 1) , torch.cat(pi , 1)
    
    max_pi = torch.max(pi , 1 , keepdim=True)[0]    
    pi = torch.exp(pi - max_pi)
    pi = pi / pi.mean(1 , keepdim=True)
    
    #Estep = (comp_nll*pi).mean(1).mean()    
    Estep_x_vec = (x_nll*pi).mean(1)
    Estep_z_vec = (z_nll*pi).mean(1)
 
    #return Estep , std_log_pi
    return Estep_x_vec,Estep_z_vec


################################################################################################
## Calculate VAE loss function
################################################################################################
def VAE_loss(x , decoder , prop_dist, option , num_sam , num_aggr):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []
   
    x=x.repeat(num_aggr,1)
    z_mu , z_log_var = prop_dist(x)
        
    for h in range(num_iter):
        sam_z , _ = prop_dist.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z = decoder(sam_z)
            
        ######################################
        ## calculate elbo
        log_p = logp_z_std_mvn(sam_z)

        log_p_all = logp_x_given_z(x , x_mu_sam_z)
        log_recon = log_p_all.sum(1)
        log_q = logp_z(sam_z , z_mu , z_log_var)

        elbo = log_recon + (log_p - log_q)
            
        log_loss_list.append(elbo.reshape(num_aggr , -1).t())
    
    log_loss_list = torch.cat(log_loss_list , 1)
                    
    return -log_loss_list.mean()


################################################################################################
## Calculate IWAE loss function
################################################################################################
def IWAE_loss(x , decoder , prop_dist , option , num_sam , num_aggr,p_add_var):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    x=x.repeat(num_aggr,1)
    
    if option =='iwae':
        z_mu , z_log_var = prop_dist(x)
        
        for h in range(num_iter):
            sam_z , _ = prop_dist.Sample_Z(z_mu , z_log_var)            
            x_mu_sam_z = decoder(sam_z)
            
            ######################################
            ## calculate elbo
            log_p = logp_z_std_mvn(sam_z)

            log_p_all = logp_x_given_z(x , x_mu_sam_z)
            log_recon = (log_p_all).sum(1)
            log_q = logp_z(sam_z , z_mu , z_log_var)

            elbo = log_recon + (log_p - log_q)
            log_loss_list.append(elbo.reshape(num_aggr , -1).t())
        
    else:

        z_mu , z_log_var = prop_dist(x)
        log2 = torch.log(torch.ones_like(z_log_var)*p_add_var)

        z_log_var = z_log_var + log2

        for h in range(num_iter):
            sam_z , _ = prop_dist.Sample_Z(z_mu , z_log_var)

            ######################################
            ## calculate weight
            log_p = logp_z_std_mvn(sam_z)
            log_q = logp_z(sam_z , z_mu , z_log_var)

            ######################################
            ## calculate RE
            x_mu = decoder(sam_z)           
            log_recon = logp_x_given_z(x , x_mu)
            ######################################
            ## calculate Regularization Term
            log_reg = torch.log(torch.abs(log_recon.sum(1) + log_p))
            h_log_loss = log_reg + (log_p  + log_recon.sum(1) - log_q)
            log_loss_list.append(h_log_loss.reshape(num_aggr , -1).t())
            
    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)
            
    return -log_loss.mean()

################################################################################################
## Train functions
################################################################################################
def train(train_loader , decoder , prop_dist , optimizer_list , num_sam , num_aggr , option , \
          beta , use_cuda,binarization,add_var,p_add_var):
    for i,(i_data) in enumerate(train_loader):    
        #if (i+1)%100 is 0:               
            #print(".......... %d th mini-batch" %(i+1))

        if use_cuda:
            i_data  = i_data.cuda()

        if binarization:
            i_data = torch.bernoulli(i_data)

        if option == 'iwae':
            loss = IWAE_loss(i_data , decoder , prop_dist , option , num_sam , num_aggr,p_add_var)

            for optimizer in optimizer_list:
                optimizer.zero_grad()

            loss.backward()

            for optimizer in optimizer_list:
                optimizer.step()
                    
        elif option == 'iwem_woa':
            ## E-step and M-step
            estep_x_vec , estep_z_vec  = calculate_Estep(i_data , decoder , prop_dist , num_sam, num_aggr,add_var)
            eloss = (estep_x_vec + estep_z_vec).mean()
            optimizer_list[0].zero_grad()
            eloss.backward()
            optimizer_list[0].step()

            ## P-step
            ## Calculate E-step values
            ploss = IWAE_loss(i_data , decoder , prop_dist , option ,  num_sam , num_aggr,p_add_var)

            optimizer_list[1].zero_grad()
            ploss.backward()
            optimizer_list[1].step()

        elif 'iwem' in option:
            ## E-step and M-step
            estep_x_vec , estep_z_vec  = calculate_Estep(i_data , decoder , prop_dist , num_sam, num_aggr,add_var)
            eloss1 = (estep_x_vec + estep_z_vec).mean()
            if beta <1.:
                eloss2 = VAE_loss(i_data , decoder , prop_dist, option , 1,1)
                eloss = beta*eloss1 + (1-beta)*eloss2
            else:
                eloss = eloss1
            
            optimizer_list[0].zero_grad()
            eloss.backward()
            optimizer_list[0].step()

            ## P-step
            if option == 'iwem_woo':
                ploss = IWAE_loss(i_data , decoder , prop_dist , 'iwae' ,  num_sam , num_aggr,p_add_var)
            else:
                ploss = IWAE_loss(i_data , decoder , prop_dist , option ,  num_sam , num_aggr,p_add_var)

            optimizer_list[1].zero_grad()
            ploss.backward()
            optimizer_list[1].step()
        else:
            print('No option!')



################################################################################################
## Approximate p(x;\theta,\phi)
################################################################################################

def Approximate_logp_x(x , decoder , prop_dist , num_sam , num_aggr,add_var):    
  
    x = x.repeat(num_aggr,1)
    num_iter = int(num_sam/num_aggr)
    z_mu, z_logvar = prop_dist(x)    

    log2 = torch.log(torch.ones_like(z_logvar)*add_var)
    
    z_logvar = z_logvar + log2

    log_loss_vec = []
    for iter in range(num_iter):
        sample_z , _ = prop_dist.Sample_Z(z_mu , z_logvar)

        ######################################
        ## calculate weight
        log_p = logp_z_std_mvn(sample_z)
        log_q = logp_z(sample_z , z_mu, z_logvar)

        ######################################
        ## calculate RE
        x_mu = decoder(sample_z)

        min_epsilon = 1e-5
        max_epsilon = 1.-1e-5

        x_mu = torch.clamp(x_mu , min=min_epsilon , max=max_epsilon)
        log_recon = (x * torch.log(x_mu) + (1.-x) * torch.log( 1.-x_mu )).sum(1)

        log_loss = (log_recon + log_p - log_q).reshape(num_aggr,-1)
        
        
        log_loss_vec.append(log_loss.data.cpu().numpy())
            
    log_loss_vec = np.vstack(log_loss_vec).astype('float64')
    
    mean_loss = log_loss_vec.mean(axis=0)
    
    app_ll = np.log(np.sum(np.exp(log_loss_vec),0)/num_sam)
    
    return app_ll , np.mean(app_ll - mean_loss)

################################################################################################
## Approximated negative log likelihood
################################################################################################

def evaluate(decoder , prop_dist , test_loader , sample_size , num_aggr , use_cuda , binarization,add_var):
    
    #####################################################
    ## calculate approximated log likelihood
    #####################################################
    test_nll = []; KL = []
    
    for i,(i_data) in enumerate(test_loader):
        if ((i+1) % 50 == 0):
            print('... %d-th iteration...'%(i+1))
        #break
        if use_cuda:
            i_data = i_data.cuda()
        if binarization:
            i_data = torch.bernoulli(i_data)
        
        i_likelihood , i_KL = Approximate_logp_x(i_data , decoder , prop_dist , sample_size , num_aggr,add_var)        
        test_nll.append(-i_likelihood); KL.append(i_KL)
        
    test_nll = np.hstack(test_nll); KL = np.hstack(KL)

    print('The approximated nll is %f'%test_nll.mean())
    #print('The KL between True and Approximated posterior is %f'%KL.mean())

    return test_nll , KL
   

    
