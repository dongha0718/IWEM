import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from  torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import scipy.io


import time
import os
import sys



import argparse

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# # # # # # # # # # #
# START EXPERIMENTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #


# Training settings
parser = argparse.ArgumentParser(description='MLE Experiment')
# arguments for optimization
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--gpu_num', type=int, default=1)
parser.add_argument('--dataname', type=str, default='bimnist')
parser.add_argument('--X_dim', type=int, default=784)
parser.add_argument('--Z_dim', type=int, default=40)
parser.add_argument('--h_dim', type=int, default=300)
parser.add_argument('--n_hidden_dec', type=int, default=2)
parser.add_argument('--n_hidden_prop', type=int, default=2)
parser.add_argument('--num_sam', type=int, default=50)
parser.add_argument('--num_aggr', type=int, default=10)
parser.add_argument('--option', type=str, default='iwem')
parser.add_argument('--alpha_inc', type=float, default=0.01)
parser.add_argument('--decoder_option', type=str, default='mlp')
parser.add_argument('--eval_option', type=str, default='valid')
parser.add_argument('--add_var', type=float, default=1.5)
parser.add_argument('--p_add_var', type=float, default=1.5)


args = parser.parse_args()

def load_data(dataname,binarization):
    if dataname == 'mnist':
        #####################################################
        ## Load MNIST dataset
        #####################################################
        os.chdir('Data/MNIST_data')
        tr_mb_size , val_mb_size , ts_mb_size = 100 , 100 , 100
        train_loader = torch.utils.data.DataLoader( datasets.MNIST('Data/MNIST_data', train=True, download=True,
                                                                   transform=transforms.Compose([
                                                                       transforms.ToTensor()
                                                                   ])),
                                                    batch_size=tr_mb_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader( datasets.MNIST('Data/MNIST_data', train=False,
                                                                  transform=transforms.Compose([transforms.ToTensor()
                                                                            ])),
                                                   batch_size=ts_mb_size, shuffle=True)

        # preparing data
        x_train = train_loader.dataset.train_data.float().numpy() / 255.
        x_train = np.reshape( x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2] ) )

        y_train = np.array( train_loader.dataset.train_labels.float().numpy(), dtype=int)

        x_test = test_loader.dataset.test_data.float().numpy() / 255.
        x_test = np.reshape( x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2] ) )

        y_test = np.array( test_loader.dataset.test_labels.float().numpy(), dtype=int)

        # validation set
        x_val = x_train[50000:60000]
        y_val = np.array(y_train[50000:60000], dtype=int)
        x_train = x_train[0:50000]
        y_train = np.array(y_train[0:50000], dtype=int)

        #binarization
        if binarization:
            np.random.seed(1234)
            x_val = np.random.binomial(1, x_val)
            x_test = np.random.binomial(1, x_test)
            x_val = x_val.astype(np.float32)
            x_test = x_test.astype(np.float32)

        #####################################################
        ## dataset to loader
        #####################################################

        #train = TensorDataset(torch.from_numpy(x_train))
        train_loader = DataLoader(x_train, batch_size=tr_mb_size, shuffle=True)
        train_valid_loader = DataLoader(x_train, batch_size=tr_mb_size, shuffle=False)

        #validation = TensorDataset(torch.from_numpy(x_val).float())
        valid_loader = DataLoader(x_val, batch_size=val_mb_size, shuffle=False)

        #test = TensorDataset(torch.from_numpy(x_test).float())
        test_loader = DataLoader(x_test, batch_size=ts_mb_size, shuffle=False)

        torch_train = torch.from_numpy(x_train)
    elif dataname == 'bimnist':
        #####################################################
        ## Load biMNIST dataset
        #####################################################

        os.chdir('Data/biMNIST_data')

        input_dim = 28 * 28
        mnist_train = np.loadtxt('binarized_mnist_train.amat',dtype = 'float32')
        mnist_valid = np.loadtxt('binarized_mnist_valid.amat',dtype = 'float32')
        mnist_test = np.loadtxt('binarized_mnist_test.amat',dtype = 'float32')

        #####################################################
        ## dataset to loader
        #####################################################

        tr_mb_size , val_mb_size , ts_mb_size = 100 , 100 , 100

        train_loader = torch.utils.data.DataLoader(mnist_train , batch_size=tr_mb_size, shuffle=True)
        train_valid_loader = torch.utils.data.DataLoader(mnist_train , batch_size=tr_mb_size, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(mnist_valid , batch_size=val_mb_size , shuffle = True)
        test_loader = torch.utils.data.DataLoader(mnist_test , batch_size=ts_mb_size , shuffle = False)

        torch_train = torch.from_numpy(mnist_train)
    elif dataname == 'omniglot':
        ###################################################
        ## Load Omniglot dataset
        ###################################################
        os.chdir('Data/omniglot_data')
        def reshape_data(data):
            return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='fortran')

        omni_raw = scipy.io.loadmat('chardata.mat')

        ####################################################
        ## split dataset
        # train and test data
        train_data = reshape_data(omni_raw['data'].T.astype('float32'))
        x_test = reshape_data(omni_raw['testdata'].T.astype('float32'))

        n_validation = 1345
        # shuffle train data
        np.random.shuffle(train_data)
        # set train and validation data
        x_train = train_data[:-n_validation]
        x_val = train_data[-n_validation:]

        # idle y's
        y_train = np.zeros( (x_train.shape[0], 1) )
        y_val = np.zeros( (x_val.shape[0], 1) )
        y_test = np.zeros( (x_test.shape[0], 1) )

        #binarization
        if binarization:
            np.random.seed(1234)
            x_val = np.random.binomial(1, x_val)
            x_test = np.random.binomial(1, x_test)
            x_val = x_val.astype(np.float32)
            x_test = x_test.astype(np.float32)

        #####################################################
        ## dataset to loader
        #####################################################
        tr_mb_size , val_mb_size , ts_mb_size = 100 , 100 , 100

        #train = TensorDataset(torch.from_numpy(x_train))
        train_loader = DataLoader(x_train, batch_size=tr_mb_size, shuffle=True)
        train_valid_loader = DataLoader(x_train, batch_size=tr_mb_size, shuffle=False)

        #validation = TensorDataset(torch.from_numpy(x_val).float())
        valid_loader = DataLoader(x_val, batch_size=val_mb_size, shuffle=False)


        #test = TensorDataset(torch.from_numpy(x_test).float())
        test_loader = DataLoader(x_test, batch_size=ts_mb_size, shuffle=False)

        torch_train = torch.from_numpy(x_train)
    elif dataname == 'caltech101':
        ###################################################
        ## Load Caltech101 dataset
        ###################################################
        os.chdir('Data/caltech101_data')
        def reshape_data(data):
            return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='fortran')
        caltech_raw = scipy.io.loadmat('caltech101_silhouettes_28_split1.mat')

        ####################################################
        ## split dataset
        # train and test data
        x_train = 1. - reshape_data(caltech_raw['train_data'].astype('float32'))
        np.random.shuffle(x_train)
        x_val = 1. - reshape_data(caltech_raw['val_data'].astype('float32'))
        np.random.shuffle(x_val)
        x_test = 1. - reshape_data(caltech_raw['test_data'].astype('float32'))

        y_train = caltech_raw['train_labels']
        y_val = caltech_raw['val_labels']
        y_test = caltech_raw['test_labels']



        #####################################################
        ## dataset to loader
        #####################################################
        tr_mb_size , val_mb_size , ts_mb_size = 100 , 100 , 100
        train_loader = DataLoader(x_train, batch_size=tr_mb_size, shuffle=True)
        train_valid_loader = DataLoader(torch.from_numpy(x_train) , batch_size=tr_mb_size, shuffle=False)
        valid_loader = DataLoader(x_val, batch_size=val_mb_size, shuffle=False)
        test_loader = DataLoader(torch.from_numpy(x_test) , batch_size=ts_mb_size , shuffle=False)

        torch_train = torch.from_numpy(x_train)
    else:
        print('Invalid DatasetName!')
        
    return train_loader,train_valid_loader,valid_loader,test_loader,torch_train

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


if __name__ == "__main__":
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #parameter setting
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    use_cuda = args.use_cuda
    gpu_num = args.gpu_num
    dataname = args.dataname
    X_dim = args.X_dim
    Z_dim = args.Z_dim
    h_dim = args.h_dim
    n_hidden_dec = args.n_hidden_dec
    n_hidden_prop = args.n_hidden_prop

    num_sam = args.num_sam; num_aggr = args.num_aggr
    option = args.option


    alpha_inc = args.alpha_inc
    
    decoder_option = args.decoder_option
    eval_option = args.eval_option
    
    add_var = args.add_var
    p_add_var = args.p_add_var
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    ## The id of GPU
    torch.cuda.set_device(gpu_num)
    if (dataname == 'bimnist') | (dataname == 'caltech101'):
        binarization = False
    else:
        binarization = True
    train_loader,train_valid_loader,valid_loader,test_loader,torch_train = load_data(dataname,binarization)

    os.chdir('../..')
    if decoder_option == 'mlp':
        from Models_Calculate import *
        from Models_MLP import *
        decoder_path = '/decoder_with_'+option+'_addvar_'+str(int(add_var*10))+'_p_addvar_'+str(int(p_add_var*10))+'_alpha_inc'+str(alpha_inc)+'_samp'+ str(num_sam)+'_eval_'+eval_option+'_191006.pth.tar'
        proposal_path = '/proposal_with_'+option+'_addvar_'+str(int(add_var*10))+'_p_addvar_'+str(int(p_add_var*10))+'_alpha_inc'+str(alpha_inc)+'_samp'+ str(num_sam)+'_eval_'+eval_option+'_191006.pth.tar'
            
    elif decoder_option == 'cnn':
        h_dim = 294
        #num_sam_tr = 10; num_sam_ts = 10; num_aggr = 10
        from Models_Calculate import *
        from Models_CNN import *
        decoder_path = '/cnn_decoder_with_'+option+'_addvar_'+str(int(add_var*10))+'_p_addvar_'+str(int(p_add_var*10))+'_alpha_inc'+str(alpha_inc)+'_samp'+ str(num_sam)+'_eval_'+eval_option+'_191006.pth.tar'
        proposal_path = '/cnn_proposal_with_'+option+'_addvar_'+str(int(add_var*10))+'_p_addvar_'+str(int(p_add_var*10))+'_alpha_inc'+str(alpha_inc)+'_samp'+ str(num_sam)+'_eval_'+eval_option+'_191006.pth.tar'
    
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    #model constructor
    decoder = Decoder(Z_dim, h_dim, X_dim , n_hidden_dec)
    prop_dist = Encoder(X_dim , h_dim, Z_dim , n_hidden_prop)

    print('Current number of the GPU is %d'%torch.cuda.current_device())
    print('Dataset is ' + dataname)
    print('option is ' + option)
    print('decoder_option is ' + decoder_option)
    print('sample size is ' + str(num_sam))

    if use_cuda:
        decoder = decoder.cuda()
        prop_dist = prop_dist.cuda()
    
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=3e-4)
    prop_optimizer = optim.Adam(prop_dist.parameters(), lr=3e-4)


    optimizer_list = [decoder_optimizer , prop_optimizer]

    #####################################################
    ## Set tuning parameters
    #####################################################
    max_epochs = 1500
    tr_mb_size , val_mb_size , ts_mb_size = 32 , 32 , 32
    save = True
    save_path =  'Model/checkpoint/'+dataname


    #####################################################
    ## Start Training
    #####################################################
    train_estep_loss_vec , test_app_nll_vec = [] , []
    train_estep_loss_x_vec , train_estep_loss_z_vec = [] , []
    train_avg_std_vec = []
    train_residual_vec = []

    train_app_nll_vec , train_KL_vec = [] , []
    test_app_nll_vec , test_KL_vec = [] , []
    train_kl_vec , train_bl_vec = [], []
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    alpha = 0.
    patience = 0.
    best_val_nll = 1000.


    for epoch in range(max_epochs):
        print('.............................................')
        print('Epoch : %d'%(epoch+1))
        start_time = time.time()

        ## train mode
        ## train
        decoder.train(); prop_dist.train()
        train(train_loader , decoder , prop_dist , optimizer_list , num_sam , num_aggr , option , alpha , use_cuda,binarization,add_var,p_add_var)
        ## evaluation mode
        ## evaluation
        decoder.eval(); prop_dist.eval()
        if eval_option is 'valid':
            val_app_nll , val_KL = evaluate(decoder , prop_dist , valid_loader , num_sam , num_sam , use_cuda , True,add_var)
        else:
            val_app_nll , val_KL = evaluate(decoder , prop_dist , test_loader , num_sam , num_sam , use_cuda , True,add_var)
        
        alpha = alpha + alpha_inc
        if alpha > 1.:
            alpha = 1.

        if best_val_nll > val_app_nll.mean():
            print('...The best model is estimated')
            patience = 0        
            best_val_nll = val_app_nll.mean()
            if save:
                torch.save(decoder.state_dict() , save_path+decoder_path) 
                torch.save(prop_dist.state_dict() , save_path+proposal_path) 
        else:
            patience += 1
            if alpha < 1.:
                patience = 0
            print('...The current patience is %d'%patience)

        print('...The best nll is %f'%best_val_nll)

        end_time = time.time()
        print('Elapsed time : %f seconds'%(np.round(end_time-start_time,1)))

        if patience is 50:
            print('.............................................')
            print('.............................................')
            print('End of learning procedure for no more improvement')
            break

    #####################################################
    ## Load the best model
    #####################################################

    decoder.load_state_dict(torch.load(save_path+decoder_path)) 
    prop_dist.load_state_dict(torch.load(save_path+proposal_path)) 

    #####################################################
    ## Calculate the nll using sample size 5000
    #####################################################
    prop_dist.eval()
    decoder.eval()
    best_test_app_nll , _ = evaluate(decoder , prop_dist , test_loader , 5000 , num_aggr , use_cuda , True,add_var)

    print(save_path)
    print(decoder_path)

