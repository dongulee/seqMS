# Sequential Model Selection
from __future__ import print_function
import argparse
import torch
# import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
# import pickle
import json
import os

import time

import utils
from utils import update_args
from utils import Logger
from utils import MNIST_Mgr
from utils import ExpMgr

import networks
from networks import CustomCNN
from networks import FCNet300
from networks import FCNet300_100
from networks import Net

## LOGGING
logger = Logger('./logs/')

# Dataset Manager
MNIST = MNIST_Mgr()

# Exp Manager
expmgr = ExpMgr(logger)


# DL Opeartions
def train(args, model, device, train_loader, optimizer, epoch): # FIXME: log training result in more detail
    model.train()
    train_loss=None
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # print(target)
        output = model(data)
        loss = model.loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            train_loss = loss.item()          
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss))
            if args.dry_run:
                break
    return train_loss
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += model.loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))

    return test_loss, test_acc

def pre_train(args):
    '''
    Train function from scratch

    args.pre_train_size (Int)
    args.mnist_test (Bool) #FIXME: -> 'train_data' = {'mnist_train', 'mnist_test'}
    args.split_ratio (Float)
    args.model_arch (String) 
    '''
    t0 = time.time()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    dataset1 = None
    if args.mnist_test==True:
        dataset1 = torch.utils.data.Subset(MNIST.get_train(), range(args.pre_train_size))
        dataset2 = MNIST.get_test()

    else:
        train_size = int(args.split_ratio * args.pre_train_size)
        dataset1 = torch.utils.data.Subset(MNIST.get_train(), range(train_size))
        dataset2 = torch.utils.data.Subset(MNIST.get_train(), range(train_size, args.pre_train_size))

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = networks.get_architecture(args.model_arch).to(device)
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    result = {
            # model
            'architecture': args.model_arch,
            'timestep': 0,
            # dataset
            'pre_train_size':args.pre_train_size,
            'mnist_test': args.mnist_test,
            'split_ratio': args.split_ratio,
            'model_arch': args.model_arch,
            # train parameters
            'epochs': args.epochs,

            }
    
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    result['train_losses'] = []
    result['test_loss/acc'] = []
    for epoch in range(1, args.epochs + 1):
        loss = train(args, model, device, train_loader, optimizer, epoch)
        result['train_losses'].append((epoch, loss))
        if epoch == args.epochs:
             result['test_loss/acc'].append((epoch, test(model, device, test_loader)))
        scheduler.step()
    
    t1 = time.time()    
    result['training_time'] = (t1-t0) # in seconds
    
    if args.save_model:
        # Model Checkpoint Filename: <arch_name>_<exp_num>_<timestep>.pt
        result['out_model'] = utils.create_ckpt_id(args)    
        torch.save({
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict()
        }, result['out_model'])
    
        logger.append_log(result)
        return result['out_model']
    logger.append_log(result)    
    return result['out_model']

def retrain(args, training_dataset, testing_dataset):
    """
    arguments to re-train existing models
    'base_model': <full_path_to_model_ckpt>
    'train_data': {'mnist_train', mnist_test'}
    'test_data': (DataVersion)
    'timestep' (INT) 
    'stepsize' (INT)
    """
    t0 = time.time()
    t1 = 0
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # training_dataset = MNIST.get_by_vid(args.train_data)
    # testing_dataset = MNIST.get_by_vid(args.test_data)
    testing_oracle = MNIST.get_test()
    

    train_loader = torch.utils.data.DataLoader(training_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(testing_dataset, **test_kwargs)
    test_loader_oracle = torch.utils.data.DataLoader(testing_oracle, **test_kwargs)

    model = networks.get_architecture(args.model_arch).to(device)
    ckpt = torch.load(args.base_model)
    model.load_state_dict(ckpt['model_state_dict'])
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    #optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    result = {
            'architecture': args.model_arch,
            'data_train': '9/10 of X_0:t=MNIST_Train[0:{}*{}]'.format(args.timestep, args.stepsize),
            'data_test': '1/10 of X_0:t=MNIST_Train[0:{}*{}]'.format(args.timestep, args.stepsize),
            'timestep': args.timestep,
            'stepsize': args.stepsize,
            'base_model': args.base_model
    }

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    result['train_losses'] = []
    result['test_loss/acc'] = []
    result['oracle_loss/acc'] = []
    for epoch in range(1, args.epochs + 1):
        loss = train(args, model, device, train_loader, optimizer, epoch)
        result['train_losses'].append((epoch, loss))
            
        if epoch == args.epochs or (epoch % args.test_interval)==0:
            t1 = time.time()
            result['test_loss/acc'].append((epoch, test(model, device, test_loader)))
            result['oracle_loss/acc'].append((epoch, test(model, device, test_loader_oracle)))
        scheduler.step()
    
    result['training_time'] = (t1-t0)
    if args.save_model:
        result['out_model'] =  utils.create_ckpt_id(args)
        torch.save({
            'model_state_dict':model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict()
        }, result['out_model'])
        
        logger.append_log(result)
        return result['out_model']
    logger.append_log(result)
    return result['out_model']

def infer(args):
    t0 = time.time()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    testing_dataset = None
    if(args.mnist_test == True):
        testing_dataset = MNIST.get_test()
    else:
        testing_dataset = MNIST.get_by_vid(args.test_data)
    test_loader = torch.utils.data.DataLoader(testing_dataset, **test_kwargs)

    print(args.base_model, " is tested for MNIST test")
    model = networks.get_architecture(args.model_arch).to(device)
    ckpt = torch.load(args.base_model)
    model.load_state_dict(ckpt['model_state_dict'])

    result = { #FIXME:
    }

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    # result['train_losses'] = []
    result['test_loss/acc'] = test(model, device, test_loader)
    t1 = time.time()
    result['test_time'] = (t1-t0)
    
    # logger.append_log(result)
    return result

# Experiments
def seqMS_exp4(args):
    """
    Initial Model Preparation
        Train: MNIST_Train[0:90]
        Test: MNIST_Train[90:100]
    
    Continual Training t=1,2,3,..., T = 30000/stepsize
        X_t_train: MNIST_Train[stepsize*(t):stepsize*(t+0.9)]
        X_t_test: MNIST_Train[stepsize*(t+0.9):stepsize*(t+0.1)]
        
        training set = X_0:t_train
        testing set = X_0:t_test
    
    Configurations
        stepsize = 100
        batch_size = 64
        epochs = 3 <- low computation cost
    """
    # Pre-train
    print('------')
    print('pretrain start')
    args_plus = update_args(args, 
    [
        ('pre_train_size', 100),
        ('split_ratio', 0.9),
        ('mnist_test', False),
        ('output_dir', "./models/"),
        ('epochs', 3),
        ('test_interval', 3)
    ])
    mt = pre_train(args_plus)
    print('pretrained model id: {}'.format(mt))
    print('------')
    
    print("sequential training start")
    
    stepsize = 100
    T = int(30000/stepsize)
    print("step size: {}, T: {}".format(stepsize, T))
    
    Xt_train_map = list(range(100, 190))
    Xt_test_map = list(range(190, 200))

    for t in range(1,T):
        X_train = torch.utils.data.Subset(MNIST.get_train(), Xt_train_map)
        X_test = torch.utils.data.Subset(MNIST.get_train(), Xt_test_map)
        args_plus = update_args(args_plus,
        [
            ('base_model', mt),
            ('timestep', t),
            ('stepsize', stepsize),
        ])
        print("timestep: {} re-train is started".format(t))
        mt = retrain(args_plus, X_train, X_test)
        print("re-trained model: {}".format(mt))
        Xt_train_map = Xt_train_map + list(range((t+1)*stepsize, int((t+1.9)*stepsize) ))
        Xt_test_map = Xt_test_map + list(range(int((t+1.9)*stepsize), (t+2)*stepsize ))
        print("new data Xt[{}:{}] has arrived".format((t+1)*stepsize, (t+2)*stepsize))

def seqMS_exp3(args):
    """
    #FIXME: bug exists
    Initial Model Preparation
        Train: MNIST_Train[0:90]
        Test: MNIST_Train[90:100]
    
    Continual Training t=1,2,3,..., T = 59000/stepsize
        Train: MNIST_Train[:stepsize*(t+1)*(split_ratio)]
        Test: MNIST_Train[stepsize*(t+1)*(split_ratio):] #FIXME: case2: Xt-1, case1: historical
        Comparison: MNIST_Test
    
    Configurations
        stepsize = 100
        batch_size = 64
        epochs = 3 <- low computation cost
    """
    # Pre-train
    print('------')
    print('pretrain start')
    args_plus = update_args(args, 
    [
        ('pre_train_size', 100),
        ('split_ratio', 0.9),
        ('mnist_test', False),
        ('output_dir', "./models/"),
        ('epochs', 3),
        ('test_interval', 3)
    ])
    mt = pre_train(args_plus)
    print('pretrained model id: {}'.format(mt))
    print('------')
    
    print("sequential training start")
    
    stepsize = 100
    T = int(59000/stepsize)
    print("step size: {}, T: {}".format(stepsize, T))

    Xtest0_dict = {
        'mnist_train': {'type': 'range', 'start_idx': 90, 'end_idx': 100},
        'mnist_test': None,
        'exp_id':args.exp_id,
        'desc':'new arrival Xt at timestep {}'.format(1)
    }
    
    Xtest_vid = MNIST.create_version(Xtest0_dict)
    print("Xtest 0 id: {}".format(Xtest_vid))
    
    for t in range(1,T):
        
        Xt_vid = MNIST.create_version({
            'mnist_train': {'type': 'range', 'start_idx': 0, 'end_idx': int((t+1)*stepsize*args.split_ratio)},
            'mnist_test': None,
            'exp_id':args.exp_id,
            'desc':'new arrival Xt at t={}'.format(t)
        })
        args_plus = update_args(args_plus,
        [
            ('base_model', mt),
            ('train_data', Xt_vid),
            ('test_data', Xtest_vid),
            ('timestep', t),
            ('stepsize', stepsize),
        ])
        print("timestep: {} re-train is started".format(t))
        print("Xt_id: {}, Xtest_id: {}".format(Xt_vid, Xtest_vid))
        mt = retrain(args_plus)
        Xtest_vid = Xt_vid
        print("re-trained model: {}".format(mt))

def seqMS_exp2(args):
    """
    Initial Model Preparation
        Train: MNIST_Train[0:900]
        Test: MNIST_Train[900:1000]
    
    Continual Training t=1,2,3,..., T = 59000/stepsize
        Train: MNIST_Train[stepsize*t:stepsize*(t+1)]
        Test: MNIST_Train[900:stepsize*t] #FIXME: case2: Xt-1, case1: historical
        Comparison: MNIST_Test
    
    Configurations
        stepsize = 1000
        batch_size = 64
        epochs = 10 .- it is another problem to get this value

    """
    # Pre-train
    args_plus = update_args(args, 
    [
        ('pre_train_size', 100),
        ('split_ratio', 0.9),
        ('mnist_test', False),
        ('model_arch', "fcnn1"),
        ('output_dir', "./models/"),
        ('epochs', 3),
        ('test_interval', 3)
    ])
    mt = pre_train(args_plus)

    stepsize = 100
    T = int(59000/stepsize)

    
    Xtest0_dict = {
        'mnist_train': {'type': 'range', 'start_idx': 90, 'end_idx': 100},
        'mnist_test': None,
        'exp_id':args.exp_id,
        'desc':'new arrival Xt at timestep {}'.format(1)
    }
    
    Xtest_vid = MNIST.create_version(Xtest0_dict)

    for t in range(1,T):
        Xt_vid = MNIST.create_version({
            'mnist_train': {'type': 'range', 'start_idx': t*stepsize, 'end_idx': (t+1)*stepsize},
            'mnist_test': None,
            'exp_id':args.exp_id,
            'desc':'new arrival Xt at t={}'.format(t)
        })
        args_plus = update_args(args_plus,
        [
            ('base_model', mt),
            ('train_data', Xt_vid),
            ('test_data', Xtest_vid),
            ('timestep', t),
            ('stepsize', stepsize),
        ])
        mt = retrain(args_plus)
        Xtest_vid = Xt_vid

def seqMS_test(args):
    logfile = './logs/19.log'
    a = None
    with open(logfile) as f:
        tmp = f.readline()
        tmp = '[' + tmp +  ']'
        a = json.loads(tmp)
    log = a
    header = log[0]
    pre_train = log[1]
    retrains = log[2:]

    test_data = MNIST.get_test()
    args_plus = update_args(args, [
        ('model_arch', 'fcnn1'),
        ('mnist_test',True ),
    ])
    for step in retrains:
        Mt = step['out_model']
        args_plus = update_args(args, [
            ('base_model', Mt)
        ])
        test_result = infer(args_plus)
        step['MNIST_test_result'] = test_result
    logger.append_log(retrains)
     

        


    
def seqMS_lc(args):
    """
    To see a learning curve learning with small dataset
    """
    args_plus = update_args(args, 
    [
        ('pre_train_size', 1000),
        ('mnist_test', True),
        ('model_arch', "fcnn2"),
        ('output_dir', "./models/"),
        ('epochs', 20),
        ('test_interval', 1)
    ])
    pre_train(args_plus)    
    
    args_plus = update_args(args, 
    [
        ('model_arch', "cnn1"),
    ])
    pre_train(args_plus)
    
    args_plus = update_args(args, 
    [
        ('model_arch', "cnn2"),
    ])
    pre_train(args_plus)

def seqMS_scratch(args):
    """
    Experiment: seqMS_scratch (=training from scratch for seqMS)
        This experiment aims to make offline-trained models. (comparison group)
        MNIST Dataset was used to train 4 models and to test accuracy of them.
    
    """
    args_plus = update_args(args, 
    [
        ('pre_train_size', 60000),
        ('mnist_test', True),
        ('model_arch', "fcnn1"),
        ('output_dir', "./models/")
    ])
    pre_train(args_plus)
    args_plus = update_args(args_plus,[('model_arch', "fcnn2")])
    pre_train(args_plus)
    args_plus = update_args(args_plus,[('model_arch', "cnn1")])
    pre_train(args_plus)
    args_plus = update_args(args_plus,[('model_arch', "cnn2")])
    pre_train(args_plus)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--model-arch', type=str, default="cnn1",
                        help='Choose model architecture')
    #parser.add_argument('--retrain', action='store_true', default=False,
    #                    help='sequential retraining experiment')
    parser.add_argument('--timestep',type=int, default=0,
                        help='sequential retraining timestep')
    parser.add_argument('--stepsize',type=int, default=100,
                        help='size of the retraining timestep')
    parser.add_argument('--split-ratio',type=float, default=0.,
                        help='training/test data split ratio')
    parser.add_argument('--exp-desc',type=str, default="No description",
                        help='description of the experiment')

    args = parser.parse_args()

    expmgr.create_exp(args, seqMS_scratch)

if __name__ == '__main__':
    main()
