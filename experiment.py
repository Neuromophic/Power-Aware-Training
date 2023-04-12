#!/usr/bin/env python

#SBATCH --job-name=Aging

#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out

#SBATCH --mail-user=hzhao@teco.edu

#SBATCH --export=ALL

#SBATCH --time=48:00:00

#SBATCH --partition=sdil
#SBATCH --gres=gpu:1

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'utils'))
import torch
import pprint
import pNN_Power_Aware as pNN
from configuration import *
from torch.utils.data import TensorDataset, DataLoader
from utils import *

args = parser.parse_args()

args.DEVICE = CheckDevice(args)
print(f'Training network on device: {args.DEVICE}.')

MakeFolder(args)

train_loader, valid_loader, test_loader, datainfo = GetDataLoader(args)
pprint.pprint(datainfo)

SetSeed(args.SEED)

setup = f"data:{datainfo['dataname']}_seed:{args.SEED}_Penalty:{args.powerestimator}_Factor:{args.powerbalance}_Progressive:{args.PROGRESSIVE}"
print(f'Training setup: {setup}.')

msglogger = GetMessageLogger(args, setup)
msglogger.info(f'Training network on device: {args.DEVICE}.')
msglogger.info(f'Training setup: {setup}.')
msglogger.info(datainfo)

if os.path.isfile(f'{args.savepath}/pNN_{setup}'):
    print(f'{setup} exists, skip this training.')
    msglogger.info('Training was already finished.')
else:
    topology = [datainfo['N_feature']] + args.hidden + [datainfo['N_class']]
    msglogger.info(f'Topology of the network: {topology}.')
    
    pnn = pNN.pNN(topology, args).to(args.DEVICE)

    lossfunction = pNN.Lossfunction(args).to(args.DEVICE)
    optimizer = torch.optim.Adam(pnn.parameters(), lr=args.LR)

    if args.PROGRESSIVE:
        pnn, best = train_pnn_progressive(pnn, train_loader, valid_loader, lossfunction, optimizer, args, msglogger, UUID=setup)
    else:
        pnn, best = train_pnn(pnn, train_loader, valid_loader, lossfunction, optimizer, args, msglogger, UUID=setup)
    
    if best:
        if not os.path.exists(f'{args.savepath}/'):
            os.makedirs(f'{args.savepath}/')
        torch.save(pnn, f'{args.savepath}/pNN_{setup}')
        msglogger.info('Training if finished.')
    else:
        msglogger.warning('Time out, further training is necessary.')