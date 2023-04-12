import numpy as np
import torch

class InvRT(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # R1n, k1, R3n, k2, R5n, Wn, k3
        # be careful, k1, k2, k3 are not normalized
        self.rt_ = torch.nn.Parameter(torch.tensor(config.neg_rt_), requires_grad=True)
        # model
        package = torch.load('./neg_model_package')
        self.eta_estimator = package['eta_estimator']
        self.eta_estimator.train(False)
        self.X_max = package['X_max']
        self.X_min = package['X_min']
        self.Y_max = package['Y_max']
        self.Y_min = package['Y_min']
    
    @property
    def RT_(self):
        # keep values in (0,1)
        rt_temp = torch.sigmoid(self.rt_)
        # calculate normalized (only R1n, R3n, R5n, Wn, Ln)
        RTn = torch.zeros([10])
        RTn[0] = rt_temp[0]    # R1n
        RTn[2] = rt_temp[2]    # R3n
        RTn[4] = rt_temp[4]    # R5n
        RTn[5] = rt_temp[5]    # Wn
        RTn[6] = rt_temp[6]    # Ln
        # denormalization
        RT = RTn * (self.X_max - self.X_min) + self.X_min
        # calculate R2, R4
        R2 = RT[0] * rt_temp[1] # R2 = R1 * k1
        R4 = RT[2] * rt_temp[3] # R4 = R3 * k2
        # stack new variable: R1, R2, R3, R4, R5, W, L
        RT_full = torch.stack([RT[0], R2, RT[2], R4, RT[4], RT[5], RT[6]])
        return RT_full
    
    @property
    def RT(self):
        RT_full = torch.zeros([10])
        RT_full[:7] = self.RT_.clone()
        RT_full[RT_full>self.X_max] = self.X_max[RT_full>self.X_max]
        RT_full[RT_full<self.X_min] = self.X_min[RT_full<self.X_min]
        return RT_full[:7].detach() + self.RT_ - self.RT_.detach()
    
    @property
    def RT_extend(self):
        R1 = self.RT[0]
        R2 = self.RT[1]
        R3 = self.RT[2]
        R4 = self.RT[3]
        R5 = self.RT[4]
        W  = self.RT[5]
        L  = self.RT[6]
        k1 = R2 / R1
        k2 = R4 / R3
        k3 = L / W
        return torch.hstack([R1, R2, R3, R4, R5, W, L, k1, k2, k3])

    @property
    def RTn_extend(self):
        return (self.RT_extend - self.X_min) / (self.X_max - self.X_min) 
    
    @property
    def eta(self):
        eta_n = self.eta_estimator(self.RTn_extend)
        eta = eta_n * (self.Y_max - self.Y_min) + self.Y_min
        return eta
    
    def forward(self, z):
        eta = self.eta
        return - (eta[0] + eta[1] * torch.tanh((z - eta[2]) * eta[3]))
    
    
    
    
class TanhRT(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.rt_ = torch.nn.Parameter(torch.tensor(config.act_rt_), requires_grad=True)
        
        # model
        package = torch.load('./act_model_package')
        self.eta_estimator = package['eta_estimator']
        self.eta_estimator.train(False)
        self.X_max = package['X_max']
        self.X_min = package['X_min']
        self.Y_max = package['Y_max']
        self.Y_min = package['Y_min']
    
    @property
    def RT(self):
        # keep values in (0,1)
        rt_temp = torch.sigmoid(self.rt_)
        # denormalization
        RTn = torch.zeros([9])
        RTn[0] = rt_temp[0]    # R1n
        RTn[1] = rt_temp[1]    # R2n
        RTn[2] = rt_temp[2]    # W1n
        RTn[3] = rt_temp[3]    # L1n
        RTn[4] = rt_temp[4]    # W2n
        RTn[5] = rt_temp[5]    # L2n
        RT = RTn * (self.X_max - self.X_min) + self.X_min
        return RT[:6]
    
    @property
    def RT_extend(self):
        R1 = self.RT[0]
        R2 = self.RT[1]
        W1 = self.RT[2]
        L1 = self.RT[3]
        W2 = self.RT[4]
        L2 = self.RT[5]
        k1 = R2 / R1
        k2 = L1 / W1
        k3 = L2 / W2
        return torch.hstack([R1, R2, W1, L1, W2, L2, k1, k2, k3])

    @property
    def RTn_extend(self):
        return (self.RT_extend - self.X_min) / (self.X_max - self.X_min) 
    
    @property
    def eta(self):
        eta_n = self.eta_estimator(self.RTn_extend)
        eta = eta_n * (self.Y_max - self.Y_min) + self.Y_min
        return eta
    
    def forward(self, z):
        eta = self.eta
        return eta[0] + eta[1] * torch.tanh((z - eta[2]) * eta[3])

            
            
            
class pLayer(torch.nn.Module):
    def __init__(self, n_in, n_out, args):
        super().__init__()
        self.args = args
        
        theta = torch.rand([n_in + 2, n_out])/100. + args.gmin
        theta[-1, :] = theta[-1, :] + args.gmax
        theta[-2, :] = args.ACT_eta3/(1.-args.ACT_eta3)*(torch.sum(theta[:-2,:], axis=0)+theta[-1,:])
        self.theta_ = torch.nn.Parameter(theta, requires_grad=True)
    
    @property
    def device(self):
        return self.args.DEVICE
    
    @property
    def theta(self):
        self.theta_.data.clamp_(-self.args.gmax, self.args.gmax)
        theta_temp = self.theta_.clone()
        theta_temp[theta_temp.abs() < self.args.gmin] = 0.
        return theta_temp.detach() + self.theta_ - self.theta_.detach()
    
    @property
    def W(self):
        return self.theta.abs() / torch.sum(self.theta.abs(), axis=0, keepdim=True)

    def INV(self, x):
        return -(self.args.NEG_eta1 + self.args.NEG_eta2 * torch.tanh((x - self.args.NEG_eta3) * self.args.NEG_eta4))
    
    def MAC(self, a):
        # 0 and positive thetas are corresponding to no negative weight circuit
        positive = self.theta.clone().to(self.device)
        positive[positive >= 0] = 1.
        positive[positive < 0]  = 0.
        negative = 1. - positive
        a_extend = torch.cat([a,
                              torch.ones( [a.shape[0], 1]).to(self.device),
                              torch.zeros([a.shape[0], 1]).to(self.device)], dim=1)
        a_neg = self.INV(a_extend)
        a_neg[:,-1] = 0.
        z = torch.matmul(a_extend, self.W * positive) + torch.matmul(a_neg, self.W * negative)
        return z
    
    def ACT(self, z):
        return self.args.ACT_eta1 + self.args.ACT_eta2 * torch.tanh((z - self.args.ACT_eta3) * self.args.ACT_eta4)
    
    def forward(self, a_previous):
        z_new = self.MAC(a_previous)
        self.g_tilde = self.ScaledConductance()
        self.power = self.Power(a_previous, z_new)
        a_new = self.ACT(z_new)
        return a_new
    
    def ScaledConductance(self):
        g_initial = self.theta.abs()
        g_max = g_initial.max(dim=0, keepdim=True)[0]
        scaler = self.args.gmax / g_max
        return g_initial * scaler
    
    def Power(self, x, y):
        x_extend = torch.cat([x,
                              torch.ones( [x.shape[0], 1]).to(self.device),
                              torch.zeros([x.shape[0], 1]).to(self.device)], dim=1)
        x_neg = self.INV(x_extend)
        x_neg[:,-1] = 0.
        
        E = x_extend.shape[0]
        M = x_extend.shape[1]
        N = y.shape[1]
        
        positive = self.theta.clone().to(self.device)
        positive[positive >= 0] = 1.
        positive[positive < 0]  = 0.
        negative = 1. - positive
        
        Power = torch.tensor(0.)
        for E in range(E):
            for m in range(M):
                for n in range(N):
                    Power += self.g_tilde[m,n] * ((x_extend[:,m]*positive[m,n]+x_neg[:,m]*negative[m,n])-y[:,n]).pow(2.).sum()
        Power = Power / E
        return Power
    
    def WeightAttraction(self):
        mean = self.theta.mean(dim=0)
        diff = self.theta - mean
        return diff.pow(2.).sum()
    
    def SetParameter(self, name, value):
        if name == 'args':
            self.args = value
    

class pNN(torch.nn.Module):
    def __init__(self, topology, args):
        super().__init__()
        self.model = torch.nn.Sequential()
        for i in range(len(topology)-1):
            self.model.add_module(f'{i}-th pLayer', pLayer(topology[i], topology[i+1], args))
    
    def forward(self, X):
        return self.model(X)
    
    @property
    def device(self):
        return self.args.DEVICE
    
    def Power(self):
        power = torch.tensor(0.)
        for l in self.model:
            power += l.power
        return power
    
    def WeightAttraction(self):
        penalty = torch.tensor(0.)
        for l in self.model:
            penalty += l.WeightAttraction()
        return penalty
    
    def SetParameter(self, name, value):
        if name == 'args':
            self.args = value
            for m in self.model:
                m.SetParameter('args', self.args)
        

class Lossfunction(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        

    def standard(self, prediction, label):   
        label = label.reshape(-1, 1)
        fy = prediction.gather(1, label).reshape(-1, 1)
        fny = prediction.clone()
        fny = fny.scatter_(1, label, -10 ** 10)
        fnym = torch.max(fny, axis=1).values.reshape(-1, 1)
        l = torch.max(self.args.m + self.args.T - fy, torch.tensor(0)) + torch.max(self.args.m + fnym, torch.tensor(0))
        L = torch.mean(l)
        return L
    
    def PowerEstimator(self, nn, x):
        _ = nn(x)
        return nn.Power()
    
    def WeightAttractor(self, nn):
        return nn.WeightAttraction()
    
    def forward(self, nn, x, label):
        if self.args.powerestimator == 'attraction':
            return self.standard(nn(x), label)*0. + self.args.powerbalance * self.WeightAttractor(nn)
        elif self.args.powerestimator == 'power':
            return self.standard(nn(x), label)*0. + self.args.powerbalance * self.PowerEstimator(nn,x)

    