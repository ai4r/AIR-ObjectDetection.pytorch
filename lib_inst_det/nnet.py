import _init_paths
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import lib_inst_det.backbone as backbone
import cv2
import pdb
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.weight_norm import WeightNorm

# If you want to use fc_type='protoCosine',
# copy and fill the below class from 'class distLinear' in https://github.com/wyharveychen/CloserLookFewShot
class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        pass

    def forward(self, x):
        pass


# only contain the networks
class NNet(nn.Module):
    def __init__(self, num_class, num_hid_ftr=1024, base_net='ResNet50', fix_base_net=True, num_layer=1, fc_type='linear', use_all_layers=False):
        super(NNet, self).__init__()
        self.use_all_layers = use_all_layers

        if self.use_all_layers:
            self.num_in_ftr = 3904
        else:
            self.num_in_ftr = 2048
        self.num_hid_ftr = num_hid_ftr
        self.num_class = num_class
        self.num_layer = num_layer

        self.sharedNet = backbone.network_dict[base_net](use_all_layers=self.use_all_layers)

        if num_layer == 3:
            self.fc1 = nn.Linear(self.num_in_ftr, self.num_hid_ftr, bias=True)
            self.fc2 = nn.Linear(self.num_hid_ftr, self.num_hid_ftr, bias=True)

            if fc_type == 'linear':
                self.fc3 = nn.Linear(self.num_hid_ftr, self.num_class, bias=True)
            elif fc_type == 'protoCosine':
                self.fc3 = distLinear(self.num_hid_ftr, self.num_class)
            else:
                raise AssertionError('fc_type should be linear or protoCosine, but ', fc_type)

        elif num_layer == 1:
            if fc_type == 'linear':
                self.fc1 = nn.Linear(self.num_in_ftr, self.num_class, bias=True)
            elif fc_type == 'protoCosine':
                self.fc1 = distLinear(self.num_in_ftr, self.num_class)
            else:
                raise AssertionError('fc_type should be linear or protoCosine, but ', fc_type)
        else:
            raise AssertionError('num_layer should be 1 or 3, but ', num_layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)

        # initialize
        if num_layer == 3:
            nn.init.normal_(self.fc1.weight, 0, 0.01)
            nn.init.normal_(self.fc2.weight, 0, 0.01)
            if fc_type == 'linear':
                nn.init.normal_(self.fc3.weight, 0, 0.01)
            elif fc_type == 'protoCosine':
                nn.init.normal_(self.fc3.L.weight, 0, 0.01)
        elif num_layer == 1:
            if fc_type == 'linear':
                nn.init.normal_(self.fc1.weight, 0, 0.01)
            elif fc_type == 'protoCosine':
                nn.init.normal_(self.fc1.L.weight, 0, 0.01)

        if fix_base_net:
            # sharedNet is fixed.
            for param in self.sharedNet.parameters():
                param.requires_grad = False


    def forward(self, x):
        x = self.sharedNet(x)

        if self.num_layer == 3:
            x = self.dropout(self.relu(self.fc1(x)))         # relu, dropout
            x = self.dropout(self.relu(self.fc2(x)))
            x = self.fc3(x)
        elif self.num_layer == 1:
            x = self.fc1(x)

        x_prob = self.softmax(x)

        return x, x_prob

# contains everything related to training and inference
class NNClassifier():
    def __init__(self, nameCategory, num_hid_ftr, num_class, base_net='ResNet50', fix_base_net=False, num_layer=3, fc_type='linear', use_all_layers=False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.nameCategory = nameCategory.replace(' ', '_')
        # self.num_class = num_class
        # self.num_hid_ftr = num_hid_ftr
        # self.base_net = base_net
        self.NNet = NNet(num_class, num_hid_ftr, base_net, fix_base_net=fix_base_net, num_layer=num_layer, fc_type=fc_type, use_all_layers=use_all_layers).to(self.device)
        if torch.cuda.is_available():
            self.NNet.cuda()

    # save the classifier to disk
    def save(self, path_to_model):
        torch.save({
            'nameCategory': self.nameCategory,
            # 'num_class': self.num_class,
            # 'num_hid_ftr': self.num_hid_ftr,
            # 'base_net': self.base_net,
            'model_state_dict': self.NNet.state_dict()
        }, path_to_model)

    # load the classifier from disk
    def load(self, path_to_model, num_class_from_files):
        # load information and network
        ret = False

        if os.path.exists(path_to_model):
            print('load model: ', path_to_model)

            if torch.cuda.is_available():
                model = torch.load(path_to_model)
            else:
                model = torch.load(path_to_model, map_location='cpu')

            self.nameCategory = model['nameCategory']
            # self.num_class = model['num_class']
            # self.num_hid_ftr = model['num_hid_ftr']
            # self.base_net = model['base_net']

            # if self.num_class == num_class_from_files:
            self.NNet.load_state_dict(model['model_state_dict'])
            self.NNet.eval()

            ret = True

        return ret


    def inference(self, x):
        y_score, y_prob = self.NNet(x)
        y_est = torch.argmax(y_prob, dim=1)

        return y_est, y_prob[0][y_est]

    def train(self, imageloader, max_epoch=200, learning_rate=0.005, test_epoch=1, stop_acc=0.99):
        # data to dataloader
        # tensor_x = torch.Tensor(data_x).to(self.device)
        # tensor_y = torch.Tensor(data_y).long().to(self.device)
        # dataset = TensorDataset(tensor_x, tensor_y)
        # data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        data_loader = DataLoader(imageloader, batch_size=64, shuffle=True)

        # optimizer
        loss_function = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.SGD(self.NNet.parameters(), lr=learning_rate, )
        # optimizer = torch.optim.Adam(self.NNet.parameters(), lr=learning_rate*0.1)

        for epoch in range(0, max_epoch):
            self.NNet.train()

            loss_epoch = 0
            for X, Y in data_loader:
                self.NNet.zero_grad()
                X = X.to(self.device)
                Y = Y.to(self.device)
                pred, _ = self.NNet(X)

                loss = loss_function(pred, Y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()

            if epoch % test_epoch == 0:
                # calculate accuracy
                with torch.no_grad():
                    self.NNet.eval()

                    correct_count = 0
                    for X, Y in data_loader:
                        X = X.to(self.device)
                        Y = Y.to(self.device)
                        tensor_y_est, _ = self.NNet(X)
                        predicted_class = torch.argmax(tensor_y_est, dim=1)
                        correct_count += (predicted_class == Y).float().sum()

                    acc = correct_count.float() / len(imageloader)
                    print('%d epoch: %f loss %f acc'%(epoch, loss_epoch / len(data_loader), acc.item()))

                if acc >= stop_acc:
                    break
            else:
                print('%d epoch: %f loss' % (epoch, loss_epoch / len(data_loader)))

        self.NNet.eval()


# additional classifiers
# contains everything related to training and inference
from sklearn.mixture import GaussianMixture
import numpy as np
class GMMClassifier():
    def __init__(self, nameCategory, num_cluster, num_class, base_net='ResNet50'):
        self.nameCategory = nameCategory.replace(' ', '_')
        self.num_class = num_class
        self.num_cluster = num_cluster
        self.sharedNet = backbone.network_dict[base_net]()
        self.sharedNet.cuda()

        self.list_GMMs = []

    # save the classifier to disk
    def save(self, path_to_model):
        # torch.save({
        #     'nameCategory': self.nameCategory,
        #     # 'num_class': self.num_class,
        #     # 'num_hid_ftr': self.num_hid_ftr,
        #     # 'base_net': self.base_net,
        #     'model_state_dict': self.NNet.state_dict()
        # }, path_to_model)
        pass

    # load the classifier from disk
    def load(self, path_to_model):
        # load information and network
        ret = False

        # if os.path.exists(path_to_model):
        #     print('load model: ', path_to_model)
        #     model = torch.load(path_to_model)
        #
        #     self.nameCategory = model['nameCategory']
        #     # self.num_class = model['num_class']
        #     # self.num_hid_ftr = model['num_hid_ftr']
        #     # self.base_net = model['base_net']
        #
        #     # if self.num_class == num_class_from_files:
        #     self.NNet.load_state_dict(model['model_state_dict'])
        #     self.NNet.eval()
        #
        #     ret = True

        return ret


    def inference(self, x):
        # ftrx = self.sharedNet(x.cuda())
        ftrx = x.view(x.shape[0], -1)
        np_x = ftrx.cpu().detach().numpy()
        logprob = np.array([self.list_GMMs[i_class].score_samples(np_x) for i_class in range(self.num_class)])
        y_est = np.argmax(logprob, axis=0)
        y_prob = np.exp(logprob)
        y_prob = y_prob / sum(y_prob)

        return y_est, y_prob[y_est]

    def train(self, imageloader):
        data_loader = DataLoader(imageloader, batch_size=len(imageloader), shuffle=False)
        X, Y = next(iter(data_loader))

        # ftrX = self.sharedNet(X.cuda())
        ftrX = X.view(X.shape[0], -1)

        np_x = ftrX.cpu().detach().numpy()
        np_y = Y.cpu().detach().numpy()

        print(ftrX.shape)       # n_data, n_dim
        print(Y)                # n_data

        for i_class in range(self.num_class):
            self.list_GMMs.append(GaussianMixture(n_components=self.num_cluster, covariance_type='spherical').fit(np_x[Y==i_class, :]))

        # calculate accuracy
        logprob = np.array([self.list_GMMs[i_class].score_samples(np_x) for i_class in range(self.num_class)])
        logprob = np.transpose(logprob)
        predicted_class = np.argmax(logprob, axis=1)
        acc = (predicted_class == np_y).mean()

        print('GMM: %f acc' % acc)
