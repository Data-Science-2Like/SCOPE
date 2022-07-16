from typing import List

import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from torch.autograd import Variable
from gensim.models.keyedvectors import KeyedVectors

from Prefetcher.prefetcher import Prefetcher
from Prefetcher.condition import _check_conditions, ConditionList, PretrainedWordEmbeddingCondition,CategoricalCondition
from Prefetcher.citeworth import load_dataset

import scipy.sparse as sp

W2V_PATH = Path("./vectors/GoogleNews-vectors-negative300.bin.gz")
W2V_IS_BINARY = True

print("Loading keyed vectors")
VECTORS = KeyedVectors.load_word2vec_format(str(W2V_PATH), binary=W2V_IS_BINARY)
print("Done")

def sample_categorical(size):
    batch_size, n_classes = size
    cat = np.random.randint(0, n_classes, batch_size)
    cat = np.eye(n_classes)[cat].astype('float32')
    cat = torch.from_numpy(cat)
    return cat


def sample_bernoulli(size):
    ber = np.random.randint(0, 1, size).astype('float32')
    return torch.from_numpy(ber)

PRIOR_SAMPLERS = {
    'categorical': sample_categorical,
    'bernoulli': sample_bernoulli,
    'gauss': torch.randn
}

PRIOR_ACTIVATIONS = {
    'categorical': 'softmax',
    'bernoulli': 'sigmoid',
    'gauss': 'linear'
}

TORCH_OPTIMIZERS = {
    'sgd': optim.SGD,
    'adam': optim.Adam
}

AE_PARAMS = {
    'n_code': 50,
    'n_epochs': 20,
    'batch_size': 5000,
    'n_hidden': 100,
    'normalize_inputs': True,
    'gen_lr' : 0.001,
    'reg_lr' : 0.001
}

CONDITIONS = ConditionList([
    ('title', PretrainedWordEmbeddingCondition(VECTORS)),
    #('section_title', CategoricalCondition(embedding_dim=32, reduce='sum', sparse=False, embedding_on_gpu=True))
])


TINY = 1e-12

class AAERecommender(Prefetcher):
    def __init__(self, model_path: str, use_section_info):
        super().__init__()
        self.use_section_info = use_section_info
        self.conditions = CONDITIONS
        self.model_params = None
        self.model = AdversarialAutoEncoder(conditions=self.conditions, **AE_PARAMS)
        self.model.load_model(model_path)
        bags, x_train = load_dataset(2019, 2018, 2)
        self.bags = bags
        print(self.model)

    def __str__(self):

        desc = "Adversarial Autoencoder"

        if self.conditions:
            desc += " conditioned on: " + ', '.join(self.conditions.keys())
        desc += '\nModel Params: ' + str(self.model_params)
        return desc

    def predict(self, already_cited: List[str], section: str, k: int) -> List[str]:

        # transform into vocab index for aae recommender
        #internal_q = [[self.bags.vocab[id] for id in already_cited]]
        internal_q = []
        not_found = []
        for id in already_cited:
            if id in self.bags.vocab.keys():
                internal_q.append(self.bags.vocab[id])
            else:
                not_found.append(id)

        if len(not_found) > 0:
            print(f"Warning: Could not find the following cited keys: {not_found}")

        pred = self._predict([internal_q])[0]


        # sort predictions by their score
        preds_sorted = sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)

        # get keys for predictions index
        return_keys = [self.bags.index2token[i] for i in preds_sorted[:k]]

        return return_keys


    def _predict(self, test_set):
        ### DONE Adapt to generic condition ###
        #X = test_set.tocsr()
        X = self.bags.tocsr(test_set)
        if self.conditions:
            condition_data_raw = self.bags.get_attributes(self.conditions.keys(), test_set[0])
            condition_data_raw = []
            # Important to not call fit here, but just transform
            condition_data = self.conditions.fit_transform(condition_data_raw)
        else:
            condition_data = None

        pred = self.model.predict(X, condition_data=condition_data)
        return pred

    def load_model(self, folder='Prefetcher', filename='test'):
        self.model.load_model(folder, filename)

class Encoder(nn.Module):
    """ Three-layer Encoder """

    def __init__(self, n_input, n_hidden, n_code, final_activation=None,
                 normalize_inputs=True, dropout=(.2, .2), activation='ReLU'):
        super(Encoder, self).__init__()
        self.lin1 = nn.Linear(n_input, n_hidden)
        self.act1 = getattr(nn, activation)()
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.act2 = getattr(nn, activation)()
        if activation == 'SELU':
            self.drop1 = nn.AlphaDropout(dropout[0])
            self.drop2 = nn.AlphaDropout(dropout[1])
        else:
            self.drop1 = nn.Dropout(dropout[0])
            self.drop2 = nn.Dropout(dropout[1])
        self.lin3 = nn.Linear(n_hidden, n_code)
        self.normalize_inputs = normalize_inputs
        if final_activation == 'linear' or final_activation is None:
            self.final_activation = None
        elif final_activation == 'softmax':
            self.final_activation = nn.Softmax(dim=1)
        elif final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        else:
            raise ValueError("Final activation unknown:", activation)

    def forward(self, inp):
        """ Forward method implementation of 3-layer encoder """
        if self.normalize_inputs:
            inp = F.normalize(inp, 1)
        # first layer
        act = self.lin1(inp)
        act = self.drop1(act)
        act = self.act1(act)
        # second layer
        act = self.lin2(act)
        act = self.drop2(act)
        act = self.act2(act)
        # third layer
        act = self.lin3(act)
        if self.final_activation:
            act = self.final_activation(act)
        return act


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, n_code, n_hidden, n_output, dropout=(.2, .2), activation='ReLU'):
        super(Decoder, self).__init__()
        self.lin1 = nn.Linear(n_code, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.lin3 = nn.Linear(n_hidden, n_output)
        if activation == 'SELU':
            self.drop1 = nn.AlphaDropout(dropout[0])
            self.drop2 = nn.AlphaDropout(dropout[1])
        else:
            self.drop1 = nn.Dropout(dropout[0])
            self.drop2 = nn.Dropout(dropout[1])
        self.act1 = getattr(nn, activation)()
        self.act2 = getattr(nn, activation)()

    def forward(self, inp):
        """ Forward implementation of 3-layer decoder """
        # first layer
        act = self.lin1(inp)
        act = self.drop1(act)
        act = self.act1(act)
        # second layer
        act = self.lin2(act)
        act = self.drop2(act)
        act = self.act2(act)
        # final layer
        act = self.lin3(act)
        act = torch.sigmoid(act)
        return act


class Discriminator(nn.Module):
    """ Discriminator """

    def __init__(self, n_code, n_hidden, dropout=(.2, .2), activation='ReLU'):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(n_code, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.lin3 = nn.Linear(n_hidden, 1)
        if activation == 'SELU':
            self.drop1 = nn.AlphaDropout(dropout[0])
            self.drop2 = nn.AlphaDropout(dropout[1])
        else:
            self.drop1 = nn.Dropout(dropout[0])
            self.drop2 = nn.Dropout(dropout[1])
        self.act1 = getattr(nn, activation)()
        self.act2 = getattr(nn, activation)()

    def forward(self, inp):
        """ Forward of 3-layer discriminator """
        act = self.lin1(inp)
        act = self.drop1(act)
        act = self.act1(act)

        act = self.lin2(act)
        act = self.drop2(act)
        act = self.act2(act)

        return torch.sigmoid(self.lin3(act))

class AdversarialAutoEncoder():
    """ Adversarial Autoencoder """

    ### DONE Adapt to generic condition ###
    def __init__(self,
                 n_hidden=100,
                 n_code=50,
                 gen_lr=0.001,
                 reg_lr=0.001,
                 prior='gauss',
                 prior_scale=None,
                 batch_size=100,
                 n_epochs=500,
                 optimizer='adam',
                 normalize_inputs=True,
                 activation='ReLU',
                 dropout=(.2, .2),
                 conditions=None,
                 verbose=True,
                 eval_each=False,
                 eval_cb=(lambda m: print('Empty'))):
        # Build models
        self.prior = prior.lower()
        self.prior_scale = prior_scale
        self.eval_each = eval_each
        self.eval_cb = eval_cb

        # Encoder final activation depends on prior distribution
        self.prior_sampler = PRIOR_SAMPLERS[self.prior]
        self.encoder_activation = PRIOR_ACTIVATIONS[self.prior]
        self.optimizer = optimizer.lower()

        self.n_hidden = n_hidden
        self.n_code = n_code
        self.gen_lr = gen_lr
        self.reg_lr = reg_lr
        self.batch_size = batch_size
        self.verbose = verbose
        self.gen_lr, self.reg_lr = gen_lr, reg_lr
        self.n_epochs = n_epochs

        self.normalize_inputs = normalize_inputs

        self.dropout = dropout
        self.activation = activation

        self.conditions = conditions

        self.prediction_size = 107774

        self.use_condition = True

        if self.use_condition:
            code_size = self.n_code + self.conditions.size_increment()
            print("Using condition, code size:", code_size)
        else:
            code_size = self.n_code
            print("Not using condition, code size:", code_size)

        self.enc = Encoder(self.prediction_size, self.n_hidden, self.n_code,
                           final_activation=self.encoder_activation,
                           normalize_inputs=self.normalize_inputs,
                           activation=self.activation,
                           dropout=self.dropout)
        self.dec = Decoder(code_size, self.n_hidden, self.prediction_size,
                           activation=self.activation, dropout=self.dropout)

        self.disc = Discriminator(self.n_code, self.n_hidden,
                                  dropout=self.dropout,
                                  activation=self.activation)

        if torch.cuda.is_available():
            self.enc = self.enc.cuda()
            self.dec = self.dec.cuda()
            self.disc = self.disc.cuda()
        optimizer_gen = TORCH_OPTIMIZERS[self.optimizer]
        # Reconstruction
        self.enc_optim = optimizer_gen(self.enc.parameters(), lr=self.gen_lr)
        self.dec_optim = optimizer_gen(self.dec.parameters(), lr=self.gen_lr)
        # Regularization
        self.gen_optim = optimizer_gen(self.enc.parameters(), lr=self.reg_lr)
        self.disc_optim = optimizer_gen(self.disc.parameters(), lr=self.reg_lr)


    def __str__(self):
        desc = "Adversarial Autoencoder"
        n_h, n_c = self.n_hidden, self.n_code
        gen, reg = self.gen_lr, self.reg_lr
        desc += " ({}, {}, {}, {}, {})".format(n_h, n_h, n_c, n_h, n_h)
        desc += " optimized by " + self.optimizer
        desc += " with learning rates Gen, Reg = {}, {}".format(gen, reg)
        desc += ", using a batch size of {}".format(self.batch_size)
        desc += "\nMatching the {} distribution".format(self.prior)
        desc += " by {} activation.".format(self.encoder_activation)
        if self.conditions:
            desc += "\nConditioned on " + ', '.join(self.conditions.keys())
        return desc

    def eval(self):
        """ Put all NN modules into eval mode """
        ### DONE Adapt to generic condition ###
        self.enc.eval()
        self.dec.eval()
        self.disc.eval()
        if self.conditions:
            # Forward call to condition modules
            self.conditions.eval()

    def zero_grad(self):
        """ Zeros gradients of all NN modules """
        self.enc.zero_grad()
        self.dec.zero_grad()
        self.disc.zero_grad()

    def load_model(self, filepath):
        #filepath = os.path.join(folder, filename)
        state = torch.load(filepath)
        self.enc.load_state_dict(state['enc'])
        self.dec.load_state_dict(state['dec'])
        self.disc.load_state_dict(state['disc'])

    def ae_step(self, batch, condition_data=None):
        ### DONE Adapt to generic condition ###
        """
        # why is this double? to AdversarialAutoEncoder => THe AE Step is very different from plain AEs
        # what is relationship to train?
        # Condition is used explicitly here, and hard coded but non-explicitly here
        Perform one autoencoder training step
        :param batch:
        :param condition: ??? ~ training_set.get_single_attribute("title") <~ side_info = unpack_playlists(playlists)
        :return:
        """
        z_sample = self.enc(batch)
        use_condition = _check_conditions(self.conditions, condition_data)
        if use_condition:
            z_sample = self.conditions.encode_impose(z_sample, condition_data)

        x_sample = self.dec(z_sample)
        recon_loss = F.binary_cross_entropy(x_sample + TINY,
                                            batch.view(batch.size(0),
                                                       batch.size(1)) + TINY)
        # Clear all related gradients
        self.enc.zero_grad()
        self.dec.zero_grad()
        if use_condition:
            self.conditions.zero_grad()

        # Compute gradients
        recon_loss.backward()

        # Update parameters
        self.enc_optim.step()
        self.dec_optim.step()
        if use_condition:
            self.conditions.step()

        return recon_loss.data.item()

    def disc_step(self, batch):
        """ Perform one discriminator step on batch """
        self.enc.eval()
        z_real = Variable(self.prior_sampler((batch.size(0), self.n_code)))
        if self.prior_scale is not None:
            z_real = z_real * self.prior_scale

        if torch.cuda.is_available():
            z_real = z_real.cuda()
        z_fake = self.enc(batch)

        # Compute discrimnator outputs and loss
        disc_real_out, disc_fake_out = self.disc(z_real), self.disc(z_fake)
        disc_loss = -torch.mean(torch.log(disc_real_out + TINY)
                                + torch.log(1 - disc_fake_out + TINY))
        self.disc_optim.zero_grad()
        disc_loss.backward()
        self.disc_optim.step()

        return disc_loss.data.item()

    def gen_step(self, batch):
        self.enc.train()
        z_fake_dist = self.enc(batch)
        disc_fake_out = self.disc(z_fake_dist)
        gen_loss = -torch.mean(torch.log(disc_fake_out + TINY))
        self.gen_optim.zero_grad()
        gen_loss.backward()
        self.gen_optim.step()

        return gen_loss.data.item()

    def predict(self, X, condition_data=None):
        ### DONE Adapt to generic condition ###
        self.eval()  # Deactivate dropout
        # In case some of the conditions has dropout
        use_condition = _check_conditions(self.conditions, condition_data)
        if self.conditions:
            self.conditions.eval()
        pred = []
        with torch.no_grad():
            for start in range(0, X.shape[0], self.batch_size):
                end = start + self.batch_size
                # batched predictions, yet inclusive
                X_batch = X[start:(start + self.batch_size)]
                if sp.issparse(X_batch):
                    X_batch = X_batch.toarray()
                X_batch = torch.FloatTensor(X_batch)
                if torch.cuda.is_available():
                    X_batch = X_batch.cuda()

                if use_condition:
                    c_batch = [c[start:end] for c in condition_data]
                # reconstruct
                z = self.enc(X_batch)
                if use_condition:
                    # z = torch.cat((z, c_batch), 1)
                    z = self.conditions.encode_impose(z, c_batch)
                X_reconstuction = self.dec(z)
                # shift
                X_reconstuction = X_reconstuction.data.cpu().numpy()
                pred.append(X_reconstuction)
        return np.vstack(pred)

if __name__ == '__main__':

    prefetcher = AAERecommender('trained/aae.torch',False)

    bags, x_train = load_dataset(2019,2018,2)


    print(prefetcher)