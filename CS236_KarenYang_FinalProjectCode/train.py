"""Training script.
usage: train.py [options]

options:
    --inner_learning_rate=ilr   Learning rate of inner loop [default: 1e-3]
    --outer_learning_rate=olr   Learning rate of outer loop [default: 1e-4]
    --batch_size=bs             Size of task to train with [default: 4]
    --inner_epochs=ie           Amount of meta epochs in the inner loop [default: 10]
    --height=h                  Height of image [default: 256]
    --length=l                  Length of image [default: 256]
    --dataset=ds                Dataset name (FIGR8, Karen) [default: Karen]
    --neural_network=nn         Either ResNet or DCGAN [default: DCGAN]
    -h, --help                  Show this help message and exit
"""
from docopt import docopt

import torch
import torch.optim as optim
import torch.autograd as autograd

from tensorboardX import SummaryWriter
import numpy as np
import os
from environments import KarenMetaEnv2
from model import DCGANGenerator, DCGANDiscriminator

import time, itertools
from options.train_options import TrainOptions
from models import create_model

from datetime import datetime
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def wassertein_loss(inputs, targets):
    return torch.mean(inputs * targets)


def calc_gradient_penalty(discriminator, real_batch, fake_batch):
    epsilon = torch.rand(real_batch.shape[0], 1, device=device)
    interpolates = epsilon.view(-1, 1, 1, 1) * real_batch + (1 - epsilon).view(-1, 1, 1, 1) * fake_batch
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = discriminator(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty


def normalize_data(data):
    data *= 2
    data -= 1
    return data


def unnormalize_data(data):
    data += 1
    data /= 2
    return data


class FIGR:
    def load_args(self, args):
        self.outer_learning_rate = float(args['--outer_learning_rate'])
        self.inner_learning_rate = float(args['--inner_learning_rate'])
        self.batch_size = int(args['--batch_size'])
        self.inner_epochs = int(args['--inner_epochs'])
        self.height = int(args['--height'])
        self.length = int(args['--length'])
        self.dataset = args['--dataset']
        self.neural_network = args['--neural_network']

        date_and_time = str(datetime.now()).split()
        self.date = date_and_time[0].replace('-', '_')
        self.time = date_and_time[1].replace(':', '_').split('.')[0]

        self.cycle_gan_args = [
            '--dataroot', './datasets/maps',    # Required but not used
            '--name', 'maps_cyclegan',          # ???
            '--model', 'cycle_gan',             # Required and necessary
            '--gpu_ids', '-1'                   # -1 to train on CPU
        ]
        self.cycle_gan_opts = TrainOptions().parse(self.cycle_gan_args)    # get training options
        print(self.cycle_gan_opts)
        # asdf

    def get_id_string(self):
        self.id_string = '{}_{}_olr{}_ilr{}_bsize{}_ie{}_h{}_l{}_d{}_t{}'.format(
            self.neural_network,
            self.dataset,
            str(self.outer_learning_rate),
            str(self.inner_learning_rate),
            str(self.batch_size),
            str(self.inner_epochs),
            str(self.height),
            str(self.length),
            self.date,
            self.time)
        print(self.id_string)
        return self.id_string

    def initialize_gan(self):
        # python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

        opt = self.cycle_gan_opts
        self.model = create_model(opt)           # create a model given opt.model and other options
        self.model.setup(opt)                    # regular setup: load and print networks; create schedulers

        self.meta_model = create_model(opt)      # create a model given opt.model and other options
        self.meta_model.setup(opt)               # regular setup: load and print networks; create schedulers

        #dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        #dataset_size = len(dataset)    # get the number of images in the dataset.
        #print('The number of training images = %d' % dataset_size)

        # Regular stuff
        self.da = self.model.netD_A
        self.db = self.model.netD_B
        self.ga = self.model.netG_A
        self.gb = self.model.netG_B

        self.da_optim = optim.Adam(params=self.da.parameters(), lr=self.outer_learning_rate, betas=(opt.beta1, 0.999))
        self.db_optim = optim.Adam(params=self.db.parameters(), lr=self.outer_learning_rate, betas=(opt.beta1, 0.999))
        self.ga_optim = optim.Adam(params=self.ga.parameters(), lr=self.outer_learning_rate, betas=(opt.beta1, 0.999))
        self.gb_optim = optim.Adam(params=self.gb.parameters(), lr=self.outer_learning_rate, betas=(opt.beta1, 0.999))

        self.d_optim = optim.Adam(
            params=itertools.chain(self.da.parameters(), self.db.parameters()),
            lr=self.outer_learning_rate,
            betas=(opt.beta1, 0.999),
        )
        self.g_optim = optim.Adam(
            params=itertools.chain(self.ga.parameters(), self.gb.parameters()),
            lr=self.outer_learning_rate,
            betas=(opt.beta1, 0.999),
        )

        self.model.optimizer_D = self.d_optim
        self.model.optimizer_G = self.g_optim
        self.model.optimizers  = [self.model.optimizer_G, self.model.optimizer_D]

        # Meta stuff
        self.meta_da = self.meta_model.netD_A
        self.meta_db = self.meta_model.netD_B
        self.meta_ga = self.meta_model.netG_A
        self.meta_gb = self.meta_model.netG_B

        self.meta_d_optim = optim.SGD(
            params=itertools.chain(self.meta_da.parameters(), self.meta_db.parameters()),
            lr=self.inner_learning_rate,
        )
        self.meta_g_optim = optim.SGD(
            params=itertools.chain(self.meta_ga.parameters(), self.meta_gb.parameters()),
            lr=self.inner_learning_rate,
        )
        self.meta_model.optimizer_D = self.meta_d_optim
        self.meta_model.optimizer_G = self.meta_g_optim
        self.meta_model.optimizers  = [self.meta_model.optimizer_G, self.meta_model.optimizer_D]

        # Idk stuff
        #self.discriminator_targets = torch.tensor([1] * self.batch_size + [-1] * self.batch_size, dtype=torch.float, device=device).view(-1, 1)
        #self.generator_targets = torch.tensor([1] * self.batch_size, dtype=torch.float, device=device).view(-1, 1)

    def load_checkpoint(self):
        checkpoint_file_name = 'Runs/' + self.id_string + '/checkpoint'
        if os.path.isfile(checkpoint_file_name):
            checkpoint = torch.load(checkpoint_file_name)
            self.da.load_state_dict(checkpoint['discriminator_a'])
            self.db.load_state_dict(checkpoint['discriminator_b'])
            self.ga.load_state_dict(checkpoint['generator_a'])
            self.gb.load_state_dict(checkpoint['generator_b'])
            self.eps = checkpoint['episode']
        else:
            self.eps = 0

    def __init__(self, args):
        self.load_args(args)
        self.set_id_string()
        self.initialize_gan()
        self.writer = SummaryWriter('Runs/' + self.id_string)
        self.env = eval(self.dataset + 'MetaEnv2(height=self.height, length=self.length, opt=opt)')
        #self.z_shape = 100
        self.load_checkpoint()

    def reset_meta_model(self):
        # Sets model mode to train
        self.meta_ga.train()
        self.meta_gb.train()
        self.meta_da.train()
        self.meta_db.train()

        # Transfers params from D/G to MetaD/G (=W_d/g from the paper)
        # Initialize Φd and Φg (discriminator/generator parameter vectors)
        self.meta_da.load_state_dict(self.da.state_dict())
        self.meta_db.load_state_dict(self.db.state_dict())
        self.meta_ga.load_state_dict(self.ga.state_dict())
        self.meta_gb.load_state_dict(self.gb.state_dict())

    def inner_loop(self, data_a_list, data_b_list):
        #print('.', end='')
        self.model.update_learning_rate()
        for (data_a, name_a), (data_b, name_b) in zip(data_a_list, data_b_list):
            data = dict(
                #A=normalize_data(data_a), A_paths=[name_a],
                #B=normalize_data(data_b), B_paths=[name_b],
                A=data_a, A_paths=[name_a],
                B=data_b, B_paths=[name_b],
            )
            self.model.set_input(data)         # unpack data from dataset and apply preprocessing
            self.model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

        '''
            # Training discriminator
            gradient_penalty = calc_gradient_penalty(self.meta_d, real_batch, fake_batch)
            discriminator_pred = self.meta_d(training_batch)
            discriminator_loss = wassertein_loss(discriminator_pred, self.discriminator_targets)
            discriminator_loss += gradient_penalty

            self.meta_d_optim.zero_grad()
            discriminator_loss.backward()
            self.meta_d_optim.step()

            # Training generator
            output = self.meta_d(self.meta_g(torch.tensor(np.random.normal(size=(self.batch_size, self.z_shape)), dtype=torch.float, device=device)))
            generator_loss = wassertein_loss(output, self.generator_targets)

            self.meta_g_optim.zero_grad()
            generator_loss.backward()
            self.meta_g_optim.step()

            return discriminator_loss.item(), generator_loss.item()
        '''

        # return discriminator_loss.item(), generator_loss.item()
        return self.model.loss_D_A, self.model.loss_D_B, self.model.loss_G

    def meta_training_loop(self):
        data_a_list, data_b_list, task = self.env.sample_training_task(self.batch_size)

        discriminator_a_total_loss = 0
        discriminator_b_total_loss = 0
        generator_total_loss = 0

        # for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        for _ in range(self.inner_epochs):
            da_loss, db_loss, g_loss = self.inner_loop(data_a_list, data_b_list)
            discriminator_a_total_loss += da_loss
            discriminator_b_total_loss += db_loss
            generator_total_loss += g_loss
        print('')

        self.writer.add_scalar('Training_discriminator_a_loss', discriminator_a_total_loss, self.eps)
        self.writer.add_scalar('Training_discriminator_b_loss', discriminator_b_total_loss, self.eps)
        self.writer.add_scalar('Training_generator_loss', generator_total_loss, self.eps)

        # Updating both generator and dicriminator (p = Φ_d/g from the paper)
        for p, meta_p in zip(self.ga.parameters(), self.meta_ga.parameters()):
            diff = p - meta_p.cpu()
            p.grad = diff
        self.ga_optim.step()

        for p, meta_p in zip(self.gb.parameters(), self.meta_gb.parameters()):
            diff = p - meta_p.cpu()
            p.grad = diff
        self.gb_optim.step()

        for p, meta_p in zip(self.da.parameters(), self.meta_da.parameters()):
            diff = p - meta_p.cpu()
            p.grad = diff
        self.da_optim.step()

        for p, meta_p in zip(self.db.parameters(), self.meta_db.parameters()):
            diff = p - meta_p.cpu()
            p.grad = diff
        self.db_optim.step()

    def validation_run(self):
        data_a_list, data_b_list, task = self.env.sample_validation_task(self.batch_size)
        #print('data_a_list 1:', data_a_list[0])

        discriminator_a_total_loss = 0
        discriminator_b_total_loss = 0
        generator_total_loss = 0

        for _ in range(self.inner_epochs):
            da_loss, db_loss, g_loss = self.inner_loop(
                copy.deepcopy(data_a_list), copy.deepcopy(data_b_list))
            discriminator_a_total_loss += da_loss
            discriminator_b_total_loss += db_loss
            generator_total_loss += g_loss
        print('')

        self.writer.add_scalar('Validation_discriminator_a_loss', discriminator_a_total_loss, self.eps)
        self.writer.add_scalar('Validation_discriminator_b_loss', discriminator_b_total_loss, self.eps)
        self.writer.add_scalar('Validation_generator_loss', generator_total_loss, self.eps)

        #print('data_a_list 2:', data_a_list[0])

        input_images = [t[0].cpu().numpy() for t in data_a_list] # (1, 3, 32, 32)
        training_images = [t[0].squeeze(0).cpu().numpy() for t in data_a_list] # (3, 32, 32)
        training_images = np.concatenate(training_images, axis=-1) # (3, 32, 128)

        imgs = []
        self.meta_ga.eval()
        with torch.no_grad():
            for input_image in input_images:
                tensor = torch.tensor(input_image, dtype=torch.float, device=device) # torch.Size([1, 3, 32, 32])
                img = self.meta_ga(tensor).squeeze(0) # torch.Size([3, 32, 32])
                img = img.detach().cpu().numpy()
                imgs.append(img)

        img_prediction = np.concatenate(imgs, axis=-1)
        img_unnormalized = unnormalize_data(np.concatenate(imgs, axis=-1))
        img = np.concatenate([training_images, img_prediction, img_unnormalized], axis=-2)
        self.writer.add_image('Validation_generated', img, self.eps)

    def checkpoint_model(self):
        checkpoint = {
            'discriminator_a': self.da.state_dict(),
            'discriminator_a': self.db.state_dict(),
            'generator_a': self.ga.state_dict(),
            'generator_b': self.gb.state_dict(),
            'episode': self.eps
        }
        torch.save(checkpoint, 'Runs/' + self.id_string + '/checkpoint')

    def training(self):
        #while self.eps <= 1000000:
        while self.eps <= 100:
            self.reset_meta_model()
            self.meta_training_loop()

            # Validation run every 10000 training loop
            # if self.eps % 10000 == 0:
            if self.eps % 10 == 0:
                print(f'\nValidating... (eps={self.eps})')
                self.reset_meta_model()
                self.validation_run()
                self.checkpoint_model()
                print('Validation done!')
            self.eps += 1

if __name__ == '__main__':
    args = docopt(__doc__)
    env = FIGR(args)
    env.training()
