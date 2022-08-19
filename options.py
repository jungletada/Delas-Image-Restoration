import os
import torch


def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_model(opt, epoch, name):
    load_epoch = '/' + str(epoch) + '-'
    return opt.save_models_dir + load_epoch + name


def load_model_ckpt(model, epoch, opt, name):
    ckpt = torch.load(get_model(opt, epoch, name))
    model.load_state_dict(ckpt['model_state_dict'])


def save_model_ckpt(model, epoch, opt, name):
    state = {'model_state_dict': model.state_dict()}
    torch.save(state, get_model(opt, epoch, name))


def load_net(model, model_path):
    model.load_state_dict(torch.load(model_path), strict=True)


def save_net(model, save_path):
    torch.save(model.state_dict(), save_path)


class Option:
    def __init__(self, use_cuda, channel=3):
        self.sigma_test = 25
        self.batch_size = 16
        self.lambda_ = 100
        self.lr = 1e-6
        self.stride = 128
        self.beta1 = 0.0
        self.beta2 = 0.99
        self.workers = 4
        self.iter = 8
        self.channel = channel
        self.name_A = str(channel) + 'nc-A.pth'
        self.name_D = str(channel) + 'nc-D.pth'

        self.root = '../testsets/'
        self.test_root = '../testsets/Kodak'
        self.save_img_dir = './denoise_result/'
        self.save_models_dir = './denoise_models/'

        create_path(self.save_img_dir)
        create_path(self.save_models_dir)

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
