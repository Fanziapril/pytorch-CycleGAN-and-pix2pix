import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from PIL import Image
import torchvision.transforms as transforms
from .VGG_PRETRAIN import VggEncoder
import numpy as np
import pdb
###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def reset_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.normal(m.weight, 0.0, 0.02)
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, 0.0, 0.0001)
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.normal(m.weight, 1.0, 0.02)
            init.constant(m.bias, 0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[], lambda_v=0.0):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids, wvae = lambda_v)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids, wvae = lambda_v)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda()
    #netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD, k_size,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[], ):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, k_size, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, k_size, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda()
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################
class VggLoss(nn.Module):
    def __init__(self, opt):
        super(VggLoss, self).__init__()
        mean = Image.open("./models/meanimg.jpg").convert('RGB')
        translist = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        toTensor = transforms.Compose(translist)
        mean = toTensor(mean)
        mean = mean.expand(opt.batchSize, 3, 224, 224)
        self.mean = Variable(mean.cuda())
        self.VGG = VggEncoder(opt)
        self.VGG = torch.nn.DataParallel(self.VGG).cuda()
    def forward(self, input, target):
        A = torch.nn.functional.adaptive_avg_pool2d(input, output_size=224)
        B = torch.nn.functional.adaptive_avg_pool2d(target, output_size=224)
        A = self.VGG(A - self.mean)
        B = self.VGG(B - self.mean)
        return torch.nn.functional.l1_loss(A, B), self.GramianLoss(A, B)
    def GramianLoss(self, A, B):
        n, c, h, w = A.size()
        A = A.view(n, c, -1)
        B = B.view(n, c, -1)
        g_A = torch.bmm(A, A.transpose(1, 2))/(h * w)
        g_B = torch.bmm(B, B.transpose(1, 2))/(h * w)
        return torch.nn.functional.mse_loss(g_A, g_B, size_average=False) / n
# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        input_avg = torch.nn.functional.avg_pool2d(input, (input.size(2), input.size(3)))
        target_avg = torch.nn.functional.avg_pool2d(target_tensor, (target_tensor.size(2),target_tensor.size(3)))
        return self.loss(input, target_tensor)
#+self.loss(input_avg, target_avg)
        
# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], wvae = 0.0):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        self.wvae = wvae
        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        self.unetconv1 = UnetConvBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        self.unetconv2 = UnetConvBlock(ngf * 8, ngf * 8, norm_layer=norm_layer)
        self.unetconv3 = UnetConvBlock(ngf * 4, ngf * 8, norm_layer=norm_layer)
        self.unetconv4 = UnetConvBlock(ngf * 2, ngf * 4, norm_layer=norm_layer)
        self.unetconv5 = UnetConvBlock(ngf, ngf * 2, norm_layer=norm_layer)
        self.unetconv6 = UnetConvBlock(output_nc, ngf, outermost=True, norm_layer=norm_layer)
        self.unetdeconv1 = UnetdeConvBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        self.unetdeconv2 = UnetdeConvBlock(ngf * 8, ngf * 8, norm_layer=norm_layer)
        self.unetdeconv3 = UnetdeConvBlock(ngf * 4, ngf * 8, norm_layer=norm_layer)
        self.unetdeconv4 = UnetdeConvBlock(ngf * 2, ngf * 4, norm_layer=norm_layer)
        self.unetdeconv5 = UnetdeConvBlock(ngf, ngf * 2, norm_layer=norm_layer)
        self.unetdeconv6 = UnetdeConvBlock(output_nc, ngf, outermost=True, norm_layer=norm_layer)
        if self.wvae > 0:
            vae1 = UnetConvBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
            vae2 = UnetConvBlock(ngf * 8, ngf * 8, norm_layer=norm_layer)
            vae3 = UnetConvBlock(ngf * 4, ngf * 8, norm_layer=norm_layer)
            vae4 = UnetConvBlock(ngf * 2, ngf * 4, norm_layer=norm_layer)
            vae5 = UnetConvBlock(ngf, ngf * 2, norm_layer=norm_layer)
            vae6 = UnetConvBlock(output_nc, ngf, outermost=True, norm_layer=norm_layer)
            self.vae = nn.Sequential(vae6, vae5, vae4, vae3, vae2, vae1)
            self.fc = nn.Linear(2048, 512)
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(512, 128)
            self.fc2 = nn.Linear(512, 128)
            self.fc3 = nn.Linear(128, 2048)

    def forward(self, input, texture, train_or_test):
        feat6 = self.unetconv6(input)
        feat5 = self.unetconv5(feat6)
        feat4 = self.unetconv4(feat5)
        feat3 = self.unetconv3(feat4)
        feat2 = self.unetconv2(feat3)
        feat1 = self.unetconv1(feat2)
        loss_G_vae = 0
        if self.wvae > 0:
            x = self.vae(texture)
            x = x.view(-1, 2048)
            x = self.fc(x)
            x = self.relu(x)
            mu = self.fc1(x)
            logvar = self.fc2(x)
            sample = self.reparatermize(mu, logvar, train_or_test)
            #print sample
            z = self.fc3(sample)
            loss_G_vae = self.vaeloss(mu, logvar)
            feat1 = feat1 + z.view_as(feat1)
        confeat1 = self.unetdeconv1(feat1)
        confeat2 = self.unetdeconv2(torch.cat([feat2, confeat1], 1))
        confeat3 = self.unetdeconv3(torch.cat([feat3, confeat2], 1))
        confeat4 = self.unetdeconv4(torch.cat([feat4, confeat3], 1))
        confeat5 = self.unetdeconv5(torch.cat([feat5, confeat4], 1))
        confeat6 = self.unetdeconv6(torch.cat([feat6, confeat5], 1))
        return loss_G_vae, confeat6

    def reparatermize(self, mu, logvar, train_and_test):
        if train_and_test == 'train':
            # std = logvar.mul(0.5).exp_()
            std = torch.exp(0.5 * logvar)
            eps = Variable(std.data.new(std.size()).normal_())
            # print eps
            return eps.mul(std).add_(mu)
            #return std + mu
        else:
            return mu
    def vaeloss(self, mu, logvar):
        KLD = (-0.5) * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
        return KLD

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)
        reset_params(self.model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)






# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetConvBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetConvBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)

        if outermost:
            seq = [downconv]
        elif innermost:
            seq = [downrelu, downconv]
        else:
            seq = [downrelu, downconv, downnorm]

        self.model = nn.Sequential(*seq)
        reset_params(self.model)
    def forward(self, x):
        return self.model(x)

class UnetdeConvBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(UnetdeConvBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            seq = [uprelu, upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            seq = [uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            if use_dropout:
                seq = [uprelu, upconv, upnorm, nn.Dropout(0.5)]
            else:
                seq = [uprelu, upconv, upnorm]

        self.model = nn.Sequential(*seq)
    def forward(self, x):
        return self.model(x)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, k_size=4, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = k_size
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        self.feat = nn.Sequential(*sequence)
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)
        self.output = 0

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            self.feature = nn.parallel.data_parallel(self.feat, input, self.gpu_ids)
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            self.feature = self.feat(input)
            return self.model(input)
