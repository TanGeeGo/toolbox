import functools
import torch
from torch.nn import init


"""
# --------------------------------------------
# select the network of G, D and F
# --------------------------------------------
"""


# --------------------------------------------
# Generator, netG, G
# --------------------------------------------
def define_G(opt):
    opt_net = opt['netG']
    net_type = opt_net['net_type']


    # ----------------------------------------
    # denoising task
    # ----------------------------------------

    # ----------------------------------------
    # RRDB
    # ----------------------------------------
    
    if net_type == 'rrdb':
        from models.network_rrdb import RRDB as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   gc=opt_net['gc'],
                   upscale=opt_net['scale'],
                   act_mode=opt_net['act_mode'],
                   upsample_mode=opt_net['upsample_mode'])

    # ----------------------------------------
    # RRDBNet
    # ----------------------------------------
    elif net_type == 'rrdbnet':  # RRDB
        from models.network_rrdbnet import RRDBNet as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   gc=opt_net['gc'],
                   sf=opt_net['scale'])
        
    # ----------------------------------------
    # UNet
    # ----------------------------------------
    elif net_type == 'unet':
        from models.network_unet import UNetRes as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nb=opt_net['nb'],
                   act_mode=opt_net['act_mode'])
    
    # ----------------------------------------
    # MIMO-UNet
    # ----------------------------------------
    elif net_type == 'mimounet':
        from models.network_mimounet import MIMOUNet as net
        netG = net(num_res=opt_net['nb'], base_channel=opt_net['nc'])
    
    elif net_type == 'mimounetplus':
        from models.network_mimounet import MIMOUNetPlus as net
        netG = net(num_res=opt_net['nb'], base_channel=opt_net['nc'])

    elif net_type == 'mimounet-mff':
        from models.network_mimounet import MIMOUNetMFF as net
        netG = net(num_res=opt_net['nb'], base_channel=opt_net['nc'])
    
    # ----------------------------------------
    # MPRNet
    # ----------------------------------------
    elif net_type == 'mprnet':
        from models.network_mprnet import MPRNet as net
        netG = net(in_c=opt_net['in_nc'],
                      out_c=opt_net['out_nc'])

    # ----------------------------------------
    # NAFNet
    # ----------------------------------------
    elif net_type == 'nafnet':
        from models.network_nafnet import NAFNet as net
        netG = net(img_channel=opt_net['in_nc'],
                   width=opt_net['nc'],
                   middle_blk_num=1,
                   enc_blk_nums=opt_net['enc_blk_nums'],
                   dec_blk_nums=opt_net['dec_blk_nums']
                   )
    elif net_type == 'nafnetlocal':
        from models.network_nafnet import NAFNetLocal as net
        netG = net(img_channel=opt_net['in_nc'],
                   width=opt_net['nc'],
                   middle_blk_num=1,
                   enc_blk_nums=opt_net['enc_blk_nums'],
                   dec_blk_nums=opt_net['dec_blk_nums'],
                   train_size=(1, 3, opt['datasets']['train']['H_size'], opt['datasets']['train']['H_size'])
                   )

    # ----------------------------------------
    # Restormer
    # ----------------------------------------
    elif net_type == 'restormer':
        from models.network_restormer import Restormer as net
        netG = net(inp_channels=opt_net['in_nc'],
                   out_channels=opt_net['out_nc'],
                   dim=opt_net['nc'],
                   num_blocks=opt_net['nb']
                   )
    
    # ----------------------------------------
    # Stripformer
    # ----------------------------------------
    elif net_type == 'stripformer':
        from models.network_stripformer import Stripformer as net
        netG = net() 

    # ----------------------------------------
    # FSANet
    # ----------------------------------------
    elif net_type == 'fsanet':
        from models.network_fsanet import FSANet as net
        netG = net(input_channel=opt_net['in_nc'],
                   output_channel=opt_net['out_nc'],
                   base_channel=opt_net['nc'],
                   kernel_num=opt_net['kr_num'],
                   sig_num=opt_net['sig_num']
                   )

    # ----------------------------------------
    # uformer
    # ----------------------------------------
    elif net_type == 'uformer':
        from models.network_uformer import Uformer as net
        netG = net(in_chans=opt_net['in_nc'],
                   dd_in=opt_net['in_nc'],
                   embed_dim=opt_net['nc'], 
                   depths=opt_net['nb'],
                   token_projection='linear',
                   token_mlp=opt_net['token_mlp'],
                   modulator=True
                   )
               
    # ----------------------------------------
    # vapsr
    # ----------------------------------------
    elif net_type == 'vapsr':
        from models.network_vapsr import vapsr as net
        netG = net(num_in_ch=opt_net['in_nc'], 
                   num_out_ch=opt_net['out_nc'])

    # ----------------------------------------
    # others
    # ----------------------------------------
    # TODO

    else:
        raise NotImplementedError('netG [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    if opt['is_train']:
        init_weights(netG,
                     init_type=opt_net['init_type'],
                     init_bn_type=opt_net['init_bn_type'],
                     gain=opt_net['init_gain'])

    return netG


# --------------------------------------------
# Discriminator, netD, D
# --------------------------------------------
def define_D(opt):
    opt_net = opt['netD']
    net_type = opt_net['net_type']

    # ----------------------------------------
    # discriminator_vgg_96
    # ----------------------------------------
    if net_type == 'discriminator_vgg_96':
        from models.network_discriminator import Discriminator_VGG_96 as discriminator
        netD = discriminator(in_nc=opt_net['in_nc'],
                             base_nc=opt_net['base_nc'],
                             ac_type=opt_net['act_mode'])

    # ----------------------------------------
    # discriminator_vgg_128
    # ----------------------------------------
    elif net_type == 'discriminator_vgg_128':
        from models.network_discriminator import Discriminator_VGG_128 as discriminator
        netD = discriminator(in_nc=opt_net['in_nc'],
                             base_nc=opt_net['base_nc'],
                             ac_type=opt_net['act_mode'])

    # ----------------------------------------
    # discriminator_vgg_192
    # ----------------------------------------
    elif net_type == 'discriminator_vgg_192':
        from models.network_discriminator import Discriminator_VGG_192 as discriminator
        netD = discriminator(in_nc=opt_net['in_nc'],
                             base_nc=opt_net['base_nc'],
                             ac_type=opt_net['act_mode'])

    # ----------------------------------------
    # discriminator_vgg_128_SN
    # ----------------------------------------
    elif net_type == 'discriminator_vgg_128_SN':
        from models.network_discriminator import Discriminator_VGG_128_SN as discriminator
        netD = discriminator()

    elif net_type == 'discriminator_patchgan':
        from models.network_discriminator import Discriminator_PatchGAN as discriminator
        netD = discriminator(input_nc=opt_net['in_nc'],
                             ndf=opt_net['base_nc'],
                             n_layers=opt_net['n_layers'],
                             norm_type=opt_net['norm_type'])

    elif net_type == 'discriminator_unet':
        from models.network_discriminator import Discriminator_UNet as discriminator
        netD = discriminator(input_nc=opt_net['in_nc'],
                             ndf=opt_net['base_nc'])

    else:
        raise NotImplementedError('netD [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    init_weights(netD,
                 init_type=opt_net['init_type'],
                 init_bn_type=opt_net['init_bn_type'],
                 gain=opt_net['init_gain'])

    return netD


# --------------------------------------------
# VGGfeature, netF, F
# --------------------------------------------
def define_F(opt, use_bn=False):
    device = torch.device('cuda' if opt['gpu_ids'] else 'cpu')
    from models.network_feature import VGGFeatureExtractor
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = VGGFeatureExtractor(feature_layer=feature_layer,
                               use_bn=use_bn,
                               use_input_norm=True,
                               device=device)
    netF.eval()  # No need to train, but need BP to input
    return netF


"""
# --------------------------------------------
# weights initialization
# --------------------------------------------
"""


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    # Args:
    #   init_type:
    #       default, none: pass init_weights
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    if init_type not in ['default', 'none']:
        # print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        # print('Pass this initialization! Initialization was done during network definition!')
        pass
