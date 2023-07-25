def set_template(args):
    # Set the templates here
    if args.template.find('jpeg') >= 0:
        args.data_train = 'DIV2K_jpeg'
        args.data_test = 'DIV2K_jpeg'
        args.epochs = 200
        args.decay = '100'

    if args.template.find('EDSR_paper') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

    if args.template.find('MDSR') >= 0:
        args.model = 'MDSR'
        args.patch_size = 48
        args.epochs = 650

    if args.template.find('MDSR_S') >= 0:
        args.model = 'MDSR_SHIFT'
        args.patch_size = 48
        args.epochs = 650
    
    if args.template.find('CARN') >= 0:
        args.model = 'CARN'
        args.patch_size = 64
        args.epochs = 1000

    
    
    if args.template.find('CARN_S') >= 0:
        args.model = 'CARN_S'
        args.patch_size = 64
        args.epochs = 1000

    if args.template.find('DRRN') >= 0:
        args.model = 'DRRN'
        args.epochs = 600
        args.decay = '200-400-600'

    if args.template.find('DRRN_S') >= 0:
        args.model = 'DRRN_SHIFT'
        args.epochs = 600
        args.decay = '200-400-600'
        
      

    if args.template.find('DDBPN') >= 0:
        args.model = 'DDBPN'
        args.patch_size = 128
        args.scale = '4'

        args.data_test = 'Set5'

        args.batch_size = 20
        args.epochs = 1000
        args.decay = '500'
        args.gamma = 0.1
        args.weight_decay = 1e-4

        args.loss = '1*MSE'

    if args.template.find('GAN') >= 0:
        args.epochs = 200
        args.lr = 5e-5
        args.decay = '150'

    if args.template.find('RCAN') >= 0:
        args.model = 'RCAN'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64
        args.chop = True

    if args.template.find('VDSR') >= 0:
        args.model = 'VDSR'
        args.n_feats = 64
        args.lr = 1e-4

    if args.template.find('VDSR_shift') >= 0:
        args.model = 'VDSR_shift'
        args.n_feats = 64
        args.lr = 6.25e-6

    if args.template.find('SAN') >= 0:
        args.model = 'san'
        args.n_resgroups = 20
        args.n_resblocks = 10
        args.n_feats = 64

    if args.template.find('SAN_PCS') >= 0:
        args.model = 'san_pcs'
        args.n_resgroups = 20
        args.n_resblocks = 10
        args.n_feats = 64


        




