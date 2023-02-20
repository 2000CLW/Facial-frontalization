from easydict import EasyDict as edict

def get_config():
    conf = edict()
    conf.batch_size = 4
    conf.lr = 0.0002
    conf.beta1 = 0.5
    conf.beta2 = 0.999
    conf.epochs = 10
    conf.save_dir = './saved_model'
    conf.np = 2
    conf.nz = 50
    conf.nd = 500
    conf.images_perID = 2
    conf.cuda = True

    return conf
