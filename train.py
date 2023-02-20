from model import Discriminator,Generator
from reader import get_batch,one_hot
import torch as t
from torch import nn, optim
from config import get_config
import numpy as np
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import visdom
import os
from torch.autograd import Variable
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train():
    image_list = './dataset/data_list.txt'
    conf = get_config()

    vis = visdom.Visdom()
    train_loader = get_batch(image_list,conf.batch_size) #xxxxxx

    D = Discriminator(conf.nd, conf.np, 3).cuda() # 进入 __init__ 函数 整个网络结构
    G = Generator(conf.np, conf.nz, 3,conf.images_perID).cuda() # # 进入 __init__ 函数 整个网络结构
    if conf.cuda:
        D.cuda()
        G.cuda()
    D.train()
    G.train()

    optimizer_D = optim.Adam(D.parameters(),lr=conf.lr,betas=(conf.beta1,conf.beta2))
    optimizer_G = optim.Adam(G.parameters(), lr=conf.lr,betas=(conf.beta1, conf.beta2))
    loss_criterion = nn.CrossEntropyLoss()
    loss_criterion_gan = nn.BCEWithLogitsLoss()

    steps = 0
    # writer = SummaryWriter()
    flag_D_strong = False
    for epoch in range(1,conf.epochs+1):
        print('%d epoch ...'%(epoch))
        g_loss = 0
        for i, batch_data in enumerate(train_loader):
            D.zero_grad()
            G.zero_grad()
            # print(batch_data[0].dtype)
            # print(type(batch_data[0]))
            batch_image = t.FloatTensor(batch_data[0]) # Tensor:(batch_size, 3, 96, 96)
            batch_id_label = t.LongTensor(batch_data[1]) # Tensor:(batch_size,) tensor([281, 411, 74, 71])
            batch_id_label_unique = t.LongTensor(batch_id_label[::conf.images_perID]) # Tensor:(batch_size/2,) tensor([281, 74])
            batch_pose_label = t.LongTensor(batch_data[2]) # Tensor:(batch_size,) tensor([1, 0, 0, 1])
            minibatch_size = len(batch_image) # batch_size
            minibatch_size_unique = len(batch_image) // conf.images_perID

            batch_ones_label = t.ones(conf.batch_size)  # 真伪识别用标签 Tensor:(batch_size,) tensor([1., 1., 1., 1.])
            batch_zeros_label = t.zeros(conf.batch_size) # Tensor:(batch_size,) tensor([0., 0., 0., 0.])

            fixed_noise = t.FloatTensor(np.random.uniform(-1, 1, (conf.batch_size, conf.nz))) # Tensor:(batch_size, 50)
            tmp = t.LongTensor(np.random.randint(conf.np, size=conf.batch_size)) # Tensor:(batch_size,) tensor([1, 0, 0, 1]]
            pose_code = one_hot(tmp, conf.np)  # Tensor:(batch_size, conf.np) tensor([[0., 1.],\n [1., 0.],\n [1., 0.],\n [0., 1.]])
            pose_code_label = t.LongTensor(tmp) # Tensor:(batch_size,) tensor([1, 0, 0, 1]] # CrossEntropy 用于误差
            # 总结同一人物特征量时
            fixed_noise_unique = t.FloatTensor(np.random.uniform(-1, 1, (minibatch_size_unique, conf.nz))) # Tensor:(batch_size/2, 50)
            tmp = t.LongTensor(np.random.randint(conf.np, size=minibatch_size_unique)) # Tensor:(batch_size/2,) tensor([0, 0])
            pose_code_unique = one_hot(tmp, conf.np)  # Tensor:(batch_size/2, conf.np) tensor([[1., 0.],\n [1., 0.]])
            pose_code_label_unique = t.LongTensor(tmp)  # Tensor:(batch_size/2,) tensor([0, 0])


            #cuda
            if conf.cuda:
                batch_image, batch_id_label, batch_pose_label, batch_ones_label, batch_zeros_label = \
                    batch_image.cuda(), batch_id_label.cuda(), batch_pose_label.cuda(), batch_ones_label.cuda(), batch_zeros_label.cuda()

                fixed_noise, pose_code, pose_code_label = \
                    fixed_noise.cuda(), pose_code.cuda(), pose_code_label.cuda()

                batch_id_label_unique, fixed_noise_unique, pose_code_unique, pose_code_label_unique = \
                    batch_id_label_unique.cuda(), fixed_noise_unique.cuda(), pose_code_unique.cuda(), pose_code_label_unique.cuda()

            batch_image, batch_id_label, batch_pose_label, batch_ones_label, batch_zeros_label = \
                Variable(batch_image), Variable(batch_id_label), Variable(batch_pose_label), Variable(
                    batch_ones_label), Variable(batch_zeros_label)

            fixed_noise, pose_code, pose_code_label = \
                Variable(fixed_noise), Variable(pose_code), Variable(pose_code_label)

            batch_id_label_unique, fixed_noise_unique, pose_code_unique, pose_code_label_unique = \
                Variable(batch_id_label_unique), Variable(fixed_noise_unique), Variable(pose_code_unique), Variable(
                    pose_code_label_unique)

            # generated:(batch_size, 3, 96, 96)
            generated = G(batch_image, pose_code, fixed_noise,single=True) # 进forward函数 Tensor:(batch_size, 3, 96, 96)
            generated_unique = G(batch_image, pose_code_unique, fixed_noise_unique) # Tensor:(batch_size/2, 3, 96, 96)

            steps += 1

            if flag_D_strong:

                if i%5 == 0:
                    # Discriminator 学习
                    flag_D_strong = Learn_D(D, loss_criterion, loss_criterion_gan, optimizer_D, batch_image, generated, \
                                            batch_id_label, batch_pose_label, batch_ones_label, batch_zeros_label, epoch, steps, conf.nd, conf)

                else:
                    # Generator学习
                    #g_loss = Learn_G(D, loss_criterion, loss_criterion_gan, optimizer_G ,generated,\
                    #        batch_id_label, batch_ones_label, pose_code_label, epoch, steps, conf.nd, conf)
                    g_loss = Learn_G(D, loss_criterion, loss_criterion_gan, optimizer_G, generated, generated_unique,\
                            batch_id_label, pose_code_label, batch_id_label_unique, pose_code_label_unique, batch_ones_label,\
                            minibatch_size_unique, epoch, steps, conf.nd, conf)
            else:

                if i%2==0:
                    # Discriminator学习
                    flag_D_strong = Learn_D(D, loss_criterion, loss_criterion_gan, optimizer_D, batch_image, generated, \
                                            batch_id_label, batch_pose_label, batch_ones_label, batch_zeros_label, epoch, steps, conf.nd, conf)

                else:
                    # Generator学习
                    #g_loss = Learn_G(D, loss_criterion, loss_criterion_gan, optimizer_G ,generated, \
                    #        batch_id_label, batch_ones_label, pose_code_label, epoch, steps, conf.nd, conf)
                    Learn_G(D, loss_criterion, loss_criterion_gan, optimizer_G, generated, generated_unique,\
                            batch_id_label,pose_code_label, batch_id_label_unique, pose_code_label_unique, batch_ones_label,\
                            minibatch_size_unique, epoch, steps, conf.nd,conf)

            if i % 10 == 0:
                # x = vutils.make_grid(generated, normalize=True, scale_each=True)
                # writer.add_image('Image', x, i)
                generated = generated.cpu().data.numpy()/2+0.5 # ndarray:(batch_size, 3, 96, 96)
                generated_unique = generated_unique.cpu().data.numpy()/2+0.5 # ndarray:(batch_size/2, 3, 96, 96)
                batch_image = batch_image.cpu().data.numpy()/2+0.5 # ndarray:(batch_size, 3, 96, 96)
                vis.images(generated,nrow=4,win='generated')
                vis.images(generated_unique, nrow=4, win='generated_unique')
                vis.images(batch_image,nrow=4,win='original')
                # vis.line(X=np.array([steps]),Y=np.array([g_loss]),win='loss', update = 'append')
                print('%d steps loss is  %f'%(steps,g_loss))
        if epoch%5 ==0:
            msg = 'Saving checkpoint :{}'.format(epoch)    #restore from epoch+1
            print(msg)
            G_state_list = G.state_dict()
            D_state_list = D.state_dict()
            t.save({
                'epoch':epoch,
                'g_net_list':G_state_list,
                'd_net_list' :D_state_list
            },
            os.path.join(conf.save_dir,'%04d.pth'% epoch))

    # writer.close()


def Learn_D(D_model, loss_criterion, loss_criterion_gan, optimizer_D, batch_image, generated, \
            batch_id_label, batch_pose_label, batch_ones_label, batch_zeros_label, epoch, steps, Nd, args):

    real_output = D_model(batch_image) # Tensor:(batch-size, Nd+1+Np)
    syn_output = D_model(generated.detach()) # # Tensor:(batch-size, Nd+1+Np) .detach() をすることで Generatorまでの逆伝播計算省略

    # id，真伪，pose计算各自的损失
    L_id    = loss_criterion(real_output[:, :Nd], batch_id_label)
    L_gan   = loss_criterion_gan(real_output[:, Nd], batch_ones_label) + loss_criterion_gan(syn_output[:, Nd], batch_zeros_label)
    L_pose  = loss_criterion(real_output[:, Nd+1:], batch_pose_label)

    d_loss = L_gan + L_id + L_pose

    d_loss.backward()
    optimizer_D.step()
    log_learning(epoch, steps, 'D', d_loss.item(), args)

    # Discriminator の強さを判別
    flag_D_strong = Is_D_strong(real_output, syn_output, batch_id_label, batch_pose_label, Nd)

    return flag_D_strong



def Learn_G(D_model, loss_criterion, loss_criterion_gan, optimizer_G ,generated, generated_unique, batch_id_label,\
            pose_code_label, batch_id_label_unique, pose_code_label_unique, batch_ones_label, minibatch_size_unique, epoch, steps, Nd, args):

    syn_output = D_model(generated) # Tensor:(batch-size, Nd+1+Np)
    syn_output_unique = D_model(generated_unique) # Tensor:(batch-size/2, Nd+1+Np)

    # id についての出力と元画像のラベル, 真偽, poseについての出力と生成時に与えたposeコード の ロスを計算
    L_id    = loss_criterion(syn_output[:, :Nd], batch_id_label)
    L_gan   = loss_criterion_gan(syn_output[:, Nd], batch_ones_label)
    L_pose  = loss_criterion(syn_output[:, Nd+1:], pose_code_label)

    L_id_unique     = loss_criterion(syn_output_unique[:, :Nd], batch_id_label_unique)
    L_gan_unique    = loss_criterion_gan(syn_output_unique[:, Nd], batch_ones_label[:minibatch_size_unique])
    L_pose_unique   = loss_criterion(syn_output_unique[:, Nd+1:], pose_code_label_unique)

    g_loss = L_gan + L_id + L_pose + L_gan_unique + L_id_unique + L_pose_unique

    g_loss.backward()
    optimizer_G.step()
    log_learning(epoch, steps, 'G', g_loss.item(), args)

def log_learning(epoch, steps, modelname, loss, args):
    text = "EPOCH : {0}, step : {1}, {2} : {3}".format(epoch, steps, modelname, loss)
    print(text)
    with open('{}/Learning_Log.txt'.format(args.save_dir),'a') as f:
        f.write("{}\n".format(text))


def Is_D_strong(real_output, syn_output, id_label_tensor, pose_label_tensor, Nd, thresh=0.9):
    """
    # Discriminator 的正确率，如果是指定以上的正确率，则视为足够强
    # id_label_tensor = batch_id_label
    # pose_label_tensor = batch_pose_label
    """
    # t.max()返回2个tensor，第1个是每行的最大值(概率),batch_size维向量； 第二个是每行最大值的索引(ID/pose),batch_size维向量
    _, id_real_ans = t.max(real_output[:, :Nd], 1)
    _, pose_real_ans = t.max(real_output[:, Nd+1:], 1)
    _, id_syn_ans = t.max(syn_output[:, :Nd], 1)

    id_real_precision = (id_real_ans==id_label_tensor).type(t.FloatTensor).sum() / real_output.size()[0] # real_output.size() = (4,503)
    pose_real_precision = (pose_real_ans==pose_label_tensor).type(t.FloatTensor).sum() / real_output.size()[0]
    gan_real_precision = (real_output[:,Nd].sigmoid()>=0.5).type(t.FloatTensor).sum() / real_output.size()[0]
    gan_syn_precision = (syn_output[:,Nd].sigmoid()<0.5).type(t.FloatTensor).sum() / syn_output.size()[0]

    total_precision = (id_real_precision+pose_real_precision+gan_real_precision+gan_syn_precision)/4

    # Variable(FloatTensor) -> Float 转换为
    total_precision = total_precision.data.item()
    if total_precision>=thresh:
        flag_D_strong = True
    else:
        flag_D_strong = False

    return flag_D_strong


if __name__=='__main__':
    train()