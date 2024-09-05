import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle

from loss import crossview_contrastive_Loss,mutual_info,H_12   
import evaluation
from util import next_batch
import util
import sys
import math
import numpy as np




# 修改，更换网络结构
class IAMOD():
    """IAMOD module."""

    def __init__(self,
                 config):
        """Constructor.

        Args:
          config: parameters defined in configure.py.
        """
        self._config = config
        # 检查表征维度是不是对齐的
        if self._config['Autoencoder']['arch1'][-1] != self._config['Autoencoder']['arch2'][-1]:
            raise ValueError('Inconsistent latent dim!')

        self._latent_dim = config['Autoencoder']['arch1'][-1]


        # 根据config构造  --> 各视角自编码器
        self.autoencoder_list = []
        for i in range(config['training']['num_view']):
            autoencoder_i = Autoencoder(config['Autoencoder']['arch'+str(i+1)], config['Autoencoder']['activations'+str(i+1)],
                                        config['Autoencoder']['batchnorm'])
            self.autoencoder_list.append(autoencoder_i)

        self.prediction_list = []
        for i in range(config['training']['num_view']):
            predictor = Prediction(config['Prediction']['p'+str(i+1)])
            self.prediction_list.append(predictor)


    def to_device(self, device):
        """ to cuda if gpu is used """

        for autoencoder_v in self.autoencoder_list:
            autoencoder_v.to(device)
        for predictor_v in self.prediction_list:
            predictor_v.to(device)
   

    def train(self, config, logger,X_list, Y_list, optimizer, device):
        """Training the model.

            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              X_list: 所有视角的数据
              Y_list: labels
              
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              模型表现auc


        """

        
        # end = False
        aucs = []
        for epoch in range(config['training']['epoch']):

            if len(X_list)==6:
                X_shuffle = shuffle(X_list[0],X_list[1],X_list[2],X_list[3],X_list[4],X_list[5])

            # 一个epoch的累加指标
            loss_all, loss_rec, loss_cl, loss_pre = 0, 0, 0, 0
            # pre_loss = sys.maxsize
            for batch_xv, batch_No in next_batch(X_shuffle, config['training']['batch_size']):
                z_v_list = []
                for i in range(len(batch_xv)):
                    z_v = self.autoencoder_list[i].encoder(batch_xv[i])
                    # z_v = self.autoencoder1.encoder(batch_xv[i])
                    z_v_list.append(z_v)

                # 重构损失的系数设置为0即可忽略该项
                recon_list = []
                reconstruction_loss = 0
                for i in range(config['training']['num_view']):
                    recon = F.mse_loss(self.autoencoder_list[i].decoder(z_v_list[i]),batch_xv[i])
                    reconstruction_loss+= recon
                    recon_list.append(recon)


                # Cross-view Contrastive_Loss
                # 改写成任意视角
                cl_loss = 0
                for i in range(config['training']['num_view']):
                    for j in range(i+1,config['training']['num_view']):
                        cl_loss_i = crossview_contrastive_Loss(z_v_list[i], z_v_list[j], config['training']['alpha'])
                        cl_loss+=cl_loss_i

                dualprediction_loss = 0
                for i in range(config['training']['num_view']):
                    if (i==config['training']['num_view']-1):
                        nextview,_ = self.prediction_list[i](z_v_list[i])
                        pre_i = F.mse_loss(nextview, z_v_list[0])
                    else:
                        nextview,_ = self.prediction_list[i](z_v_list[i])
                        pre_i = F.mse_loss(nextview, z_v_list[i+1])
                    dualprediction_loss+= pre_i

                # 重构损失系数config['training']['lambda_recon']可设置为0
                loss = config['training']['lambda_cl']*cl_loss + reconstruction_loss * config['training']['lambda_recon']


                if epoch >= config['training']['start_dual_prediction']:
                    loss += dualprediction_loss * config['training']['lambda_dual']



                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_all += loss.item()
                loss_rec += reconstruction_loss.item()
                # loss_rec1 += recon1.item()
                # loss_rec2 += recon2.item()
                loss_pre += dualprediction_loss.item()
                loss_cl += cl_loss.item()



            # 打印一个epoch的指标
            if (epoch + 1) % config['print_num'] == 0:
                output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f} " \
                         "===> Dual prediction loss = {:.4f}  ===> Contrastive loss = {:.4e} ===> Loss = {:.4e}" \
                    .format((epoch + 1), config['training']['epoch'], loss_rec, loss_pre, loss_cl, loss_all)

                logger.info("\033[2;29m" + output + "\033[0m")
            

            # 每100epoch显示一次
            auc = self.evaluation(config, logger, X_list, Y_list, device)
            aucs.append(auc)
            # self.writer.add_scalar('auc:',auc,epoch)

        
        return np.max(np.array(aucs))

    def mine_train(self, config, logger,X_list, Y_list, optimizer, device):
        """使用mine估计两个随机变量的互信息。

            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              x1_train: data of view 1
              x2_train: data of view 2
              Y_list: labels
              mask: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              clustering performance: acc, nmi ,ari


        """

        aucs = []
        for epoch in range(config['training']['epoch']):

            if len(X_list)==6:
                X_shuffle = shuffle(X_list[0],X_list[1],X_list[2],X_list[3],X_list[4],X_list[5])
            if len(X_list)==5:
                X_shuffle = shuffle(X_list[0],X_list[1],X_list[2],X_list[3],X_list[4])
            if len(X_list)==4:
                X_shuffle = shuffle(X_list[0],X_list[1],X_list[2],X_list[3])
            if len(X_list)==3:
                X_shuffle = shuffle(X_list[0],X_list[1],X_list[2])      
            if len(X_list)==2:
                X_shuffle = shuffle(X_list[0],X_list[1])    
          
            # 损失 （一个epoch的累计）
            loss_all, loss_rec, loss_cl, loss_pre = 0, 0, 0, 0
            H_epoch, MI_epoch = 0, 0
            for batch_xv, batch_No in next_batch(X_shuffle, config['training']['batch_size']):
                # 前向传播
                z_v_list = []
                for i in range(len(batch_xv)):
                    z_v = self.autoencoder_list[i].encoder(batch_xv[i])
                    z_v_list.append(z_v)


                # 计算重构loss
                recon_list = []
                reconstruction_loss = 0
                for i in range(config['training']['num_view']):
                    recon = F.mse_loss(self.autoencoder_list[i].decoder(z_v_list[i]),batch_xv[i])   # TODO F.mse_loss用法
                    reconstruction_loss+= recon
                    recon_list.append(recon)


                # 对比loss   (一个batch)
                cl_loss = 0
                mi = 0
                H = 0
                
                for i in range(config['training']['num_view']):
                    for j in range(i+1,config['training']['num_view']):
                        cl_loss_i = crossview_contrastive_Loss(z_v_list[i], z_v_list[j], config['training']['alpha'])
                        cl_loss+=cl_loss_i
                        mi_v = mutual_info(z_v_list[i], z_v_list[j])
                        mi += mi_v
                        H12_v = H_12(z_v_list[i], z_v_list[j])
                        H += H12_v



                # 跨视角预测     (一个batch)
                # 改写成环预测  
                dualprediction_loss = 0
                for i in range(config['training']['num_view']):
                    if (i==config['training']['num_view']-1):
                        nextview,_ = self.prediction_list[i](z_v_list[i])
                        pre_i = F.mse_loss(nextview, z_v_list[0])
                    else:
                        nextview,_ = self.prediction_list[i](z_v_list[i])
                        pre_i = F.mse_loss(nextview, z_v_list[i+1])
                    dualprediction_loss+= pre_i


                loss = config['training']['lambda_cl']*cl_loss + reconstruction_loss * config['training']['lambda_recon']

                # ----开始进行跨视角预测, 不一致信息剪枝
                if epoch >= config['training']['start_dual_prediction']:
                    loss += dualprediction_loss * config['training']['lambda_dual']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 监控loss
                loss_all += loss.item()
                loss_rec += reconstruction_loss.item()
                loss_pre += dualprediction_loss.item()
                loss_cl += cl_loss.item()
                H_epoch += H.item()
                MI_epoch += mi.item()

            self.writer.add_scalar('cl_loss',loss_cl,epoch)
            self.writer.add_scalar('reconstruction_loss',loss_rec,epoch)
            self.writer.add_scalar('dualprediction_loss',loss_pre,epoch)
            self.writer.add_scalar('all_loss',loss_all,epoch)  
           
            self.writer.add_scalar('H',H_epoch,epoch)  
            self.writer.add_scalar('MI_epoch',MI_epoch,epoch) 

            # 打印一个epoch的指标
            if (epoch + 1) % config['print_num'] == 0:
                output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f} " \
                         "===> Dual prediction loss = {:.4f}  ===> Contrastive loss = {:.4e} ===> Loss = {:.4e}" \
                    .format((epoch + 1), config['training']['epoch'], loss_rec, loss_pre, loss_cl, loss_all)

                logger.info("\033[2;29m" + output + "\033[0m")
            

            # 评估
            auc = self.evaluation(config, logger, X_list, Y_list, device)
            aucs.append(auc)
            self.writer.add_scalar('AUC:',auc,epoch)

        
        return np.max(np.array(aucs))

    def info_train(self, config, logger,X_list, Y_list, optimizer, device):
        """信息监测的训练.

            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              x1_train: data of view 1
              x2_train: data of view 2
              Y_list: labels
              mask: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              clustering performance: acc, nmi ,ari


        """

        aucs = []
        for epoch in range(config['training']['epoch']):

            if len(X_list)==6:
                X_shuffle = shuffle(X_list[0],X_list[1],X_list[2],X_list[3],X_list[4],X_list[5])
            if len(X_list)==5:
                X_shuffle = shuffle(X_list[0],X_list[1],X_list[2],X_list[3],X_list[4])
            if len(X_list)==4:
                X_shuffle = shuffle(X_list[0],X_list[1],X_list[2],X_list[3])
            if len(X_list)==3:
                X_shuffle = shuffle(X_list[0],X_list[1],X_list[2])      
            if len(X_list)==2:
                X_shuffle = shuffle(X_list[0],X_list[1])    
          
            # 损失 （一个epoch的累计）
            loss_all, loss_rec, loss_cl, loss_pre = 0, 0, 0, 0
            H_epoch, MI_epoch = 0, 0
            for batch_xv, batch_No in next_batch(X_shuffle, config['training']['batch_size']):
                # 前向传播
                z_v_list = []
                for i in range(len(batch_xv)):
                    z_v = self.autoencoder_list[i].encoder(batch_xv[i])
                    z_v_list.append(z_v)


                # 计算重构loss
                recon_list = []
                reconstruction_loss = 0
                for i in range(config['training']['num_view']):
                    recon = F.mse_loss(self.autoencoder_list[i].decoder(z_v_list[i]),batch_xv[i])   # TODO F.mse_loss用法
                    reconstruction_loss+= recon
                    recon_list.append(recon)


                # 对比loss   (一个batch)
                cl_loss = 0
                mi = 0
                H = 0
                # TODO 查看互信息和两个熵的和
                for i in range(config['training']['num_view']):
                    for j in range(i+1,config['training']['num_view']):
                        cl_loss_i = crossview_contrastive_Loss(z_v_list[i], z_v_list[j], config['training']['alpha'])
                        cl_loss+=cl_loss_i
                        mi_v = mutual_info(z_v_list[i], z_v_list[j])
                        mi += mi_v
                        H12_v = H_12(z_v_list[i], z_v_list[j])
                        H += H12_v



                # 跨视角预测     (一个batch)
                # 改写成环预测： 1-->2 -->3 -->1     TODO 检查
                dualprediction_loss = 0
                for i in range(config['training']['num_view']):
                    if (i==config['training']['num_view']-1):
                        nextview,_ = self.prediction_list[i](z_v_list[i])
                        pre_i = F.mse_loss(nextview, z_v_list[0])
                    else:
                        nextview,_ = self.prediction_list[i](z_v_list[i])
                        pre_i = F.mse_loss(nextview, z_v_list[i+1])
                    dualprediction_loss+= pre_i


                loss = config['training']['lambda_cl']*cl_loss + reconstruction_loss * config['training']['lambda_recon']

                # ----开始进行跨视角预测, 不一致信息剪枝
                if epoch >= config['training']['start_dual_prediction']:
                    loss += dualprediction_loss * config['training']['lambda_dual']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 监控loss
                loss_all += loss.item()
                loss_rec += reconstruction_loss.item()
                loss_pre += dualprediction_loss.item()
                loss_cl += cl_loss.item()
                H_epoch += H.item()
                MI_epoch += mi.item()

            self.writer.add_scalar('cl_loss',loss_cl,epoch)
            self.writer.add_scalar('reconstruction_loss',loss_rec,epoch)
            self.writer.add_scalar('dualprediction_loss',loss_pre,epoch)
            self.writer.add_scalar('all_loss',loss_all,epoch)  
            # TODO 查看熵和互信息
            self.writer.add_scalar('H',H_epoch,epoch)  
            self.writer.add_scalar('MI_epoch',MI_epoch,epoch) 

            # 打印一个epoch的指标
            if (epoch + 1) % config['print_num'] == 0:
                output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f} " \
                         "===> Dual prediction loss = {:.4f}  ===> Contrastive loss = {:.4e} ===> Loss = {:.4e}" \
                    .format((epoch + 1), config['training']['epoch'], loss_rec, loss_pre, loss_cl, loss_all)

                logger.info("\033[2;29m" + output + "\033[0m")
            

            # 评估
            auc = self.evaluation(config, logger, X_list, Y_list, device)
            aucs.append(auc)
            self.writer.add_scalar('AUC:',auc,epoch)

        
        return np.max(np.array(aucs))

    def ablation_train(self, config, logger,X_list, Y_list, optimizer, device):
        """ 消融实验的训练.
            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              x1_train: data of view 1
              x2_train: data of view 2
              Y_list: labels
              mask: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              clustering performance: acc, nmi ,ari
        """

        aucs = []
        for epoch in range(config['training']['epoch']):
            # X_shuffle=[]
            # for i in range(len(X_list)):
            #     X_shuffle.append(shuffle(X_list[i]))
            if len(X_list)==6:
                X_shuffle = shuffle(X_list[0],X_list[1],X_list[2],X_list[3],X_list[4],X_list[5])
            if len(X_list)==5:
                X_shuffle = shuffle(X_list[0],X_list[1],X_list[2],X_list[3],X_list[4])
            if len(X_list)==4:
                X_shuffle = shuffle(X_list[0],X_list[1],X_list[2],X_list[3])
            if len(X_list)==3:
                X_shuffle = shuffle(X_list[0],X_list[1],X_list[2])      
            if len(X_list)==2:
                X_shuffle = shuffle(X_list[0],X_list[1])    
          
            # 损失 （一个epoch的累计）
            loss_all, loss_rec, loss_cl, loss_pre = 0, 0, 0, 0
            H_epoch, MI_epoch = 0, 0
            for batch_xv, batch_No in next_batch(X_shuffle, config['training']['batch_size']):
                # 前向传播
                z_v_list = []
                for i in range(len(batch_xv)):
                    z_v = self.autoencoder_list[i].encoder(batch_xv[i])
                    z_v_list.append(z_v)



                # cl_loss = 0
                mi = 0
                H = 0
                # 
                for i in range(config['training']['num_view']):
                    for j in range(i+1,config['training']['num_view']):
                        
                        mi_v = mutual_info(z_v_list[i], z_v_list[j])
                        mi += mi_v
                        H12_v = H_12(z_v_list[i], z_v_list[j])
                        H += H12_v



                # 跨视角预测     (一个batch)
                # 改写成环预测
                dualprediction_loss = 0
                for i in range(config['training']['num_view']):
                    if (i==config['training']['num_view']-1):
                        nextview,_ = self.prediction_list[i](z_v_list[i])
                        pre_i = F.mse_loss(nextview, z_v_list[0])
                    else:
                        nextview,_ = self.prediction_list[i](z_v_list[i])
                        pre_i = F.mse_loss(nextview, z_v_list[i+1])
                    dualprediction_loss+= pre_i

                
                loss = -(config['training']['lambda_H']*H + config['training']['lambda_mi']*mi)

                # ----开始进行跨视角预测, 不一致信息剪枝
                if epoch >= config['training']['start_dual_prediction']:
                    loss += dualprediction_loss * config['training']['lambda_dual']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 监控loss
                loss_all += loss.item()
                # loss_rec += reconstruction_loss.item()
                loss_pre += dualprediction_loss.item()
                # loss_cl += cl_loss.item()
                H_epoch += H.item()
                MI_epoch += mi.item()

            # self.writer.add_scalar('cl_loss',loss_cl,epoch)
            self.writer.add_scalar('reconstruction_loss',loss_rec,epoch)
            self.writer.add_scalar('dualprediction_loss',loss_pre,epoch)
            self.writer.add_scalar('all_loss',loss_all,epoch)  
            # 查看熵和互信息
            self.writer.add_scalar('H',H_epoch,epoch)  
            self.writer.add_scalar('MI_epoch',MI_epoch,epoch) 

            # 打印一个epoch的指标
            if (epoch + 1) % config['print_num'] == 0:
                output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f} " \
                         "===> Dual prediction loss = {:.4f}  ===> Contrastive loss = {:.4e} ===> Loss = {:.4e}" \
                    .format((epoch + 1), config['training']['epoch'], loss_rec, loss_pre, loss_cl, loss_all)

                logger.info("\033[2;29m" + output + "\033[0m")
            

            # 评估
            auc = self.evaluation(config, logger, X_list, Y_list, device)
            aucs.append(auc)
            self.writer.add_scalar('AUC:',auc,epoch)

        
        return np.max(np.array(aucs))

    def noteval_train(self, config, logger,X_list, Y_list, optimizer, device):
        aucs = []
        for epoch in range(config['training']['epoch']):
            if len(X_list)==6:
                X_shuffle = shuffle(X_list[0],X_list[1],X_list[2],X_list[3],X_list[4],X_list[5])
            if len(X_list)==5:
                X_shuffle = shuffle(X_list[0],X_list[1],X_list[2],X_list[3],X_list[4])
            if len(X_list)==4:
                X_shuffle = shuffle(X_list[0],X_list[1],X_list[2],X_list[3])
            if len(X_list)==3:
                X_shuffle = shuffle(X_list[0],X_list[1],X_list[2])      
            if len(X_list)==2:
                X_shuffle = shuffle(X_list[0],X_list[1])    
          
            # 损失 （一个epoch的累计）
            loss_all, loss_rec, loss_cl, loss_pre = 0, 0, 0, 0
            H_epoch, MI_epoch = 0, 0
            for batch_xv, batch_No in next_batch(X_shuffle, config['training']['batch_size']):
                # 前向传播
                z_v_list = []
                for i in range(len(batch_xv)):
                    z_v = self.autoencoder_list[i].encoder(batch_xv[i])
                    z_v_list.append(z_v)


                # 计算重构loss
                recon_list = []
                reconstruction_loss = 0
                for i in range(config['training']['num_view']):
                    recon = F.mse_loss(self.autoencoder_list[i].decoder(z_v_list[i]),batch_xv[i])   # TODO F.mse_loss用法
                    reconstruction_loss+= recon
                    recon_list.append(recon)


                # 对比loss   (一个batch)
                cl_loss = 0
                mi = 0
                H = 0
                # TODO 查看互信息和两个熵的和
                for i in range(config['training']['num_view']):
                    for j in range(i+1,config['training']['num_view']):
                        cl_loss_i = crossview_contrastive_Loss(z_v_list[i], z_v_list[j], config['training']['alpha'])
                        cl_loss+=cl_loss_i
                        mi_v = mutual_info(z_v_list[i], z_v_list[j])
                        mi += mi_v
                        H12_v = H_12(z_v_list[i], z_v_list[j])
                        H += H12_v



                # 跨视角预测     (一个batch)
                # 改写成环预测： 1-->2 -->3 -->1     TODO 检查
                dualprediction_loss = 0
                for i in range(config['training']['num_view']):
                    if (i==config['training']['num_view']-1):
                        nextview,_ = self.prediction_list[i](z_v_list[i])
                        pre_i = F.mse_loss(nextview, z_v_list[0])
                    else:
                        nextview,_ = self.prediction_list[i](z_v_list[i])
                        pre_i = F.mse_loss(nextview, z_v_list[i+1])
                    dualprediction_loss+= pre_i


                loss = config['training']['lambda_cl']*cl_loss + reconstruction_loss * config['training']['lambda_recon']

                # ----开始进行跨视角预测, 不一致信息剪枝
                if epoch >= config['training']['start_dual_prediction']:
                    loss += dualprediction_loss * config['training']['lambda_dual']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 监控loss
                loss_all += loss.item()
                loss_rec += reconstruction_loss.item()
                loss_pre += dualprediction_loss.item()
                loss_cl += cl_loss.item()
                H_epoch += H.item()
                MI_epoch += mi.item()

            self.writer.add_scalar('cl_loss',loss_cl,epoch)
            self.writer.add_scalar('reconstruction_loss',loss_rec,epoch)
            self.writer.add_scalar('dualprediction_loss',loss_pre,epoch)
            self.writer.add_scalar('all_loss',loss_all,epoch)  
            # 查看熵和互信息
            self.writer.add_scalar('H',H_epoch,epoch)  
            self.writer.add_scalar('MI_epoch',MI_epoch,epoch) 

            # 打印一个epoch的指标
            if (epoch + 1) % config['print_num'] == 0:
                output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f} " \
                         "===> Dual prediction loss = {:.4f}  ===> Contrastive loss = {:.4e} ===> Loss = {:.4e}" \
                    .format((epoch + 1), config['training']['epoch'], loss_rec, loss_pre, loss_cl, loss_all)

                logger.info("\033[2;29m" + output + "\033[0m")
            

            # 评估
            if epoch % 10 ==0:
                auc = self.evaluation(config, logger, X_list, Y_list, device)
                aucs.append(auc)
                self.writer.add_scalar('AUC:',auc,epoch)

        return np.max(np.array(aucs))

    def valid_mi_equal_dual(self, config, logger,X_list, Y_list, optimizer, device):
        status = 1      # 熵+互信息
        # status = 2      # 熵+对偶预测
        aucs = []
        for epoch in range(config['training']['epoch']):
            if len(X_list)==6:
                X_shuffle = shuffle(X_list[0],X_list[1],X_list[2],X_list[3],X_list[4],X_list[5])
            if len(X_list)==5:
                X_shuffle = shuffle(X_list[0],X_list[1],X_list[2],X_list[3],X_list[4])
            if len(X_list)==4:
                X_shuffle = shuffle(X_list[0],X_list[1],X_list[2],X_list[3])
            if len(X_list)==3:
                X_shuffle = shuffle(X_list[0],X_list[1],X_list[2])      
            if len(X_list)==2:
                X_shuffle = shuffle(X_list[0],X_list[1])    
          
            # 损失 （一个epoch的累计）
            loss_all, loss_rec, loss_cl, loss_pre = 0, 0, 0, 0
            H_epoch, MI_epoch = 0, 0
            for batch_xv, batch_No in next_batch(X_shuffle, config['training']['batch_size']):
                # 前向传播
                z_v_list = []
                for i in range(len(batch_xv)):
                    z_v = self.autoencoder_list[i].encoder(batch_xv[i])
                    z_v_list.append(z_v)


                # 计算重构loss
                reconstruction_loss = 0
                for i in range(config['training']['num_view']):
                    recon = F.mse_loss(self.autoencoder_list[i].decoder(z_v_list[i]),batch_xv[i])   # TODO F.mse_loss用法
                    reconstruction_loss+= recon



                # 对比loss   (一个batch)
                cl_loss = 0
                mi = 0
                H = 0
                # 计算互信息和熵
                for i in range(config['training']['num_view']):
                    for j in range(i+1,config['training']['num_view']):
                        cl_loss_i = crossview_contrastive_Loss(z_v_list[i], z_v_list[j], config['training']['alpha'])
                        cl_loss+=cl_loss_i
                        mi_v = mutual_info(z_v_list[i], z_v_list[j])
                        mi += mi_v
                        H12_v = H_12(z_v_list[i], z_v_list[j])
                        H += H12_v



                # 跨视角预测     (一个batch)
                # 改写成环预测： 1-->2 -->3 -->1     TODO 检查
                dualprediction_loss = 0
                for i in range(config['training']['num_view']):
                    if (i==config['training']['num_view']-1):
                        nextview,_ = self.prediction_list[i](z_v_list[i])
                        pre_i = F.mse_loss(nextview, z_v_list[0])
                    else:
                        nextview,_ = self.prediction_list[i](z_v_list[i])
                        pre_i = F.mse_loss(nextview, z_v_list[i+1])
                    dualprediction_loss+= pre_i


                # loss = config['training']['lambda_cl']*cl_loss + reconstruction_loss * config['training']['lambda_recon']
                loss = -H
                
                if (status==1 and epoch>= 14):
                    loss += - mi

                # ----开始进行跨视角预测, 不一致信息剪枝
                if (status==2 and epoch>= 40):
                    loss += dualprediction_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 监控loss
                loss_all += loss.item()
                loss_rec += reconstruction_loss.item()
                loss_pre += dualprediction_loss.item()
                loss_cl += cl_loss.item()
                H_epoch += H.item()
                MI_epoch += mi.item()

            self.writer.add_scalar('cl_loss',loss_cl,epoch)
            self.writer.add_scalar('reconstruction_loss',loss_rec,epoch)
            self.writer.add_scalar('dualprediction_loss',loss_pre,epoch)
            self.writer.add_scalar('all_loss',loss_all,epoch)  
            # 查看熵和互信息
            self.writer.add_scalar('H_epoch',H_epoch,epoch)  
            self.writer.add_scalar('MI_epoch',MI_epoch,epoch) 

            # 打印一个epoch的指标
            if (epoch + 1) % config['print_num'] == 0:
                output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f} " \
                         "===> Dual prediction loss = {:.4f}  ===> Contrastive loss = {:.4e} ===> Loss = {:.4e}" \
                    .format((epoch + 1), config['training']['epoch'], loss_rec, loss_pre, loss_cl, loss_all)

                logger.info("\033[2;29m" + output + "\033[0m")
            

            # 评估

            auc = self.evaluation(config, logger, X_list, Y_list, device)
            aucs.append(auc)
            self.writer.add_scalar('AUC:',auc,epoch)

        return np.max(np.array(aucs))

    # 
    def evaluation(self, config, logger, X_list, Y_list, device):
        with torch.no_grad():
            num_view = config['training']['num_view']
            for i in range(num_view):
                self.autoencoder_list[i].eval()
                self.prediction_list[i].eval()

            encoder_list = []
            recon_list = []
            pre_list = []
            for i in range(num_view):
                v_encode = self.autoencoder_list[i].encoder(X_list[i])
                encoder_list.append(v_encode)
                recon_v = torch.norm(self.autoencoder_list[i].decoder(v_encode) - X_list[i],dim=1)
                recon_list.append(recon_v)

            for i in range(num_view):
                if (i==num_view-1):
                    nextview,_ = self.prediction_list[i](encoder_list[i])
                    # pre_i = F.mse_loss(nextview, encoder_list[0])
                    pre_i = torch.norm(nextview - encoder_list[0],dim=1)  
                    # TODO pre1 = torch.norm(img2txt - v2_encode,dim=1)  
                else:
                    nextview,_ = self.prediction_list[i](encoder_list[i])
                    pre_i = torch.norm(nextview - encoder_list[i+1],dim=1)
                    # pre_i = F.mse_loss(nextview, encoder_list[i+1])
                pre_list.append(pre_i)

            # 转numpy
            encoder_np_list = []
            recon_np_list = []
            pre_np_list = []
            for i in range(num_view):
                encoder_np_list.append(encoder_list[i].cpu().numpy())
                recon_np_list.append(recon_list[i].cpu().numpy())
                pre_np_list.append(pre_list[i].cpu().numpy())
            scores = evaluation.cal_scores(encoder_np_list,config['num_neibor'],recon_np_list,pre_np_list,config)
            
            # 计算AUC
            auc = util.compute_auc(Y_list[0],scores)

            logger.info("\033[2;29m" + 'view_auc ' + str(auc) + "\033[0m")
            # 测试完记得将模型置为训练模式
            for i in range(num_view):
                self.autoencoder_list[i].train()
                self.prediction_list[i].train()
            # self.autoencoder1.train(), self.autoencoder2.train()
            # self.img2txt.train(), self.txt2img.train()

        return auc





# 视角内编码解码器
class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          encoder_dim: Should be a list of ints, hidden sizes of
            encoder network, the last element is the size of the latent representation.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Autoencoder, self).__init__()

        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        encoder_layers.append(nn.Softmax(dim=1))
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        """Encode sample features.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [n_nodes, latent_dim] float tensor, representation Z.
        """
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        """Decode sample features.

            Args:
              latent: [num, latent_dim] float tensor, representation Z.

            Returns:
              x_hat: [n_nodes, feat_dim] float tensor, reconstruction x.
        """
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        """Pass through autoencoder.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor, representation Z.
              x_hat:  [num, feat_dim] float tensor, reconstruction x.
        """
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent

# 视角间预测器
class Prediction(nn.Module):
    """Dual prediction module that projects features from corresponding latent space."""

    def __init__(self,
                 prediction_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          prediction_dim: Should be a list of ints, hidden sizes of
            prediction network, the last element is the size of the latent representation of autoencoder.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Prediction, self).__init__()

        self._depth = len(prediction_dim) - 1
        self._activation = activation
        self._prediction_dim = prediction_dim

        encoder_layers = []
        for i in range(self._depth):
            encoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i + 1]))
            if batchnorm:
                encoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i + 1]))
            if self._activation == 'sigmoid':
                encoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                encoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                encoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(self._depth, 0, -1):
            decoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i - 1]))
            if i > 1:
                if batchnorm:
                    decoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i - 1]))

                if self._activation == 'sigmoid':
                    decoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    decoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    decoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        decoder_layers.append(nn.Softmax(dim=1))
        self._decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Data recovery by prediction.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor.
              output:  [num, feat_dim] float tensor, recovered data.
        """
        latent = self._encoder(x)
        output = self._decoder(latent)
        return output, latent


class MINE(nn.Module):
    def __init__(self,hidden_size_1,hidden_size_2,hidden_size):
        super(MINE, self).__init__()
        self.layers = nn.Sequential(nn.Linear(hidden_size_1+hidden_size_2, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))


    def forward(self, x, y):
        batch_size = x.size(0)
        tiled_x = torch.cat([x, x, ], dim=0)
        idx = torch.randperm(batch_size)
        # shuffle操作来获得边缘分布采样的样本
        shuffled_y = y[idx]
        concat_y = torch.cat([y, shuffled_y], dim=0)
        inputs = torch.cat([tiled_x, concat_y], dim=1)
        logits = self.layers(inputs)
        if torch.isnan(logits).any()==True:
            print("T有值为nan")
        pred_xy = logits[:batch_size]   # 联合分布
        pred_x_y = logits[batch_size:]  # px乘py
        mi = np.log2(np.exp(1)) * (torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y))))    # 估计的互信息
        # compute loss, you'd better scale exp to bit
        if torch.isnan(mi).any()==True:
            print("mi有值为nan")
        return mi

