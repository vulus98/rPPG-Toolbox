"""Trainer for TSCAN."""

import logging
import os
from collections import OrderedDict
import cv2 
import numpy as np
import torch
import torch.optim as optim
from metrics.metrics import calculate_metrics,calculate_metrics_SR
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.TS_CAN import TSCAN
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm
from utils.utils import noise_signal_generator,sine_wave
from neural_methods.loss.rPPGRemovalLoss import Total_Loss
from dataset.data_loader.BaseLoader import BaseLoader 
import random
class TscanTrainer(BaseTrainer):

    def __init__(self, config):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.frame_depth = config.MODEL.TSCAN.FRAME_DEPTH
        self.model = TSCAN(frame_depth=self.frame_depth, img_size=config.TRAIN.DATA.PREPROCESS.H).to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu * self.frame_depth
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.best_epoch = 0

    def train(self, data_loader):
        """ TODO:Docstring"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        min_valid_loss = 1
        for epoch in range(self.max_epoch_num):
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].to(
                    self.device), batch[1].to(self.device)
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)
                data = data[:(N * D) // self.base_len * self.base_len]
                labels = labels[:(N * D) // self.base_len * self.base_len]
                self.optimizer.zero_grad()
                pred_ppg,_ = self.model(data)
                loss = self.criterion(pred_ppg, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch + 1}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                tbar.set_postfix(loss=loss.item())
            valid_loss = self.valid(data_loader)
            self.save_model(epoch)
            print('validation loss: ', valid_loss)
            if (valid_loss < min_valid_loss) or (valid_loss < 0):
                min_valid_loss = valid_loss
                self.best_epoch = epoch
                print("Update best model! Best epoch: {}".format(self.best_epoch))
                self.save_model(epoch)
        print("best trained epoch:{}, min_val_loss:{}".format(self.best_epoch, min_valid_loss))

    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")
        print("===Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid, labels_valid = valid_batch[0].to(
                    self.device), valid_batch[1].to(self.device)
                N, D, C, H, W = data_valid.shape
                data_valid = data_valid.view(N * D, C, H, W)
                labels_valid = labels_valid.view(-1, 1)
                data_valid = data_valid[:(N * D) // self.base_len * self.base_len]
                labels_valid = labels_valid[:(N * D) // self.base_len * self.base_len]
                pred_ppg_valid,_ = self.model(data_valid)
                loss = self.criterion(pred_ppg_valid, labels_valid)
                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        print("===Testing===")
        predictions = dict()
        labels = dict()
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
        else:
            best_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
            print("Testing uses non-pretrained model!")
            print(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))
        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        with torch.no_grad():
            for _, test_batch in enumerate(data_loader['test']):
                batch_size = test_batch[0].shape[0]
                data_test, labels_test = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)
                data_test = data_test[:(N * D) // self.base_len * self.base_len]
                labels_test = labels_test[:(N * D) // self.base_len * self.base_len]
                pred_ppg_test,masks = self.model(data_test)
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]

        calculate_metrics(predictions, labels, self.config)

    def signal_removal(self, data_loader):
        """ rPPG signal removal from videos."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        print("===signal removing===")
        labels = dict()
        videos = dict()
        modified_video_diff=dict()
        if self.config.TOOLBOX_MODE == "rPPG_removal":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Signal removing uses pretrained model!")
        else:
            best_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
            print("Signal removing uses non-pretrained model!")
            print(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))
        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        self.criterion=Total_Loss()
        for i, test_batch in enumerate(data_loader['test']):
            batch_size = test_batch[0].shape[0]
            data_test, labels_test= test_batch[0].to(
                self.config.DEVICE), test_batch[1].to(self.config.DEVICE)#, test_batch[4].to(self.config.DEVICE)
            N, D, C, H, W = data_test.shape
            data_test = data_test.view(N * D, C, H, W)
            # N_o, D_o, H_o, W_o, C_o = original_ims_test.shape
            #original_ims_test = original_ims_test.view(N_o*D_o, H_o, W_o, C_o)
            labels_test = labels_test.view(-1, 1)
            data_test = data_test[:(N * D) // self.base_len * self.base_len]
            labels_test = labels_test[:(N * D) // self.base_len * self.base_len]
            #original_ims_test = original_ims_test[:(N * D) // self.base_len * self.base_len]
            for idx in range(batch_size):
                subj_index = test_batch[2][idx]
                sort_index = int(test_batch[3][idx])
                if subj_index not in videos.keys():
                    labels[subj_index] = dict()
                    videos[subj_index] = dict()
                labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                videos[subj_index][sort_index] = data_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
        with torch.no_grad():
            for i,(video,label) in enumerate(zip(videos.values(),labels.values())):
                label=self.reform_data_from_dict(label)[:10*self.chunk_len]
                video=self.reform_data_from_dict(video)[:10*self.chunk_len]
                rand_freq=random.randint(60,120)
                #noise_label=sine_wave(rand_freq,10*self.chunk_len+1)
                noise_label=noise_signal_generator(0,10,10*self.chunk_len+1)
                noise_label=BaseLoader.diff_normalize_label(noise_label)
                if not i:
                    label_array=label
                    original_video_array=video.unsqueeze(4)
                    noise_array=np.reshape(noise_label,(-1,1)) 
                else:
                    label_array=torch.cat((label_array,label),dim=1)
                    original_video_array=torch.cat((original_video_array,video.unsqueeze(4)),dim=4)
                    noise_array=np.concatenate((noise_array,np.reshape(noise_label,(-1,1))),axis=1)
            video_array=original_video_array.detach().clone().to(self.config.DEVICE)
            noise_array=torch.from_numpy(noise_array).to(self.config.DEVICE)
            original_video_array=original_video_array.detach().clone()
        (sample_len,num_samples)=label_array.shape
        num_epochs=50
        num_samples=6
        extract_new_masks=0
        noise_array=label_array[:,torch.randperm(num_samples)]
        import time
        s=time.time()
        for i in range(5,num_samples):
            modified_video=video_array[:,:,:,:,i].requires_grad_(True)
            with torch.no_grad():
                pred_ppg_test,attention_masks = self.model(modified_video)
                face_masks=self.detect_masks(modified_video,attention_masks,data_loader)
                face_masks=torch.stack(face_masks)
            for epoch in range(num_epochs):
                pred_ppg_test,attention_masks = self.model(modified_video)
                loss=self.criterion(pred_ppg_test,noise_array[:,i],original_video_array[:,:,:,:,i],modified_video,0.1)
                total_loss=sum(loss)
                total_loss.backward()
                print(i, epoch, loss[0].item(), loss[1].item(),total_loss.item())
                with torch.no_grad():
                    #if not epoch:      
                        # if faces_detected and not extract_new_masks:
                        #     face_masks=np.load('/local/home/vbozic/PURE_dataset/MasksData/masks{0}.npy'.format(i))
                        #     face_masks=torch.from_numpy(face_masks).to(self.config.DEVICE)
                        # else:
                        #     face_masks=self.detect_masks(modified_video,attention_masks,data_loader)
                        
                    modified_video[:,3:,:,:]-=1000*torch.stack(list(map(lambda k: modified_video.grad.data[k,3:,:,:]*face_masks[k,:,:],range(sample_len))))
                    #modified_video[:,3:,:,:]-=1000*modified_video.grad.data[:,3:,:,:]
                    modified_video[:,3:,:,:]=torch.tensor(BaseLoader.standardized_data(modified_video[:,3:,:,:].cpu().numpy()),device=self.config.DEVICE)
                    modified_video[:-1,:3,:,:]=torch.tensor(BaseLoader.diff_normalize_data(modified_video[:,3:,:,:].cpu().numpy()),device=self.config.DEVICE)
            # if extract_new_masks and (not faces_detected):
            #     np.save('/local/home/vbozic/PURE_dataset/MasksData/masks{0}.npy'.format(i), face_masks.cpu().numpy())
            print('\n')
        l=[100,200,300,400]
        print("time", time.time()-s)   
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter('CAN_modified.mp4', fourcc, float(30), (72,72),True)
        images=self.rgb_img_reconstruction(video_array[:,3:,:,:,5])
        for (j,image) in enumerate(images):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if(j in l):
                cv2.imwrite("CAN_modified_image{0}.png".format(j),image)
            video.write(image)
        video.write(images)
        video.release()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter('CAN_original.mp4', fourcc, float(30), (72,72),True)
        images=self.rgb_img_reconstruction(original_video_array[:,3:,:,:,5])
        for (j,image) in enumerate(images):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if(j in l):
                cv2.imwrite("CAN_original_image{0}.png".format(j),image)
            video.write(image)
        video.write(images)
        video.release()
        # predictions=torch.zeros_like(label_array,device=self.config.DEVICE)
        # modified_video_diff=torch.zeros_like(label_array,device=self.config.DEVICE)
        # with torch.no_grad():
        #     for i in range(num_samples):
        #         rgb_modified=torch.tensor(self.rgb_img_reconstruction(video_array[:,3:,:,:,i]),dtype=torch.float32)
        #         rgb_original=torch.tensor(self.rgb_img_reconstruction(original_video_array[:,3:,:,:,i]),dtype=torch.float32)
        #         rgb_modified=torch.mean(rgb_modified,axis=3)
        #         rgb_original=torch.mean(rgb_original,axis=3)
        #         predictions[:,i] = self.model(video_array[:,:,:,:,i])[0].squeeze(1)
        #         video_diff=(rgb_modified-rgb_original)**2
        #         video_diff=torch.reshape(video_diff,(-1,72*72))
        #         video_diff=torch.sqrt(torch.mean(video_diff,axis=1))
        #         modified_video_diff[:,i] = video_diff
        # max_pixel_val=255
        # calculate_metrics_SR(predictions, noise_array, self.config,True,modified_video_diff,max_pixel_val)


    def detect_masks(self,data_test,masks,data_loader):
        prev_faces=[0,0,71,71]
        face_masks=[]
        grays=self.rgb_img_reconstruction(data_test[:,3:,:,:])
        for (gray,mask) in zip(grays,masks):
            gray=cv2.cvtColor(gray,cv2.COLOR_RGB2GRAY)
            faces=data_loader['test'].dataset.facial_detection(gray)
            if not (faces[0]+faces[1]):
                faces=prev_faces
            prev_faces=faces
            x,y,w,h=prev_faces
            coors=np.asarray([x+int(0.2*w),x+int(0.8*w),y+int(0.1*h), y+int(h)])
            coors_x=coors[:2]
            coors_y=coors[2:]
            # import matplotlib.pyplot as plt
            # plt.figure
            # plt.imshow(mask.squeeze(0).cpu().numpy())
            # plt.savefig('test_image.png')
            data_mask=(mask.squeeze(0)/max(mask.flatten())).cpu().numpy()
            data_mask=cv2.resize(data_mask,dsize=(72,72))
            border_mask=np.zeros_like(data_mask,dtype=np.uint8)
            inner_mask=np.ones((coors_y[1]-coors_y[0],coors_x[1]-coors_x[0]))
            border_mask[coors_y[0]:coors_y[1],coors_x[0]:coors_x[1]]=inner_mask
            data_mask*=border_mask
            binary_mask=np.zeros_like(data_mask)
            binary_mask[data_mask>0.9]=1
            face_masks.append(torch.from_numpy(binary_mask).to(self.config.DEVICE))
        return face_masks
            


    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)

    def rgb_img_reconstruction(self,images):
        images=images.cpu().detach().numpy()
        min_images=np.reshape(images,(-1,3,72*72))
        mins=np.min(min_images,axis=2)
        images=np.transpose(images,(2,3,0,1))
        images-=mins
        max_images=np.reshape(images,(72*72,-1,3))
        maxs=np.max(max_images,axis=0)
        images/=maxs
        images*=255
        images=np.transpose(images,(2,0,1,3))
        images=images.astype(np.uint8)
        return images
    
    def reform_data_from_dict(self,data):
        sort_data = sorted(data.items(), key=lambda x: x[0])
        sort_data = [i[1] for i in sort_data]
        sort_data = torch.cat(sort_data, dim=0)
        return sort_data.detach().clone().requires_grad_(True)