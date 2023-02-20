"""PhysNet Trainer."""
import os

import numpy as np
import torch
import torch.optim as optim
from metrics.metrics import calculate_metrics,calculate_metrics_SR
from signal_methods.signal_predictor import signal_predict
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.model.TS_CAN import TSCAN
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm
from utils.utils import noise_signal_generator,sine_wave
from neural_methods.loss.rPPGRemovalLoss import Total_Loss
import cv2
from dataset.data_loader.BaseLoader import BaseLoader
import random
class PhysnetTrainer(BaseTrainer):

    def __init__(self, config):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=config.MODEL.PHYSNET.FRAME_NUM).to(self.device)  # [3, T, 128,128]
        self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        self.removal_LR=config.SIGNAL_REMOVAL.LR
        self.test_DL=config.SIGNAL_REMOVAL.TEST_DL_METHOD
        self.test_model_path=config.SIGNAL_REMOVAL.TEST_MODEL_PATH
        self.num_chunks=config.SIGNAL_REMOVAL.NUM_CHUNKS
        self.target_signal=config.SIGNAL_REMOVAL.TARGET_SIGNAL_TYPE 
        self.removal_num_epochs=config.SIGNAL_REMOVAL.NUM_EPOCHS
        self.videos_saved_path=config.SIGNAL_REMOVAL.VIDEOS_SAVED_PATH
        self.test_classical_methods=config.SIGNAL_REMOVAL.TEST_CLASSICAL_METHODS
        self.classical_methods=config.SIGNAL_REMOVAL.CLASSICAL_TESTING_METHODS
        if(self.test_DL):
            self.test_model = TSCAN(frame_depth=20, img_size=config.TRAIN.DATA.PREPROCESS.H).to(self.device)
            self.test_model = torch.nn.DataParallel(self.test_model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        self.loss_model = Neg_Pearson()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config.TRAIN.LR)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
        self.config = config
        self.best_epoch = 0
        self.chunk_len=config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH

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
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    batch[0].to(torch.float32).to(self.device))
                BVP_label = batch[1].to(
                    torch.float32).to(self.device)
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                            torch.std(BVP_label)  # normalize
                loss = self.loss_model(rPPG, BVP_label)
                loss.backward()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch + 1}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                self.optimizer.step()
                self.optimizer.zero_grad()
                tbar.set_postfix(loss=loss.item())
            valid_loss = self.valid(data_loader)
            self.save_model(epoch)
            print('validation loss: ', valid_loss)
            if (valid_loss < min_valid_loss) or (valid_loss < 0):
                min_valid_loss = valid_loss
                self.best_epoch = epoch
                print("Update best model! Best epoch: {}".format(self.best_epoch))
                self.save_model(epoch)
        print("best trained epoch:{}, min_val_loss:{}".format(
            self.best_epoch, min_valid_loss))

    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")
        print(" ====Validing===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                BVP_label = valid_batch[1].to(
                    torch.float32).to(self.device)
                rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    valid_batch[0].to(torch.float32).to(self.device))
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                            torch.std(BVP_label)  # normalize
                loss_ecg = self.loss_model(rPPG, BVP_label)
                valid_loss.append(loss_ecg.item())
                valid_step += 1
                vbar.set_postfix(loss=loss_ecg.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Runs the model on test sets."""
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
            print(self.config.INFERENCE.MODEL_PATH)
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
                data, label = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                pred_ppg_test, _, _, _ = self.model(data)
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = label[idx]
        calculate_metrics(predictions, labels, self.config)


    def signal_removal(self, data_loader):
        """ rPPG signal removal from videos."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print("===signal removing===")
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
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
                
        if(self.test_DL):
            if not os.path.exists(self.test_model_path):
                raise ValueError("Test model path error! Please check SIGNAL_REMOVAL.TEST_MODEL_PATH in your yaml.")
            self.test_model.load_state_dict(torch.load(self.test_model_path))
            self.test_model = self.test_model.to(self.config.DEVICE)
            self.test_model.eval()

        video_array, original_video_array, target_array, num_samples=self.prepare_data(data_loader)
        video_array=self.run_input_GD(video_array,original_video_array,target_array,num_samples)
        self.save_videos(video_array,num_samples)
        self.evaluate_modified_videos(video_array,original_video_array,target_array,num_samples)

    def run_input_GD(self,video_array,original_video_array,target_array,num_samples):
        self.criterion=Total_Loss()
        for i in range(num_samples):
            modified_video=video_array[:,:,:,:,i]
            loss_video_data=torch.permute(torch.reshape(original_video_array[:,:,:,:,i],shape=(self.num_chunks,-1,3,72,72)),dims=(0,2,1,3,4))
            for epoch in range(self.removal_num_epochs):
                with torch.no_grad():
                    reshaped_data=torch.permute(torch.reshape(modified_video,shape=(self.num_chunks,-1,3,72,72)),dims=(0,2,1,3,4)).detach().clone().requires_grad_(True)
                pred_ppg_test,_,_,_ = self.model(reshaped_data)
                pred_ppg_test=torch.reshape(pred_ppg_test,shape=(-1,1))
                loss=self.criterion(pred_ppg_test,target_array[:,i],loss_video_data,reshaped_data,0.01)
                total_loss=sum(loss)
                total_loss.backward()
                print(i, epoch, loss[0].item(), loss[1].item(),total_loss.item())
                with torch.no_grad():
                    modified_video-=self.removal_LR*torch.reshape(torch.permute(reshaped_data.grad.data,dims=(0,2,1,3,4)),shape=(-1,3,72,72))
                    modified_video=torch.tensor(BaseLoader.standardized_data(modified_video.cpu().numpy()),device=self.config.DEVICE)
            print('\n')
        return video_array
        
    def evaluate_modified_videos(self,video_array,original_video_array,target_array,num_samples): 
        with torch.no_grad():       
            predictions_train=torch.zeros_like(target_array,device=self.config.DEVICE)
            predictions_test=torch.zeros_like(target_array,device=self.config.DEVICE)
            modified_video_diff=torch.zeros_like(target_array,device=self.config.DEVICE)

            for i in range(num_samples):
                rgb_modified=torch.tensor(self.rgb_img_reconstruction(video_array[:,:,:,:,i]),dtype=torch.float32)
                rgb_original=torch.tensor(self.rgb_img_reconstruction(original_video_array[:,:,:,:,i]),dtype=torch.float32)
                rgb_modified=torch.mean(rgb_modified,axis=3)
                rgb_original=torch.mean(rgb_original,axis=3)
                rPPG= self.model(torch.permute(torch.reshape(video_array[:,:,:,:,i],shape=(self.num_chunks,-1,3,72,72)),dims=(0,2,1,3,4)))[0]
                predictions_train[:,i] =torch.reshape(rPPG,shape=(-1,))
                video_diff=(rgb_modified-rgb_original)**2
                video_diff=torch.reshape(video_diff,(-1,72*72))
                video_diff=torch.sqrt(torch.mean(video_diff,axis=1))
                modified_video_diff[:,i] = video_diff
                
            max_pixel_val=255
            print("PhysNet Test:")
            calculate_metrics_SR(predictions_train, target_array, self.config,True,modified_video_diff,max_pixel_val)
            if(self.test_DL): 
                print("TS-CAN Test:")
                diffs=torch.tensor(BaseLoader.diff_normalize_data(video_array[:,:,:,:,i].cpu().numpy()),device=self.config.DEVICE)
                diffs=torch.cat((diffs,torch.zeros_like(diffs[0,:,:,:],device=self.device).unsqueeze(0)),dim=0)
                rPPG=torch.cat((diffs,video_array[:,:,:,:,i]),dim=1)
                predictions_test[:,i] = self.test_model(rPPG)[0].squeeze(1)
                calculate_metrics_SR(predictions_test, target_array, self.config,True,modified_video_diff,max_pixel_val)
            if(self.test_classical_methods):
                print("Classical tests:")
                rPPG=torch.permute(video_array,dims=(0,2,3,1,4))
                for signal_method in self.classical_methods:
                    signal_predict(self.config, rPPG, target_array, signal_method,True)

    def prepare_data(self, data_loader):
        with torch.no_grad():
            labels = dict()
            videos = dict()
            for i, test_batch in enumerate(data_loader['test']):
                batch_size = test_batch[0].shape[0]
                videos_test, labels_test= test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in videos.keys():
                        labels[subj_index] = dict()
                        videos[subj_index] = dict()
                    labels[subj_index][sort_index] = labels_test[idx]
                    videos[subj_index][sort_index] = torch.permute(videos_test[idx],dims=(1,0,2,3))
            original_label_array=[]
            for i,(video,label) in enumerate(zip(videos.values(),labels.values())):
                label=self.reform_data_from_dict(label)
                video=self.reform_data_from_dict(video)
                if(len(label)>=self.num_chunks*self.chunk_len):
                    label=label[:self.num_chunks*self.chunk_len]
                    video=video[:self.num_chunks*self.chunk_len]
                    if(self.target_signal=="RANDOM_FREQ_SINUSOID"):
                        rand_freq=random.randint(65,115)
                        target_label=sine_wave(rand_freq,self.num_chunks*self.chunk_len+1)
                        target_label=BaseLoader.diff_normalize_label(target_label)
                    elif(self.target_signal=="LPF_GAUSS_NOISE"):
                        target_label=noise_signal_generator(0,10,15*self.chunk_len+1)
                        target_label=BaseLoader.diff_normalize_label(target_label)
                    elif(self.target_signal=="OTHER_PERSON_RPPG"):
                        target_label=label.cpu().numpy()
                    else:
                        raise ValueError("TARGET SIGNAL NOT SUPPORTED!")

                    if not len(original_label_array):
                        original_label_array=label.unsqueeze(1)
                        original_video_array=video.unsqueeze(4)
                        target_array=np.reshape(target_label,(-1,1)) 
                    else:
                        original_label_array=torch.cat((original_label_array,label.unsqueeze(1)),dim=1)
                        original_video_array=torch.cat((original_video_array,video.unsqueeze(4)),dim=4)
                        target_array=np.concatenate((target_array,np.reshape(target_label,(-1,1))),axis=1)
            video_array=original_video_array.detach().clone().to(self.config.DEVICE)
            target_array=torch.from_numpy(target_array).to(self.config.DEVICE)
            #original_video_array=original_video_array.detach().clone()
            (sample_len,num_samples)=original_label_array.shape
            if(self.target_signal=="OTHER_PERSON_RPPG"):
                target_array=target_array[:,np.random.permutation(num_samples)]
            return video_array,original_video_array, target_array,num_samples

    def save_videos(self,video_array,num_samples):
        with torch.no_grad():
            for i in range(num_samples):   
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(os.path.join(self.videos_saved_path,'modified_{0}.mp4'.format(i)), fourcc, float(30), (72,72),True)
                images=self.rgb_img_reconstruction(video_array[:,:,:,:,i])
                for (j,image) in enumerate(images):
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    video.write(image)
                video.write(images)
                video.release()
                
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