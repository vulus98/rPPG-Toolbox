# About
Remote photoplethysmography (rPPG) is a contactless method for the remote measurement of the blood volume pulse from the video of a person's face. Since these physiological signals represent private information, advancements in this field raise concerns about compromising this data and privacy issues to which this may lead. Thus, ways of concealing or removing the rPPG signal were much needed.
 
 To counter the methods for extraction of this signal, a video editing method was introduced in [Privacy-Phys paper](https://ieeexplore.ieee.org/document/9806161), efficiently modifying face areas of a video from where the rPPG signal is estimated and replacing the original signal with a dummy one, concealing it. This basic pipeline has been modified, its performance in different settings and overall limitations have been inspected. Experimental results show the versatility of the given technique, achieving state-of-art quality of rPPG removal while extending its possible applications in different scenarios. On top of that, the dataset for further improvements and training of novel end-to-end architectures for rPPG signal removal has been created.

# Setup

STEP1: `bash setup.sh` 

STEP2: `conda activate rppg-toolbox` 

STEP3: `pip install -r requirements.txt` 

# Training on PURE and testing on UBFC with TSCAN 

STEP1: Download the PURE raw data by asking the [paper authors](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure).

STEP2: Download the UBFC raw data via [link](https://sites.google.com/view/ybenezeth/ubfcrppg)

STEP3: Modify `./configs/PURE_PURE_UBFC_TSCAN_BASIC.yaml` 

STEP4: Run `python main.py --config_file ./configs/PURE_PURE_UBFC_TSCAN_BASIC.yaml` 

Note1: Preprocessing requires only once; thus turn it off on the yaml file when you train the network after the first time. 

Note2: The example yaml setting will allow 80% of PURE to train and 20% of PURE to valid. 
After training, it will use the best model(with the least validation loss) to test on UBFC.

# Training on SCAMPS and testing on UBFC with DeepPhys

STEP1: Download the SCAMPS via this [link](https://github.com/danmcduff/scampsdataset) and split it into train/val/test folders.

STEP2: Download the UBFC via [link](https://sites.google.com/view/ybenezeth/ubfcrppg)

STEP3: Modify `./configs/SCAMPS_SCAMPS_UBFC_DEEPPHYS_BASIC.yaml` 

STEP4: Run `python main.py --config_file ./configs/SCAMPS_SCAMPS_UBFC_DEEPPHYS_BASIC.yaml`

Note1: Preprocessing requires only once; thus turn it off on the yaml file when you train the network after the first time. 

Note2: The example yaml setting will allow 80% of SCAMPS to train and 20% of SCAMPS to valid. 
After training, it will use the best model(with the least validation loss) to test on UBFC.

# Predicting BVP signal and calculate heart rate on UBFC with POS/CHROME/ICA

STEP1: Download the UBFC via [link](https://sites.google.com/view/ybenezeth/ubfcrppg)

STEP3: Modify `./configs/UBFC_SIGNAL.yaml` 

STEP4: Run `python main.py --config_file ./configs/UBFC_SIGNAL.yaml`

# Yaml File Setting
The rPPG-Toolbox uses yaml file to control all parameters for training and evaluation. 
You can modify the existing yaml files to meet your own training and testing requirements.

Here are some explanation of parameters:
* #### TOOLBOX_MODE: 

  * `train_and_test`: train on the dataset and use the newly trained model to test.
  * `only_test`: you need to set INFERENCE-MODEL_PATH, and it will use pre-trained model initialized with the MODEL_PATH to test.
  * `rPPG_removal`: exists only for PhysNet and PURE dataset, uses already pretrained PhysNet to run
  rPPG signal modification.
  * `signal method`: use signal methods to predict rppg BVP signal and calculate heart rate.
* #### TRAIN / VALID / TEST / SIGNAL DATA: 
  * `DATA_PATH`: The input path of raw data
  * `CACHED_PATH`: The output path to preprocessed data. This path also houses a directory of .csv files containing data paths to files loaded by the dataloader. This filelist (found in default at CACHED_PATH/DataFileLists). These can be viewed for users to understand which files are used in each data split (train/val/test)
  * `EXP_DATA_NAME` If it is "", the toolbox generates a EXP_DATA_NAME based on other defined parameters. Otherwise, it uses the user-defined EXP_DATA_NAME.  
  * `BEGIN" & "END`: The portion of the dataset used for training/validation/testing. For example, if the `DATASET` is PURE, `BEGIN` is 0.0 and `END` is 0.8 under the TRAIN, the first 80% PURE is used for training the network. If the `DATASET` is PURE, `BEGIN` is 0.8 and `END` is 1.0 under the VALID, the last 20% PURE is used as the validation set. It is worth noting that validation and training sets don't have overlapping subjects.  
  * `DATA_TYPE`: How to preprocess the video data
  * `LABEL_TYPE`: How to preprocess the label data
  * `DO_CHUNK`: Whether to split the raw data into smaller chunks
  * `CHUNK_LENGTH`: The length of each chunk (number of frames)
  * `CROP_FACE`: Whether to perform face detection
  * `DYNAMIC_DETECTION`: If False, face detection is only performed at the first frame and the detected box is used to crop the video for all of the subsequent frames. If True, face detection is performed at a specific frequency which is defined by `DYNAMIC_DETECTION_FREQUENCY`. 
  * `DYNAMIC_DETECTION_FREQUENCY`: The frequency of face detection (number of frames) if DYNAMIC_DETECTION is True
  * `LARGE_FACE_BOX`: Whether to enlarge the rectangle of the detected face region in case the detected box is not large enough for some special cases (e.g., motion videos)
  * `LARGE_BOX_COEF`: The coefficient of enlarging. See more details at `https://github.com/ubicomplab/rPPG-Toolbox/blob/main/dataset/data_loader/BaseLoader.py#L162-L165`. 

  
* #### MODEL : Set used model (support Deepphys / TSCAN / Physnet right now) and their parameters.
* #### SIGNAL METHOD: Set used signal method. Example: ["ica", "pos", "chrome"]
* #### METRICS: Set used metrics. Example: ['MAE','RMSE','MAPE','Pearson']

# Dataset
The toolbox supports three datasets, which are SCAMPS, UBFC, and PURE (COHFACE support will be added shortly). Cite corresponding papers when using.
For now, we only recommend training with PURE or SCAMPS due to the level of synchronization and volume of the dataset.
* [SCAMPS](https://arxiv.org/abs/2206.04197)
  
    * D. McDuff, M. Wander, X. Liu, B. Hill, J. Hernandez, J. Lester, T. Baltrusaitis, "SCAMPS: Synthetics for Camera Measurement of Physiological Signals", Arxiv, 2022
    * In order to use this dataset in a deep model, you should organize the files as follows:
    -----------------
         data/SCAMPS/Train/
            |-- P00001.mat
            |-- P00002.mat
            |-- P00003.mat
         |...
         data/SCAMPS/Val/
            |-- P00001.mat
            |-- P00002.mat
            |-- P00003.mat
         |...
         data/SCAMPS/Test/
            |-- P00001.mat
            |-- P00002.mat
            |-- P00003.mat
         |...
    -----------------

* [UBFC](https://sites.google.com/view/ybenezeth/ubfcrppg)
  
    * S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.
    * In order to use this dataset in a deep model, you should organize the files as follows:
    -----------------
         data/UBFC/
         |   |-- subject1/
         |       |-- vid.avi
         |       |-- ground_truth.txt
         |   |-- subject2/
         |       |-- vid.avi
         |       |-- ground_truth.txt
         |...
         |   |-- subjectn/
         |       |-- vid.avi
         |       |-- ground_truth.txt
    -----------------
* [COHFACE](https://www.idiap.ch/en/dataset/cohface)
    * Guillaume Heusch, André Anjos, Sébastien Marcel, “A reproducible study on remote heart rate measurement”, arXiv, 2016.
    * In order to use this dataset in a deep model, you should organize the files as follows:
    -----------------
         data/COHFACE/
         |   |-- 1/
         |      |-- 0/
         |          |-- data.avi
         |          |-- data.hdf5
         |      |...
         |      |-- 3/
         |          |-- data.avi
         |          |-- data.hdf5
         |...
         |   |-- n/
         |      |-- 0/
         |          |-- data.avi
         |          |-- data.hdf5
         |      |...
         |      |-- 3/
         |          |-- data.avi
         |          |-- data.hdf5
    -----------------
    
* [PURE](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure)
    * Stricker, R., Müller, S., Gross, H.-M.Non-contact Video-based Pulse Rate Measurement on a Mobile Service Robot
in: Proc. 23st IEEE Int. Symposium on Robot and Human Interactive Communication (Ro-Man 2014), Edinburgh, Scotland, UK, pp. 1056 - 1062, IEEE 2014
    * In order to use this dataset in a deep model, you should organize the files as follows:
    
    -----------------
        data/PURE/
         |   |-- 01-01/
         |      |-- 01-01/
         |      |-- 01-01.json
         |   |-- 01-02/
         |      |-- 01-02/
         |      |-- 01-02.json
         |...
         |   |-- ii-jj/
         |      |-- ii-jj/
         |      |-- ii-jj.json
    -----------------
# rPPG signal removal mode
  As stated before, this mode of operation works on PURE dataset and PhysNet extractor (for simple testing, just use `PURE_PURE_PURE_PHYSNET_BASIC.yaml`). You should first run the data preprocessing and training with this same yaml file (`train_and_test` mode with `DO_PREPROCESS` set to `True`) and then proceed to `rPPG_removal` mode. It should take a Testing part of PURE dataset (12 videos from last 2 persons) and modify them, substituting the original with the sinusuidal target signal. Then, it will run tests and evaluate videos in terms of MAE and PSNR. For customization of this sample test configuration, next chapter explains the meaning of the hyperparameters used in `rPPG_removal` mode. 
## rPPG_removal Yaml Hyperparameters
* TEST_DL_METHOD: After testing on PhysNet, test also on TS-CAN rPPG extractor. It assumes TS-CAN network is already pretrained and stored in folder `PreTrainedModels`.
* TEST_MODEL_PATH: Path to the pretrained TS-CAN model
* TARGET_SIGNAL_TYPE: Type of the target signal. Options: `RANDOM_FREQ_SINUSOID`, `LPF_GAUSS_NOISE` or `OTHER_PERSON_RPPG`
* NUM_EPOCHS: Number of epochs for which we run GD on input in modification
* LR: Learning rate during rPPG modification GD
* VIDEOS_SAVED_PATH: Path to the folders where one wish to save the modified videos
* TEST_CLASSICAL_METHODS: Boolean denoting whether classical extractors are used in testing
* CLASSICAL_TESTING_METHODS: Choice of classical rPPG extractors that will be used for testing
## Add A New Dataloader

* Step1 : Create a new python file in dataset/data_loader, e.g. MyLoader.py

* Step2 : Implement the required functions, including:

  ```python
  def preprocess_dataset(self, config_preprocess)
  ```
  ```python
  @staticmethod
  def read_video(video_file)
  ```
  ```python
  @staticmethod
  def read_wave(bvp_file):
  ```

* Step3 :[Optional] Override optional functions. In principle, all functions in BaseLoader can be override, but we **do not** recommend you to override *\_\_len\_\_, \_\_get\_item\_\_,save,load*.
* Step4 :Set or add configuration parameters.  To set paramteters, create new yaml files in configs/ .  Adding parameters requires modifying config.py, adding new parameters' definition and initial values.

## Citation

Please cite the following paper if you use the toolbox. 

Title: Deep Physiological Sensing Toolbox

Xin Liu, Xiaoyu Zhang, Girish Narayanswamy, Yuzhe Zhang, Yuntao Wang, Shwetak Patel, Daniel McDuff

https://arxiv.org/abs/2210.00716


