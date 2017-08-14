# CNN_denoise
CNN learns feature mapping between corrupted and clean speech

This project uses very deep CNNs to learn feature mapping between corrupted speech features to clean ones
The working example provided here uses a database called mixer-6, the corrupted speech has reverberation.

Requirements:
1. The code is tested using Python 3.5. I recommend you should install python 3.5 by Anaconda 4.0.0 from https://repo.continuum.io/archive/. Anaconda has a lot of useful packages, such as scipy and numpy, which will save you time and avoid errors.
2. The models are created using Pytorch. Install python here http://pytorch.org/. You should have GPU and CUDA.
3. Optionally, you need a python module python_speech_features https://github.com/jameslyons/python_speech_features. But this is only useful in the last step in which you compute the delta features. But you can write your own code to do that. Also, you need scipy for the DCT operation. Anaconda provides scipy already.
4. Optionally, you need HTK. Our raw features are supposed to be in HTK format as the starting point. Also, the HTK tool HCompV will be used to compute the global mean and variance of the features.


Run the example
1. First, generate the training features, both from reverberant and clean wave:

   mkdir fb_clean && mkdir fb_reverb
   
   HCopy -A -T 1 -C convert.cfg -S reverb_wav2fb.lst
   
   HCopy -A -T 1 -C convert.cfg -S clean_wav2fb.lst
   
   The file convert.cfg tells HTK to output 31 channel log-mel features. You can generate whatever features you like, and store them in    HTK format.
2. Compute the global mean and variance from training data, for both reverberant and clean speech:

   HCompV -A -T 1 -c . -k '*.%%%' -q mv -S train_reverb_fea.lst
   
   mv fea stat_reverb
   
   HCompV -A -T 1 -c . -k '*.%%%' -q mv -S train_clean_fea.lst
   
   mv fea stat_clean
   
   The -k option tells HCompV to compute statistics from all files ending with the same last 3 characters in their filename, namely, globally, and -q mv tells HCompV to compute both mean and variance.
3. Run preprocess_cnn.py. This script convert the HTK features to the required pytorch tensor format. Later, when training the models, it's important to load a large chunk of features into memory, rather than loading one utterance at a time. proprocess_cnn.py randomly groups speech frames into a 0.5 hour chunk stored in a .h5 file. Later, these .h5 chunks will be loaded into memory to greatly speech up data loading. Also, preprocess_cnn.py does global mean and variance normalization. The reverberant speech has a longer duration than the clean data. preprocess.py simply throws away the extra frames of the reverberant data from the end.
4. Run train_cnn.py. It should be clear what the parameters mean if you know VGG and ResNet. The detailed model structures are in the module set_model_vgg.py and set_model_res.py.
5. Run test_cnn.py. This script takes an input list of corrupted utterances, normalizes their log-mel features using the global mean and variance from the reverberant training data, passes them through the trained CNN, and generates the estimated clean data. The estimated log-mel spectrum then goes through DCT and appends delta and double deltas to form the final features.
