import numpy as np
import torch
import set_model_vgg
#import set_model_res    # for resnet
import htk_io
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.fftpack import dct
from python_speech_features import delta       # install python_speech_features, google to find it, or you can just write your own function to compute delta

gpu_dtype = torch.cuda.FloatTensor


eps = 1e-8



def read_mv(stat):
   mean_flag = var_flag = False
   m = v = None
   with open(stat) as s:
      for line in s:
         line = line.strip()
         if len(line) < 1: continue
         if "MEAN" in line:
            mean_flag = True
            continue
         if mean_flag:
            m = list(map(float, line.split()))
            mean_flag = False
            continue
         if "VARIANCE" in line:
            var_flag = True
            continue
         if var_flag:
            v = list(map(float, line.split()))
            var_flag = False
            continue
   return np.array(m, dtype = np.float64), np.array(v, dtype = np.float64)


def org_data(utt_feat, win_size_before, win_size_after):
   frm_num, feat_dim = utt_feat.shape
   width = win_size_before + win_size_after + 1

   out_feat = np.zeros((frm_num, 1, feat_dim, width))

   utt_feat = np.pad(utt_feat, ((win_size_before, win_size_after), (0,0)), mode = 'edge')    # pad the starting and ending frames
   
   for i in range(frm_num):
      frm_idx = i + win_size_before
      block_data = utt_feat[frm_idx - win_size_before : frm_idx + win_size_after + 1, :]
      
      block_data = block_data.T
      block_data = block_data.reshape(1, block_data.shape[0], block_data.shape[1])
      
      out_feat[i] = block_data

   return out_feat


def gen_post(feat_list, stat_file, model, win_size_before = 15, win_size_after = 15, num_targets = 31):
   model.eval()             # Put the model in test mode (the opposite of model.train(), essentially)
   
   m, v = read_mv(stat_file)
   if m is None or v is None:
      raise Exception("mean or variance vector does not exist")
   
   with open(feat_list) as f:
      for line in f:
         line = line.strip()
         if len(line) < 1: continue
         print ("generating features for file", line)
         io = htk_io.fopen(line)
         utt_feat = io.getall()
         utt_feat -= m       # normalize mean
         utt_feat /= (np.sqrt(v) + eps)     # normalize var
         feat_numpy = org_data(utt_feat, win_size_before, win_size_after)
         out_feat = np.zeros((utt_feat.shape[0], num_targets))
         for i in range(feat_numpy.shape[0] // 100):     # chop the speech into shorter segments, to prevent gpu out of memory
             start_idx = i * 100
             end_idx = i * 100 + 100
             feat_chunk = feat_numpy[start_idx:end_idx]
             feat_tensor = torch.from_numpy(feat_chunk).type(gpu_dtype)
             x = Variable(feat_tensor.type(gpu_dtype), volatile = True)
             scores = model(x)
             out_feat[start_idx:end_idx] = scores.data.cpu().numpy()
         num_remain = feat_numpy.shape[0] % 100
         if num_remain > 0:
            feat_chunk = feat_numpy[-num_remain:]
            feat_tensor = torch.from_numpy(feat_chunk).type(gpu_dtype)
            x = Variable(feat_tensor.type(gpu_dtype), volatile = True)
            scores = model(x)
            out_feat[-num_remain:] = scores.data.cpu().numpy()
         
         out_feat = dct(out_feat, type=2, axis=1, norm='ortho')[:,1:numcep+1]
         out_feat_delta = delta(out_feat, 2)
         out_feat_ddelta = delta(out_feat_delta, 2)
         out_feat = np.concatenate((out_feat, out_feat_delta, out_feat_ddelta), axis = 1)   
   
         out_file = line.replace(".fea", ".mfc")
         io = htk_io.fopen(out_file, mode="wb", veclen = out_feat.shape[1])
         io.writeall(out_feat)
         print ("features saved in %s\n" %out_file)
            




if __name__ == '__main__':
   numcep = 13
   model_path = 'your/model/path/filename'
   model = set_model_vgg.setup_model(num_targets = 31)
   #model = set_model_res.resnet17(input_height = 31, input_width = 31, num_targets = 31)    # if you want to use resnet, set it here, height = frequency, width = time
   model = model.type(gpu_dtype)
   model.load_state_dict(torch.load(model_path))     # load model params
   gen_post(feat_list = 'all_fbank.lst', stat_file = 'stat_reverb', model = model, win_size_before = 15, win_size_after = 15, num_targets = 31)
