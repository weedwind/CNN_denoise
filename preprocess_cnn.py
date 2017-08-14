import numpy as np
import h5py
import htk_io
import os
import gc


stat_file_src = "stat_reverb"  # has global mean and variance of the reverb features 
stat_file_tgt = "stat_clean"   # has global mean and variance of the clean features
win_size_before = 15           
win_size_after = 15

out_folder_base = 'train_cnn'

chunk_hour = 0.5       # randomly select this many hours of frames from the buffer and save into a h5 chunk

buffer_hour = 40

chunk_num_frms = chunk_hour * 3600 * 1000 / 10

buffer_num_frms = buffer_hour * 3600 * 1000 / 10


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
                


def save_hd5(filename, obj):
   data_set = np.stack([pair[0] for pair in obj], axis = 0)
   label_set = np.stack([pair[1] for pair in obj], axis = 0)
   with h5py.File(filename, 'w') as hf:
      hf.create_dataset('data', data = data_set)
      hf.create_dataset('label', data = label_set)
   print ('Saved %d frames into %s' %(data_set.shape[0], filename))




def make_chunk(filename, arr, arr_len):
   
   remain_arr = []
   dump_arr = []
   prob = float(chunk_num_frms) / arr_len

   for e in arr:
      flag_choose = np.random.binomial(1, prob)
      if flag_choose == 1:
         dump_arr.append(e)
         arr_len -= 1
      else:
         remain_arr.append(e)

   save_hd5(filename, dump_arr)

   del dump_arr[:]
   del dump_arr
   gc.collect()

   return remain_arr, arr_len
   



def proc_frame(feat_list):

   if not os.path.exists(out_folder_base):
        os.makedirs(out_folder_base)

   m_src, v_src = read_mv(stat_file_src)
   if m_src is None or v_src is None:
      raise Exception("mean or variance vector for the source features does not exist")

   m_tgt, v_tgt = read_mv(stat_file_tgt)
   if m_tgt is None or v_tgt is None:
      raise Exception("mean or variance vector for the target features does not exist")

   utt_count = 0
   chunk_idx = -1
   data_cache = []
   buffer_len = 0

   f = open(feat_list, 'r')

   while True:
      if buffer_len < buffer_num_frms:
         line = f.readline()
         if line == '':
            print ('All utterances processed')
            f.close()
            break

         line = line.strip()
         if len(line) < 1: continue

         line_split = line.split()
         if len(line_split) == 2:
            src_feat_file, tgt_feat_file = line_split
         else:
            raise Exception("target feat file missing")

         io_tgt = htk_io.fopen(tgt_feat_file)
         utt_feat_tgt = io_tgt.getall()
         frm_num_tgt, feat_dim_tgt = utt_feat_tgt.shape
         utt_feat_tgt -= m_tgt         # mean normalization
         utt_feat_tgt /= (np.sqrt(v_tgt) + eps)     # var normalization

         io_src = htk_io.fopen(src_feat_file)
         utt_feat_src = io_src.getall()
         frm_num_src, feat_dim_src = utt_feat_src.shape
         
         if frm_num_src > frm_num_tgt:
            print ("%d source frames, match to %d target frames" %(frm_num_src, frm_num_tgt))

         utt_feat_src -= m_src         # mean normalization
         utt_feat_src /= (np.sqrt(v_src) + eps)     # var normalization
         utt_feat_src = np.pad(utt_feat_src, ((win_size_before, win_size_after), (0,0)), mode = 'edge')    # pad the starting and ending frames    
         start = win_size_before
         end = frm_num_tgt + win_size_before

         count = 0

         for i in range(start, end):     # process one utterance
             count += 1
             block_data = None
             block_data = utt_feat_src[i - win_size_before : i + win_size_after + 1, :]
             block_data = block_data.T
             block_data = block_data.reshape(1, block_data.shape[0], block_data.shape[1])
             label = utt_feat_tgt[i - win_size_before]

             data_cache.append( (block_data, label) )     # fill the buffer
             buffer_len += 1

         if count != frm_num_tgt:
            raise Exception("The number of processed frames %d should equal the number of frames %d in the target utterance" %(count, frm_num_tgt))
         else:
            print ("Processed %d of %d frames for file %s" %(count, frm_num_tgt, src_feat_file))
         utt_count += 1
         print (utt_count)
      else:     # output to hard drive
         chunk_idx += 1
         print ('Saving data chunk %d...' %chunk_idx)
         out_file = out_folder_base + '/' + str(chunk_idx) + '.h5'
         data_cache, buffer_len = make_chunk(out_file, data_cache, buffer_len)

   ###
   while buffer_len > 0:
      chunk_idx += 1
      print ('Saving remaining data chunk %d...' %chunk_idx)
      out_file = out_folder_base + '/' + str(chunk_idx) + '.h5'
      if buffer_len > chunk_num_frms:
         data_cache, buffer_len = make_chunk(out_file, data_cache, buffer_len)
      else:
         save_hd5(out_file, data_cache)
         buffer_len = 0
      




if __name__ == '__main__':
   proc_frame("train_src_tgt.lst")

