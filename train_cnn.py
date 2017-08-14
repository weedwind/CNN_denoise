import torch
import gc
import os
import glob
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
import copy
import shutil
import set_model_vgg      # if you want to use ResNet, import set_model_res
#import set_model_res
from random import sample
import numpy as np
import h5py

gpu_dtype = torch.cuda.FloatTensor

train_dir_base = 'train_cnn'

data_files = glob.glob(train_dir_base + '/' + '*.h5')

num_chunks = len(data_files)

load_chunks_every = 3       # load this many chunks into memory every time


def get_loader(chunk_list):
   data = []
   label = []
   for f in chunk_list:
       print ('Loading data from %s' %f)
       with h5py.File(f, 'r') as hf:
          data.append(np.asarray(hf['data']))
          label.append(np.asarray(hf['label']))
   data = torch.FloatTensor(np.concatenate(data, axis = 0))
   label = torch.FloatTensor(np.concatenate(label, axis = 0))
   print ('Total %d frames loaded' %data.size(0))

   dset_train = TensorDataset(data, label)
   loader_train = DataLoader(dset_train, batch_size = 256, shuffle = True, num_workers = 10, pin_memory = False)
   return loader_train



def train_one_epoch(model, loss_fn, optimizer, print_every = 100):
     
  data_list = sample(data_files, num_chunks)
    
  model.train()
  t = 0
   
  for i in range(0, num_chunks, load_chunks_every):
    chunk_list = data_list[i: i + load_chunks_every]
    loader = get_loader(chunk_list)
    for x, y in loader:
        x_var = Variable(x.type(gpu_dtype))
        y_var = Variable(y.type(gpu_dtype))
        
        scores = model(x_var)
            
        loss = loss_fn(scores, y_var)
        if (t + 1) % print_every == 0:
          print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t += 1
    del loader
    gc.collect()



#def check_accuracy(model, loader):
#    total_error = 0
#    num_samples = 0
#    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
#    for x, y in loader:
#        x_var = Variable(x.type(gpu_dtype), volatile=True)

#        scores = model(x_var).data.cpu()
#        batch_loss = ((scores - y)**2).sum()
#        batch_size = scores.size(0) * scores.size(1)
#        total_error += batch_loss
#        num_samples += batch_size
#    acc = float(total_error) / num_samples
#    print 'validation error is %.4f' %(acc,)
#    return -acc




def adjust_learning_rate(optimizer, decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay





def train_epochs(model, loss_fn, init_lr, model_dir):
   if os.path.exists(model_dir):
      shutil.rmtree(model_dir)
   os.makedirs(model_dir)

   optimizer = optim.Adam(model.parameters(), lr = init_lr)     # setup the optimizer
   
   learning_rate = init_lr
   max_iter = 5
   start_halfing_iter = 2
   halfing_factor = 0.1
   
   count = 0
   half_flag = False

   while count < max_iter:
     count += 1

     if count >= start_halfing_iter:
        half_flag = True

     print ("Starting epoch", count)
     

     if half_flag:
        learning_rate *= halfing_factor
        adjust_learning_rate(optimizer, halfing_factor)     # decay learning rate

     model_path = model_dir + '/epoch' + str(count) + '_lr' + str(learning_rate) + '.pkl'
     train_one_epoch(model, loss_fn, optimizer)      # train one epoch
     torch.save(model.state_dict(), model_path)
 

   print ("End training")  
     
         

    
if __name__ == '__main__':
   model = set_model_vgg.setup_model(num_targets = 31)        # I used 31 features as targets, put yours here, also adjust the input dimensions in set_model_vgg.py
   #model = set_model_res.resnet17(input_height = 31, input_width = 31, num_targets = 31)    # if you want to use resnet, set it here, height = frequency, width = time
   model = model.type(gpu_dtype)

   loss_fn = nn.MSELoss()
   loss_fn = loss_fn.type(gpu_dtype)


   train_epochs(model, loss_fn, 1e-3, 'weights_cnn')


