import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


def setup_model(num_targets, m = 0.9):
   # input 31 x 31 (F-T)
   model = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding = 1, stride=1),
                         nn.BatchNorm2d(num_features = 64, momentum = m),
                         nn.ReLU(inplace=True),
                         # 64 x 31 x 31
                         #nn.Dropout(p = 0.2),                          

                         nn.Conv2d(64, 64, kernel_size = 3, padding = 1, stride = 1),
                         nn.BatchNorm2d(num_features = 64, momentum = m),
                         nn.ReLU(inplace = True),
                         # 64 x 31 x 31
                         #nn.Dropout(p = 0.2),

                         nn.MaxPool2d(kernel_size = (2,1), stride = (2,1)),
                         # 64 x 15 x 31

                         nn.Conv2d(64, 128, kernel_size = 3, padding = 1, stride = 1),
                         nn.BatchNorm2d(num_features = 128, momentum = m),
                         nn.ReLU(inplace = True),
                         # 128 x 15 x 31
                         #nn.Dropout(p = 0.2),

                         nn.Conv2d(128, 128, kernel_size = 3, padding = 1, stride = 1),
                         nn.BatchNorm2d(num_features = 128, momentum = m),
                         nn.ReLU(inplace = True),
                         # 128 x 15 x 31
                         #nn.Dropout(p = 0.2),

                         nn.MaxPool2d(kernel_size = (2,1), stride = (2,1)),
                         # 128 x 7 x 31

                         nn.Conv2d(128, 256, kernel_size = 3, padding = 1, stride = 1),
                         nn.BatchNorm2d(num_features = 256, momentum = m),
                         nn.ReLU(inplace = True),
                         # 256 x 7 x 31
                         #nn.Dropout(p = 0.2),

                         nn.Conv2d(256, 256, kernel_size = 3, padding = 1, stride = 1),
                         nn.BatchNorm2d(num_features = 256, momentum = m),
                         nn.ReLU(inplace = True),
                         # 256 x 7 x 31
                         #nn.Dropout(p = 0.2),

                         nn.MaxPool2d(kernel_size = 2, stride = 2),
                         # 256 x 3 x 15

                         nn.Conv2d(256, 512, kernel_size = 3, padding = 1, stride = 1),
                         nn.BatchNorm2d(num_features = 512, momentum = m),
                         nn.ReLU(inplace = True),
                         # 512 x 3 x 15
                         #nn.Dropout(p = 0.2),

                         nn.Conv2d(512, 512, kernel_size = 3, padding = 1, stride = 1),
                         nn.BatchNorm2d(num_features = 512, momentum = m),
                         nn.ReLU(inplace = True),
                         # 512 x 3 x 15
                         #nn.Dropout(p = 0.2),

                         nn.Conv2d(512, 512, kernel_size = 3, padding = 1, stride = 1),
                         nn.BatchNorm2d(num_features = 512, momentum = m),
                         nn.ReLU(inplace = True),
                         # 512 x 3 x 15
                         #nn.Dropout(p = 0.2),
                         
                         nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)),
                        #nn.MaxPool2d(kernel_size = 2, stride = 2),
                         # 512 x 1 x 7

                         Flatten(),

                         nn.Linear(in_features =  1 * 7 * 512, out_features = 2048),
                         nn.BatchNorm1d(num_features = 2048, momentum = m),
                         nn.ReLU(inplace=True),
                         #nn.Dropout(p = 0.2),

                         nn.Linear(in_features =  2048, out_features = 2048),
                         nn.BatchNorm1d(num_features = 2048, momentum = m),
                         nn.ReLU(inplace=True),
                         #nn.Dropout(p = 0.2),


                         nn.Linear(in_features = 2048, out_features = num_targets)          # You should adjust the out_features to the dimension of your target

                     )

   return model

