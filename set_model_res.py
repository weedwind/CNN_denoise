import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.init as nn_init




def conv3x3(in_planes, out_planes, stride = (1,1)):
   return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride, padding = 1)     # a 3 x 3 conv layer


def conv1x1(in_planes, out_planes, stride = (1,1)):         # a 1 x 1 conv layer
   return nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride, bias = False)


def bn(num_maps, m = 0.9):
   return nn.BatchNorm2d(num_features = num_maps, momentum = m)       # a batch normalization layer



class BasicBlock(nn.Module):
   " A basic residual block without bottleneck layers "

   expansion = 1     # if in_channel * expansion != out_channel: project input

   def __init__(self, inplanes, planes, stride = (1,1), downsample = None):
       super(BasicBlock, self).__init__()
       self.conv1 = conv3x3(inplanes, planes, stride)     # the first conv layer, may have stride = 2
       self.bn1 = bn(planes)                              # the first bn layer
       self.relu1 = nn.ReLU(inplace = True)
       self.conv2 = conv3x3(planes, planes)               # the second conv layer, stride = 1
       self.downsample = downsample
       self.bn2 = bn(planes)                              # the second bn layer, applied after x + residual
       self.relu2 = nn.ReLU(inplace = True)

   def forward(self, x):
       residual = x

       out = self.conv1(x)
       out = self.bn1(out)
       out = self.relu1(out)

       out = self.conv2(out)

       if (self.downsample is not None):
          residual = self.downsample(x)      # downsample is a 1x1 conv layer to match dimension

       out += residual
       out = self.bn2(out)
       out = self.relu2(out)
       
       return out


class Bottleneck(nn.Module):
   " A bottleneck residual block "

   expansion = 4

   def __init__(self, inplanes, planes, stride = (1,1), downsample = None):
      super(Bottleneck, self).__init__()
      self.conv1 = conv1x1(inplanes, planes)       # bottleneck the input
      self.bn1 = bn(planes)         # the first bn layer
      self.relu1 = nn.ReLU(inplace = True)
      self.conv2 = conv3x3(planes, planes, stride = stride)      # the middle 3x3 conv layer, input_depth = output_depth and stride may equal 2
      self.bn2 = bn(planes)        # the middle bn layer
      self.relu2 = nn.ReLU(inplace = True)
      self.conv3 = conv1x1(planes, planes * 4)      # recover the bottleneck
      self.bn3 = bn(planes * 4)
      self.relu3 = nn.ReLU(inplace = True)
      self.downsample = downsample

   def forward(self, x):
      residual = x
  
      out = self.conv1(x)
      out = self.bn1(out)
      out = self.relu1(out)

      out = self.conv2(out)
      out = self.bn2(out)
      out = self.relu2(out)

      out = self.conv3(out)

      if (self.downsample is not None):
         residual = self.downsample(x)

      out += residual
      out = self.bn3(out)
      out = self.relu3(out)

      return out




class ResNet(nn.Module):

   def __init__(self, block_type, block_nums, input_depth, input_height, input_width, num_classes):
       super(ResNet, self).__init__()
       self.inplanes = input_depth
       self.layer64 = self.__make_layer(block_type, 64, block_nums[0], stride = (1,1))  # This layer does not have pooling
       self.layer128 = self.__make_layer(block_type, 128, block_nums[1], stride = (2,2))
       self.layer256 = self.__make_layer(block_type, 256, block_nums[2], stride = (2,2))
       self.layer512 = self.__make_layer(block_type, 512, block_nums[3], stride = (2,2))

       H, W = input_height, input_width
       
       for i in range(3):
         H = int((H -1)/2) + 1

       for j in range(3):
         W = int((W - 1)/2) + 1

       self.avgpool = nn.AvgPool2d(kernel_size = (H, W))
       self.fc = nn.Linear(512 * block_type.expansion, num_classes)

       for m in self.modules():
          if isinstance(m, nn.Conv2d):
             fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
             m.weight.data.normal_(0, math.sqrt(2.0 /(fan_in + fan_out) ))
             if m.bias is not None:
                m.bias.data.zero_()
                
          elif isinstance(m, nn.BatchNorm2d):
             m.weight.data.fill_(1)
             m.bias.data.zero_()

   

   def __make_layer(self, block_type, planes, num_blocks, stride = (1,1)):
       downsample = None
       if stride != (1,1) or self.inplanes != planes * block_type.expansion:   # project the input
           downsample = conv1x1(self.inplanes, planes * block_type.expansion, stride)     # project by 1 x 1 conv
       layers = []
       first_block = block_type(self.inplanes, planes, stride, downsample)    # setup the first residual block 
       layers.append(first_block)
       self.inplanes = planes * block_type.expansion   # this is the input depth for the next block

       for i in range(1, num_blocks):
          layers.append(block_type(self.inplanes, planes))     # the rest of the blocks

       return nn.Sequential(*layers)
   
   def forward(self, x):
      x = self.layer64(x)
      x = self.layer128(x)
      x = self.layer256(x)
      x = self.layer512(x)

      x = self.avgpool(x)
      x = x.view(x.size()[0], -1)
      x = self.fc(x)

      return x



def resnet17(input_height, input_width, num_targets):
   return ResNet(BasicBlock, [2,2,2,2], 1, input_height, input_width, num_targets)


def resnet33(input_height, input_width, num_targets):
   return ResNet(BasicBlock, [3,4,6,3], 1, input_height, input_width, num_targets)


def resnet49(input_height, input_width, num_targets):
   return ResNet(BasicBlock, [3,4,14,3], 1, input_height, input_width, num_targets)


def resnet100(input_height, input_width, num_targets):
   return ResNet(Bottleneck, [3,4,23,3], 1, input_height, input_width, num_targets)


def resnet151(input_height, input_width, num_targets):
   return ResNet(Bottleneck, [3,8,36,3], 1, input_height, input_width, num_targets)




