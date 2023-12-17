from networks import *
from torchsummary import summary

if __name__ == '__main__':
    model = NAFNet().cuda()
    summary(model, (3, 512, 512), batch_size=4)