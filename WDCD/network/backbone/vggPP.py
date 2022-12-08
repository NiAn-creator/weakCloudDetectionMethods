import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
import torch
import torch.nn.functional as F

eps = 1e-08
batch_size = 2
final_depth = 1024

__all__ = ['VGG16_PP']
model_urls = {'vgg16PP': 'https://download.pytorch.org/models/vgg16-397923af.pth'}

class VGG16_PP(nn.Module):
    def __init__(self, init_weights=False):
        super(VGG16_PP, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)

        self.conv5 = nn.Conv2d(128, 256, 3)
        self.conv6 = nn.Conv2d(256, 256, 3)
        self.conv7 = nn.Conv2d(256, 256, 3)

        self.conv8 = nn.Conv2d(256, 512, 3)
        self.conv9 = nn.Conv2d(512, 512, 3)
        self.conv10 = nn.Conv2d(512, final_depth, 3)
        self.conv11 = nn.Conv2d(final_depth, 1, 20)  # GCP, feature size 20*20
        # self.conv11 = nn.Conv2d(final_depth, 1, 29)  # GCP, feature size 29*29

        self.fc1 = nn.Conv2d(final_depth, 2, 1)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        h,w = x.size()[2],x.size()[3]
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.max_pool2d(y, 2)
        y = F.relu(self.conv3(y))
        y = F.relu(self.conv4(y))
        y = F.max_pool2d(y, 2)
        y = F.relu(self.conv5(y))
        y = F.relu(self.conv6(y))
        y = F.relu(self.conv7(y))
        y = F.max_pool2d(y, 2)
        y = F.relu(self.conv8(y))
        y = F.relu(self.conv9(y))
        y = F.relu(self.conv10(y))      #  2,1024,20,20
        weight = self.conv11.weight     #  1,1024,20,20
        a = y * weight
        a = torch.sum(torch.sum(a, dim=-1), dim=-1) # delta, the activation value
        a11 = a.view((batch_size,final_depth,1,1))
        a_re = a11.expand(((batch_size,final_depth,230,230)))

        # # if the input is 320*320, use this line code
        # a_re = a11.expand(((batch_size,final_depth,300,300)))

        cls = self.fc1(a11)
        cls = torch.squeeze(cls)

        # a = a.expand(230, 230, final_depth)
        # a = torch.unsqueeze(a, dim=-1)
        # a = a.permute(3, 2, 1, 0)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))

        # if input size is 320*320, use this code
        # mean = torch.mean(x, dim=(2, 3))
        # mean = mean.view((batch_size, final_depth, 1, 1))
        # mean = mean.expand(((batch_size, final_depth, 300, 300)))

        # # if input size is 250*250, use this code
        mean = torch.mean(x, dim=(2, 3))
        mean = mean.view((batch_size, final_depth, 1, 1))
        mean = mean.expand(((batch_size, final_depth, 230, 230)))

        # mean = torch.mean(torch.mean(x, dim=-1), dim=-1)
        # mean1 = mean.expand(230, 230, final_depth)
        # mean11 = torch.unsqueeze(mean1, dim=-1)
        # mean111 = mean11.permute(3, 2, 1, 0)

        oup = x * a_re / (mean + eps)
        oup = F.interpolate(oup, size=(h,w),mode='bilinear', align_corners=False)

        oup = self.fc1(oup)

        return oup,cls

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def vgg16_PP(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG16_PP(**kwargs)
    if pretrained:
        checkpoint = load_state_dict_from_url(model_urls['vgg16_PP'], progress=True)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if (k in model_dict)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model
