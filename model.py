import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import onnxruntime as ort


class Resnet50FPN(nn.Module):
    def __init__(self):
        super(Resnet50FPN, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        children = list(self.resnet.children())
        self.conv1 = nn.Sequential(*children[:4])
        self.conv2 = children[4]
        self.conv3 = children[5]
        self.conv4 = children[6]
    def forward(self, im_data):
        feat = OrderedDict()
        feat_map = self.conv1(im_data)
        feat_map = self.conv2(feat_map)
        feat_map3 = self.conv3(feat_map)
        feat_map4 = self.conv4(feat_map3)
        feat['map3'] = feat_map3
        feat['map4'] = feat_map4
        return feat


class Resnet50FPNONNX:
    def __init__(self, onnx_path='resnet50fpn_extractor.onnx', use_gpu=False):
        self.onnx_path = onnx_path
        providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def __call__(self, im_data):
        # im_data is torch tensor, convert to numpy
        input_data = im_data.detach().cpu().numpy()
        outputs = self.session.run(self.output_names, {self.input_name: input_data})
        # outputs: feat_map3, feat_map4
        feat = OrderedDict()
        feat['map3'] = torch.from_numpy(outputs[0]).to(im_data.device)
        feat['map4'] = torch.from_numpy(outputs[1]).to(im_data.device)
        return feat


class CountRegressor(nn.Module):
    def __init__(self, input_channels,pool='mean'):
        super(CountRegressor, self).__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, 7, padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )

    def forward(self, im):
        num_sample =  im.shape[0]
        if num_sample == 1:
            output = self.regressor(im.squeeze(0))
            if self.pool == 'mean':
                output = torch.mean(output, dim=(0),keepdim=True)
                return output
            elif self.pool == 'max':
                output, _ = torch.max(output, 0,keepdim=True)
                return output
        else:
            for i in range(0,num_sample):
                output = self.regressor(im[i])
                if self.pool == 'mean':
                    output = torch.mean(output, dim=(0),keepdim=True)
                elif self.pool == 'max':
                    output, _ = torch.max(output, 0,keepdim=True)
                if i == 0:
                    Output = output
                else:
                    Output = torch.cat((Output,output),dim=0)
            return Output


class CountRegressor_ONNX:
    def __init__(self, onnx_path='count_regressor.onnx', use_gpu=False, pool='mean'):
        self.onnx_path = onnx_path
        self.pool = pool
        providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, im):
        # im shape [batch, channels, H, W], but since num_sample=1, squeeze to [channels, H, W]

        input_data = im.squeeze(0).detach().cpu().numpy()
        output = self.session.run([self.output_name], {self.input_name: input_data})[0]
        output = torch.from_numpy(output).to(im.device)
        output = torch.mean(output, dim=0, keepdim=True)
        return output


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):                
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


            
            