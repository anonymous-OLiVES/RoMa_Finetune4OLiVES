import torch
import torch.nn as nn
import torchvision.models as tvm
from romatch.utils.utils import get_autocast_params

class Projector(nn.Module):
    def __init__(self, dim, amp = False, amp_dtype = torch.float16):
        super().__init__()
        self.dim = dim
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        layers = []
        self.fc = nn.Sequential(nn.Linear(dim, dim*4, bias=False),
                                nn.BatchNorm1d(dim*4),
                                nn.ReLU(inplace=True),  # first layer
                                nn.Linear(dim*4, dim*4, bias=False),
                                nn.BatchNorm1d(dim*4),
                                nn.ReLU(inplace=True),  # second layer
                                nn.Linear(dim*4, dim*4, bias=False))  #
    def forward(self, x):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(device_type=autocast_device, enabled=autocast_enabled, dtype=autocast_dtype):
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

class Predictor(nn.Module):
    def __init__(self, fea_dim, pred_dim, amp = False, amp_dtype = torch.float16):
        super().__init__()
        self.fea_dim = fea_dim
        self.pred_dim = pred_dim
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.predictor = nn.Sequential(nn.Linear(fea_dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(pred_dim, fea_dim))
    def forward(self, x):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(device_type=autocast_device, enabled=autocast_enabled, dtype=autocast_dtype):
            return self.predictor(x)


class VGG19(nn.Module):
    def __init__(self, pretrained=True, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        if pretrained:
            weights = tvm.vgg.VGG19_BN_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.layers = nn.ModuleList(tvm.vgg19_bn(weights=weights).features[:40])
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(device_type=autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
            feats = {}
            scale = 1
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x
                    scale = scale*2
                x = layer(x)
            return feats

class CNNandDinov2(nn.Module):
    def __init__(self, cnn_kwargs = None, amp = False, dinov2_weights = None, amp_dtype = torch.float16):
        super().__init__()
        if dinov2_weights is None:
            dinov2_weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", map_location="cpu")
        from .transformer import vit_large
        vit_kwargs = dict(img_size= 518,
            patch_size= 14,
            init_values = 1.0,
            ffn_layer = "mlp",
            block_chunks = 0,
        )

        dinov2_vitl14 = vit_large(**vit_kwargs).eval()
        dinov2_vitl14.load_state_dict(dinov2_weights)
        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        self.cnn = VGG19(**cnn_kwargs)
        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.amp:
            dinov2_vitl14 = dinov2_vitl14.to(self.amp_dtype)
        self.dinov2_vitl14 = [dinov2_vitl14] # ugly hack to not show parameters to DDP
    
    
    def train(self, mode: bool = True):
        return self.cnn.train(mode)
    
    def forward(self, x, upsample = False):
        B,C,H,W = x.shape
        feature_pyramid = self.cnn(x)
        
        if not upsample:
            with torch.no_grad():
                if self.dinov2_vitl14[0].device != x.device:
                    self.dinov2_vitl14[0] = self.dinov2_vitl14[0].to(x.device).to(self.amp_dtype)
                dinov2_features_16 = self.dinov2_vitl14[0].forward_features(x.to(self.amp_dtype))
                features_16 = dinov2_features_16['x_norm_patchtokens'].permute(0,2,1).reshape(B,1024,H//14, W//14)
                del dinov2_features_16
                feature_pyramid[16] = features_16
        return feature_pyramid