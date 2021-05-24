from .s3dg import S3D
from .resnet_2d3d import r2d3d50, r2d3d18
from .r21d import R2Plus1DNet
from .r3d import R3DNet
from .c3d import C3D

def select_backbone(network, first_channel=3):
    param = {'feature_size': 1024}
    if network == 's3d':
        model = S3D(input_channel=first_channel)
    elif network == 'c3d':
        model = C3D()
        param['feature_size'] = 512
    elif network == 's3dg':
        model = S3D(input_channel=first_channel, gating=True)
    elif network == 'r50':
        param['feature_size'] = 2048
        model = r2d3d50(input_channel=first_channel)
    elif network == 'r21d':
        param['feature_size'] = 512
        model = R2Plus1DNet()
    elif network == 'r3d':
        param['feature_size'] = 512
        model = R3DNet()
    elif network == 'r2d3d18':
        param['feature_size'] = 256
        model = r2d3d18()
    else:
        raise NotImplementedError
    # all output features are 5d tensors after relu activation (B, C, T, H, W)
    # global average pool when using these features for feature embedding
    return model, param