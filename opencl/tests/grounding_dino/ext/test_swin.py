import os
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import openvino as ov
import torch.nn.functional as F
from openvino.frontend.pytorch import ModuleExtension

ext_path = f"{os.path.dirname(os.path.abspath(__file__))}/build/libstubop.so"
onnx_path = 'model.onnx'
torch_path = 'model.xml'

###############################################################
# onnx
# https://blog.openvino.ai/blog-posts/custom-pytorch-operations
class StubAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv: torch.Tensor, mask: torch.Tensor, num_heads, scale, window_size, embed_dims) -> torch.Tensor:
        B, N, C = qkv.shape
        qkv = qkv.reshape(B, N, 3, num_heads,
                                C // num_heads // 3).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn + mask
        attn = F.softmax(attn, dim=-1)

        return (attn @ v).transpose(1, 2).reshape(B, N, C // 3)

    @staticmethod
    def symbolic(g:torch.Graph, qkv: torch.Tensor, mask: torch.Tensor, num_heads, scale, window_size, embed_dims) -> torch.Tensor:
        attr = g.op("Constant", value_t=torch.ByteTensor(list(bytes(
                    f'''
                        out_dt:0 
                        out_shape:[-1,{window_size*window_size},{embed_dims}]
                        type:WSAttention 
                        NUM_HEADS:{num_heads}
                        SCALE:{scale}
                        NUM_KV_HEADS:{num_heads}
                        HEAD_SIZE:{embed_dims}
                    ''', 
                    'utf8'))))
        return g.op("StubOP", qkv, mask, attr)            

class MyModelONNX(nn.Module):
    def __init__(self, num_heads, scale, window_size, embed_dims):
      super(MyModelONNX, self).__init__()
      self.num_heads = num_heads
      self.scale = scale
      self.window_size = window_size
      self.embed_dims = embed_dims

    def forward(self, qkv: torch.Tensor, mask: torch.Tensor):
      return StubAttention.apply(qkv, mask, self.num_heads, self.scale, self.window_size, self.embed_dims)

def get_ref_model_onnx():
    return MyModelONNX(num_heads=4, scale=0.1767766952966369, window_size=12, embed_dims=32)

def export_onnx(qkv_shape, mask_shape):
    print(f'convert onnx model to "{onnx_path}"...', end='')
    np.random.seed(324)
    torch.manual_seed(32)

    model = get_ref_model_onnx()
    qkv = Variable(torch.randn(qkv_shape))
    mask = Variable(torch.randn(mask_shape))
    model.eval()

    with torch.no_grad():
        torch.onnx.export(model, (qkv, mask), onnx_path,
                        input_names=['qkv', 'mask'],
                        output_names=['attn'])

    print('done')

###############################################################
# torch
class MyModelTorch(nn.Module):
    def __init__(self, num_heads, scale, window_size, embed_dims):
       super(MyModelTorch, self).__init__()
       self.num_heads = num_heads
       self.scale = scale
       self.window_size = window_size
       self.embed_dims = embed_dims

    def forward(self, qkv: torch.Tensor, mask: torch.Tensor):
        B, N, C = qkv.shape
        qkv = qkv.reshape(B, N, 3, self.num_heads,
                                C // self.num_heads // 3).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn + mask
        attn = F.softmax(attn, dim=-1)

        return (attn @ v).transpose(1, 2).reshape(B, N, C // 3)

def get_ref_model_torch():
    return MyModelTorch(num_heads=4, scale=0.1767766952966369, window_size=12, embed_dims=32)

def export_torch(qkv_shape, mask_shape):
    print(f'convert torch model to "{torch_path}"...', end='')
    np.random.seed(324)
    torch.manual_seed(32)

    model = get_ref_model_torch()
    qkv = Variable(torch.randn(qkv_shape))
    mask = Variable(torch.randn(mask_shape))
    model.eval()

    # https://github.com/slyalin/vllm/blob/30605c85ea2792a9145aec034e3102644a204d6d/vllm/worker/model_runner.py#L262
    with torch.no_grad():
        ov_model = ov.convert_model(
            model,
            example_input=(qkv, mask),
            extension=[
                ModuleExtension(
                    MyModelTorch,
                    target_op='StubOP',
                    evaluate=lambda module, *args, **kwargs: torch.ones(
                        list(args[0].shape[:-1]) + [args[0].shape[-1] // 3],
                        dtype=torch.float32),
                    convert=lambda module, target_op, *args, **kwargs: target_op(
                        args[0], args[1], 
                        torch.ByteTensor(list(bytes(
                        f'''
                            out_dt:0 
                            out_shape:[-1,{module.window_size*module.window_size},{module.embed_dims}]
                            type:WSAttention 
                            NUM_HEADS:{module.num_heads}
                            SCALE:{module.scale}
                            NUM_KV_HEADS:{module.num_heads}
                            HEAD_SIZE:{module.embed_dims}
                        ''', 
                        'utf8'))))
                ),
                ext_path
            ]
        )
        ov.serialize(ov_model, torch_path, f"{torch_path[:-3]}bin")

    print('done')

#####################################################################################
## Test
def test_ov(model_path, qkv, mask):
    core = ov.Core()
    core.add_extension(ext_path)
    net = core.read_model(model_path)
    # ov.serialize(net, 'result.xml', 'result.bin')
    exec_net = core.compile_model(net, 'GPU')
    req = exec_net.create_infer_request()
    results = req.infer({
        "qkv": qkv,
        "mask": mask
    })
    values = list(results.values())[0]
    return values

qkv_shape=[1, 144, 384]
mask_shape = [1, 4, 144, 144]
use_onnx = True
use_torch= True
if use_onnx:
    export_onnx(qkv_shape=qkv_shape, mask_shape=mask_shape)
if use_torch:
    export_torch(qkv_shape=qkv_shape, mask_shape=mask_shape)

qkv = np.random.random(qkv_shape)
mask = np.random.random(mask_shape)
# TODO: remove
mask = np.zeros(mask_shape, dtype=np.float32)
if use_onnx:
    print(f'test "{onnx_path}"...', end='')
    cur = test_ov(onnx_path, qkv, mask)
    ref_model = get_ref_model_onnx()
    ref = ref_model(torch.from_numpy(qkv), torch.from_numpy(mask)).detach().numpy()
    if not np.allclose(ref, cur, atol=1e-3, rtol=1e-3):
        print(f'onnx faild:\n{ref=}\n{cur=}')
    print('done.')
if use_torch:
    print(f'test "{torch_path}"...', end='')
    cur = test_ov(torch_path, qkv, mask)
    ref_model = get_ref_model_torch()
    ref = ref_model(torch.from_numpy(qkv), torch.from_numpy(mask)).detach().numpy()
    if not np.allclose(ref, cur, atol=1e-3, rtol=1e-3):
        print(f'torch failed:\n{ref=}\n{cur=}')
    print('done.')