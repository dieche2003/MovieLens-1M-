"""禁用 socket 出站连接，确认模型能从本地完整推理。"""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import socket
def blocked(*args, **kwargs):
    raise AssertionError('Offline model attempted network access')
socket.socket.connect=blocked
import torch
torch.set_num_threads(4)
from FlagEmbedding import FlagModel
import numpy as np
m=FlagModel(r'D:\xiangmu-dianying\bge-base-en-v1.5', devices='cpu',use_fp16=False,
            pooling_method='cls',normalize_embeddings=True,trust_remote_code=False)
x=m.encode(['Toy Story (1995)','Jumanji (1995)'],batch_size=2,max_length=128)
assert x.shape==(2,768) and np.isfinite(x).all()
assert np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-5)
print('PASS: local-only CPU BGE inference with outbound connections blocked.',flush=True)
