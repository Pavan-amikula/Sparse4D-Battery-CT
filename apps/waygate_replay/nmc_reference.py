"""Frozen epoch-12 NMC model evaluated on the 2000bar TIFF stack.
Six fixed regions, 75 simulated noiseless views, no training or tuning.
Different-specimen identity is unconfirmed. This is a separate-stack simulation test.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import uuid
import zipfile
import numpy as np
SIZE=128
PATCH=80
UNIT_SCALE=.4
CHECKPOINT_SHA="85d942f6d2697ab57a91c2e3204b034817e5555603183ef9bf90cfc7eaba8750"
PROTOCOL_SHA="c180e08642f780d85ac121be74732bfc492735d9e88691e7a9ed3e9326b33c62"


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def build_model(torch):
    nn=torch.nn;F=torch.nn.functional
    def block(a,b,stride=1):
        return nn.Sequential(nn.Conv3d(a,b,3,padding=1,stride=stride),nn.GELU(),
                             nn.Conv3d(b,b,3,padding=1),nn.GELU())
    class BatteryResidualUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.first=block(1,12);self.down1=block(12,24,2);self.down2=block(24,36,2)
            self.up1=block(60,24);self.up0=block(36,12);self.correction=nn.Conv3d(12,1,1)
            nn.init.zeros_(self.correction.weight);nn.init.zeros_(self.correction.bias)
        def forward(self,x):
            a=self.first(x);b=self.down1(a);c=self.down2(b)
            c=F.interpolate(c,size=b.shape[2:],mode='trilinear',align_corners=False)
            d=self.up1(torch.cat([c,b],dim=1))
            d=F.interpolate(d,size=a.shape[2:],mode='trilinear',align_corners=False)
            return x.float()+self.correction(self.up0(torch.cat([d,a],dim=1))).float()
    return BatteryResidualUNet()

def tile_positions(size=SIZE,patch=PATCH,stride=40):
    if patch>size:raise ValueError('Patch exceeds volume size.')
    positions=list(range(0,size-patch+1,stride))
    if positions[-1]!=size-patch:positions.append(size-patch)
    return positions

def tiled_predict(array,predict,patch=PATCH,stride=40):
    size=array.shape[0]
    if array.shape!=(size,)*3:raise ValueError('Expected cubic inference volume.')
    positions=tile_positions(size,patch,stride)
    h=np.maximum(np.hanning(patch),.05).astype(np.float32)
    weight=h[:,None,None]*h[None,:,None]*h[None,None,:]
    result=np.zeros(array.shape,dtype=np.float32);weights=np.zeros_like(result)
    for z in positions:
        for y in positions:
            for x in positions:
                sl=np.s_[z:z+patch,y:y+patch,x:x+patch]
                pred=predict(array[sl].astype(np.float32)/UNIT_SCALE)*UNIT_SCALE
                if pred.shape!=(patch,)*3 or not np.isfinite(pred).all():raise ValueError('Invalid model tile.')
                result[sl]+=pred*weight;weights[sl]+=weight
    if np.any(weights<=0):raise ValueError('Uncovered inference voxels.')
    return result/weights

def make_geometry(astra,g):
    # Preserve training simulation voxel pitch, not claimed NMC physical spacing.
    width = g['volume_width_mm'] * SIZE/256
    vg = astra.create_vol_geom(SIZE,SIZE,SIZE,-width/2,width/2,-width/2,width/2,-width/2,width/2)
    pg = astra.create_proj_geom('cone', *g['reduced_detector_pitch_xy_mm'],250,250,
                               np.deg2rad(np.arange(0,1200,16)*.3),
                               g['source_origin_mm'],g['source_detector_mm']-g['source_origin_mm'])
    return pg,vg,width

def reconstruct(astra,measured,pg,vg,initial=None,iterations=20):
    ids=[];algorithms=[]
    try:
        pid=astra.data3d.create('-proj3d',pg,measured);ids.append(pid)
        rid=astra.data3d.create('-vol',vg,0 if initial is None else np.ascontiguousarray(initial));ids.append(rid)
        if initial is None:
            cfg=astra.astra_dict('FDK_CUDA')
            cfg.update(ProjectionDataId=pid,ReconstructionDataId=rid)
            aid=astra.algorithm.create(cfg);algorithms.append(aid);astra.algorithm.run(aid)
        cfg=astra.astra_dict('SIRT3D_CUDA')
        cfg.update(ProjectionDataId=pid,ReconstructionDataId=rid)
        cfg['option']={'MinConstraint':0.0}
        aid=astra.algorithm.create(cfg);algorithms.append(aid)
        astra.algorithm.run(aid,iterations=iterations)
        result=astra.data3d.get(rid).astype(np.float32)
        if result.shape != (SIZE,)*3 or not np.isfinite(result).all():
            raise ValueError('Invalid SIRT reconstruction.')
        return result
    finally:
        for aid in algorithms:astra.algorithm.delete(aid)
        for ident in ids:astra.data3d.delete(ident)

def project(astra,volume,pg,vg):
    ident=None
    try:
        ident,result=astra.create_sino3d_gpu(np.ascontiguousarray(volume),pg,vg)
        if not np.isfinite(result).all():raise ValueError('Invalid projection.')
        return result.astype(np.float32)
    finally:
        if ident is not None:astra.data3d.delete(ident)

def rmse(a,b):
    d=a.astype(np.float64)-b.astype(np.float64)
    return float(np.sqrt(np.mean(d*d)))

