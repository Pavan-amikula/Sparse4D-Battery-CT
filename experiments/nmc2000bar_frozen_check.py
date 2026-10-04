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

def read_regions(root,training_protocol):
    import tifffile
    files=[root/f'NMC_90wt_2000bar_{i:03d}.tif' for i in range(1,219)]
    if not all(p.is_file() for p in files):raise ValueError('Expected all 218 NMC_90wt_2000bar slices.')
    rows=[{'case_id':f'external_{i:03d}','z':[45,173],'y':[y,y+SIZE],'x':[1728,1856]}
          for i,y in enumerate([256,448,640,832,1024,1216])]
    arrays=np.empty((6,SIZE,SIZE,SIZE),dtype=np.float32)
    sources=[]
    for zi,path in enumerate(files[45:173]):
        a=tifffile.imread(path)
        if a.shape!=(2048,2048) or a.dtype!=np.uint16:raise ValueError('Unexpected TIFF shape/dtype: '+path.name)
        for i,row in enumerate(rows):
            arrays[i,zi]=a[row['y'][0]:row['y'][1],1728:1856].astype(np.float32)*(UNIT_SCALE/65535)
        sources.append({'name':path.name,'sha256':sha(path)})
        if (zi+1)%32==0:print(f'Read {zi+1}/128 source slices',flush=True)
    old=[r['sha256'] for r in training_protocol['source_files']]
    if [r['sha256'] for r in sources]==old:
        raise ValueError('Selected TIFFs are byte-identical to the 0bar source. Not a new stack test.')
    for row,a in zip(rows,arrays):
        if not np.isfinite(a).all() or float(a.std())<1e-8:
            raise ValueError('Constant or invalid fixed region: '+row['case_id']+'. No automatic crop replacement.')
    return arrays,rows,sources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=Path.cwd()/'results'/'battery_extension')
    parser.add_argument('--open',action='store_true')
    args=parser.parse_args()
    import astra
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not astra.use_cuda() or not torch.cuda.is_available():raise ValueError('CUDA ASTRA and PyTorch are required.')
    started=perf_counter()
    checkpoint_path=args.training/'best_model.pt'
    protocol_path=args.training/'protocol.json'
    if sha(checkpoint_path)!=CHECKPOINT_SHA or sha(protocol_path)!=PROTOCOL_SHA:
        raise ValueError('Use the reviewed, fixed epoch-12 NMC checkpoint and protocol.')
    training=json.loads((args.training/'report.json').read_text(encoding='utf-8'))
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    if (training['checkpoint_sha256']!=CHECKPOINT_SHA or training['best_epoch']!=12
        or training['protocol_sha256']!=PROTOCOL_SHA):raise ValueError('Training report identity differs.')
    if astra.__version__!=training['astra_version'] or str(torch.__version__)!=training['torch_version']:
        raise ValueError('Run with the same ASTRA/PyTorch versions as NMC training.')
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    cfg=checkpoint['config']
    if (checkpoint['epoch']!=12 or checkpoint['protocol_sha256']!=PROTOCOL_SHA
        or cfg['architecture']!='BatteryResidualUNet1ch_12_24_36_zero_head'
        or cfg['patch_size']!=PATCH or cfg['unit_scale']!=UNIT_SCALE):
        raise ValueError('Unexpected checkpoint architecture or scaling.')
    model=build_model(torch).cuda();model.load_state_dict(checkpoint['model_state'],strict=True)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    g=protocol['geometry_source'];pg,vg,width=make_geometry(astra,g)
    print('NMC 2000BAR FROZEN CHECK | epoch 12 | 6 fixed regions | simulated views | no training',flush=True)
    targets,rows,sources=read_regions(args.root,protocol)
    tests=[];first=None
    def predict(patch):
        with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16,enabled=cfg['mixed_precision']):
            x=torch.from_numpy(np.ascontiguousarray(patch[None,None])).cuda()
            return model(x)[0,0].float().cpu().numpy()
    for i,(target,row) in enumerate(zip(targets,rows)):
        print(f'Evaluating region {i+1}/6...',flush=True)
        measured=project(astra,target,pg,vg)
        initial=reconstruct(astra,measured,pg,vg)
        quantized=initial.astype(np.float16).astype(np.float32)
        if not np.isfinite(quantized).all():raise ValueError('Input quantization overflow.')
        raw=tiled_predict(quantized,predict)
        # Match the quantized SIRT20 input/control used in the NMC training evaluation.
        physics23=reconstruct(astra,measured,pg,vg,quantized,3)
        corrected=reconstruct(astra,measured,pg,vg,raw,3)
        volumes={'sirt20':quantized,'model_raw':raw,'sirt23':physics23,'model_sirt3':corrected}
        metrics={name:{'voxel_rmse':rmse(v,target),
                       'fit_projection_rmse':rmse(project(astra,v,pg,vg),measured)} for name,v in volumes.items()}
        tests.append({'region':row,'metrics':metrics,'input_quantization_max_abs':float(np.max(abs(initial-quantized)))})
        print(row['case_id']+' | '+' | '.join(f'{k} {v["voxel_rmse"]:.6f}' for k,v in metrics.items()),flush=True)
        if first is None:first={'reference':target.copy(),**volumes}
    if sha(checkpoint_path)!=CHECKPOINT_SHA:raise ValueError('Checkpoint changed during evaluation.')
    means={k:{metric:float(np.mean([t['metrics'][k][metric] for t in tests]))
              for metric in ['voxel_rmse','fit_projection_rmse']} for k in volumes}
    def gain(a,b):
        x=means[a]['voxel_rmse']
        return 100*(x-means[b]['voxel_rmse'])/x if x>0 else None
    improved=sum(t['metrics']['model_sirt3']['voxel_rmse']<t['metrics']['sirt23']['voxel_rmse'] for t in tests)
    args.out.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run=args.out/f'nmc2000bar_frozen_{stamp}_{uuid.uuid4().hex[:6]}';run.mkdir()
    fig,axes=plt.subplots(3,5,figsize=(15,9),layout='constrained')
    low,high=np.percentile(first['reference'],[1,99]);slices={}
    if high<=low:low,high=float(first['reference'].min()),float(first['reference'].max())
    for col,(name,a) in enumerate(first.items()):
        for row,(direction,sl) in enumerate([('axial',np.s_[64,:,:]),('coronal',np.s_[:,64,:]),('sagittal',np.s_[:,:,64])]):
            slices[name+'_'+direction]=a[sl]
            axes[row,col].imshow(a[sl],cmap='gray',vmin=low,vmax=high)
            axes[row,col].set_title(name+' | '+direction,fontsize=9);axes[row,col].axis('off')
    fig.suptitle('2000bar stack | frozen 0bar NMC model | first fixed region | common display scale')
    fig.savefig(run/'nmc2000bar_comparison.png',dpi=150);plt.close(fig)
    np.savez_compressed(run/'review_slices.npz',**slices)
    report={'status':'frozen_nmc2000bar_separate_stack_simulation_test',
            'created_utc':datetime.now(timezone.utc).isoformat(),'checkpoint_sha256':CHECKPOINT_SHA,
            'checkpoint_epoch':12,'training_performed':False,'training_protocol_sha256':PROTOCOL_SHA,
            'training_report_sha256':sha(args.training/'report.json'),'script_sha256':sha(Path(__file__)),
            'source_files':sources,'different_specimen_confirmed':False,'test_cases':tests,'mean_metrics':means,
            'raw_model_voxel_rmse_reduction_vs_sirt20_percent':gain('sirt20','model_raw'),
            'model_sirt3_voxel_rmse_reduction_vs_sirt23_percent':gain('sirt23','model_sirt3'),
            'model_sirt3_better_regions_vs_sirt23':improved,
            'geometry_source':g,'simulation_width':width,
            'protocol':{'views':75,'angles_degrees':(np.arange(0,1200,16)*.3).tolist(),'noise':'none',
                        'intensity_rule':'uint16 / 65535 * 0.4; arbitrary simulation attenuation units',
                        'physical_calibration_known':False,'model_patch':80,'model_stride':40,'model_tiles_per_region':27,
                        'input_float16_roundtrip':True,'sirt_min_constraint':0,'extra_sirt_steps_both_controls':3,
                        'performance_based_selection':False},
            'total_seconds':perf_counter()-started,'torch_version':str(torch.__version__),'astra_version':astra.__version__,
            'limitations':['Six regions from one 2000bar TIFF stack are not six independent specimens.',
                'The physical relationship between 0bar and 2000bar specimens is unconfirmed; this is a separate-stack test.',
                'Reference volumes are reconstructed TIFF data, not independent physical ground truth.',
                'Noiseless simulated projections use matching forward/inverse geometry, which can overestimate real acquisition performance.',
                'Intensity conversion and geometry are simulation assumptions, not calibrated NMC attenuation or voxel spacing.',
                'Fit projection RMSE uses reconstruction views and is not a held-out projection metric.',
                'No training, tuning, crop selection by performance, or weight updates were performed.',
                'These six regions are now consumed evaluation data; further tuning requires another independent evaluation.',
                'This does not establish defect detection, battery health, or time-resolved 4D reconstruction accuracy.']}
    (run/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    bundle=run/'nmc2000bar_review.zip'
    with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ['report.json','nmc2000bar_comparison.png','review_slices.npz']:z.write(run/name,name)
        z.write(Path(__file__),'nmc2000bar_frozen_check.py')
    with zipfile.ZipFile(bundle) as z:
        if z.testzip() is not None:raise ValueError('Review ZIP integrity failed.')
    print('\nNMC 2000BAR SUMMARY | FROZEN MODEL | SIMULATED VIEWS')
    for k,v in means.items():print(f'{k}: mean voxel RMSE {v["voxel_rmse"]:.6f}; mean fit projection RMSE {v["fit_projection_rmse"]:.6f}')
    print(f'Raw model reduction vs SIRT20: {gain("sirt20","model_raw"):.2f}%')
    print(f'Model + SIRT3 reduction vs SIRT23: {gain("sirt23","model_sirt3"):.2f}%')
    print(f'Hybrid lower voxel error in {improved}/6 regions. Positive percent = better; negative = worse.')
    print('Separate-stack simulation test; different-specimen independence remains unconfirmed.')
    print('Upload ZIP: '+str(bundle.resolve()))
    if args.open and os.name=='nt':os.startfile(str((run/'nmc2000bar_comparison.png').resolve()))


if __name__=='__main__':
    try:main()
    except (OSError,ValueError,KeyError,RuntimeError,ImportError,zipfile.BadZipFile) as exc:
        print('NMC 2000bar check stopped: '+str(exc),file=sys.stderr)
        raise SystemExit(1)
