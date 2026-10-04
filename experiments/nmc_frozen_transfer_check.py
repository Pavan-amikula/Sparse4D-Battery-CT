"""Fixed NMC electrode transfer probe. Simulated views from reconstructed TIFFs.
No training or tuning. Intensity units and geometry are simulation assumptions.
The embedded network and tiling functions are copied from the supplied training script.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
import uuid
import zipfile
import numpy as np

SIZE = 128
PATCH = 80
UNIT_SCALE = 0.4
CHECKPOINT_SHA = "99769aa2fbf06029b88487bccea1053cefafacd44435bc7b7783919ada6df86d"
TRAINING_SOURCE_SHA = "be79e8ca362d14bc148f486ec8984a27062e4aa85f08ea0cb2651d35cedc468e"
METHODS = ["sirt20", "input_quantized", "model_raw", "sirt23", "model_sirt3"]


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

def read_crop(folder):
    import tifffile
    files = sorted(folder.glob('NMC_90wt_0bar_*.tif'), key=lambda p: int(re.search(r'_(\d+)\.tif$', p.name).group(1)))
    numbers = [int(re.search(r'_(\d+)\.tif$', p.name).group(1)) for p in files]
    if numbers != list(range(1,219)):
        raise ValueError('Expected exactly NMC_90wt_0bar_001.tif through _218.tif.')
    start = (len(files)-SIZE)//2
    volume = np.empty((SIZE,)*3, dtype=np.float32)
    sources = []
    for i,path in enumerate(files[start:start+SIZE]):
        a = tifffile.imread(path)
        if a.shape != (2048,2048) or a.dtype != np.uint16:
            raise ValueError('Unexpected TIFF shape/type: '+path.name)
        # Fixed conversion, no histogram fitting or performance-based selection.
        volume[i] = a[960:1088,960:1088].astype(np.float32)*(UNIT_SCALE/65535.0)
        sources.append({'name':path.name, 'sha256':sha(path)})
        if (i+1)%32 == 0:
            print(f'Read {i+1}/{SIZE} central slices', flush=True)
    if not np.isfinite(volume).all() or float(np.std(volume)) < 1e-8:
        raise ValueError('Central block is constant or invalid; no automatic alternative crop will be selected.')
    return volume, {'files_total':len(files), 'z_indices_zero_based':[start,start+SIZE],
                    'y_indices_zero_based':[960,1088], 'x_indices_zero_based':[960,1088],
                    'intensity_rule':'uint16 / 65535 * 0.4; arbitrary simulation attenuation units',
                    'source_files':sources, 'reference_range':[float(volume.min()),float(volume.max())]}


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


def check_checkpoint(checkpoint):
    cfg=checkpoint['config']
    if (checkpoint['epoch'] != 8 or checkpoint.get('synthetic_only') is not True
        or cfg['architecture'] != 'BatteryResidualUNet1ch_12_24_36_zero_head'
        or cfg['unit_scale_per_mm'] != UNIT_SCALE or cfg['patch_size'] != PATCH):
        raise ValueError('Expected the reviewed synthetic-only epoch-8 battery model.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--geometry-report',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=Path.cwd()/'results'/'battery_extension')
    parser.add_argument('--open',action='store_true')
    args=parser.parse_args()
    import astra
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not astra.use_cuda() or not torch.cuda.is_available():
        raise ValueError('Run in the CUDA ASTRA/PyTorch environment used for battery training.')
    started=perf_counter()
    checkpoint_path=args.training/'best_model.pt'
    if sha(checkpoint_path) != CHECKPOINT_SHA:
        raise ValueError('Checkpoint differs from the previously evaluated epoch-8 model.')
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    check_checkpoint(checkpoint)
    geometry_report=json.loads(args.geometry_report.read_text(encoding='utf-8'))
    if (geometry_report.get('candidate_iterations') != 20
            or geometry_report.get('fixed_iteration_counts') != [3,10,20]):
        raise ValueError('Select the earlier reviewed SIRT iteration report.')
    g=geometry_report['geometry']
    if astra.__version__ != geometry_report['astra_version']:
        raise ValueError('ASTRA version differs from the earlier reconstruction environment.')
    model=build_model(torch).cuda()
    model.load_state_dict(checkpoint['model_state'],strict=True);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    print('NMC FROZEN TRANSFER | one fixed 128-cubed crop | SIMULATED views | no training',flush=True)
    target,crop=read_crop(args.root)
    pg,vg,width=make_geometry(astra,g)
    print('Generating 75 noiseless simulated cone-beam views...',flush=True)
    measured=project(astra,target,pg,vg)
    print('Running FDK + SIRT20...',flush=True)
    initial=reconstruct(astra,measured,pg,vg)
    quantized=initial.astype(np.float16).astype(np.float32)
    if not np.isfinite(quantized).all():raise ValueError('Float16 overflow.')
    tiles=0
    expected=len(tile_positions(SIZE))**3
    def predict(patch):
        nonlocal tiles
        with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16,
                                                enabled=checkpoint['config']['mixed_precision']):
            x=torch.from_numpy(np.ascontiguousarray(patch[None,None])).cuda()
            pred=model(x)[0,0].float().cpu().numpy()
        tiles+=1
        if tiles%9 == 0:print(f'Model tiles: {tiles}/{expected}',flush=True)
        return pred
    raw=tiled_predict(quantized,predict)
    if tiles != expected:raise ValueError('Unexpected tile count.')
    print('Applying three additional SIRT steps to both comparison methods...',flush=True)
    physics23=reconstruct(astra,measured,pg,vg,initial,3)
    corrected=reconstruct(astra,measured,pg,vg,raw,3)
    volumes={'sirt20':initial,'input_quantized':quantized,'model_raw':raw,
             'sirt23':physics23,'model_sirt3':corrected}
    # Evaluation begins only after every fixed output is complete. No candidate selection.
    metrics={name:{'voxel_rmse':rmse(a,target),
                   'fit_projection_rmse':rmse(project(astra,a,pg,vg),measured)}
             for name,a in volumes.items()}
    def gain(a,b):
        base=metrics[a]['voxel_rmse']
        return 100*(base-metrics[b]['voxel_rmse'])/base if base>0 else None
    if sha(checkpoint_path) != CHECKPOINT_SHA:raise ValueError('Checkpoint changed during evaluation.')
    args.out.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run=args.out/f'nmc_transfer_{stamp}_{uuid.uuid4().hex[:6]}'
    run.mkdir()
    display_names=['reference','sirt20','model_raw','sirt23','model_sirt3']
    display={'reference':target,**volumes}
    low,high=np.percentile(target,[1,99])
    if high<=low:low,high=float(target.min()),float(target.max())
    fig,axes=plt.subplots(3,5,figsize=(15,9),layout='constrained')
    titles=['NMC TIFF reference','SIRT20','Frozen model','SIRT23','Model + SIRT3']
    slices={}
    for col,(name,title) in enumerate(zip(display_names,titles)):
        for row,(direction,sl) in enumerate([('axial',np.s_[64,:,:]),('coronal',np.s_[:,64,:]),
                                             ('sagittal',np.s_[:,:,64])]):
            a=display[name][sl];slices[name+'_'+direction]=a
            axes[row,col].imshow(a,cmap='gray',vmin=low,vmax=high)
            axes[row,col].set_title(title+' | '+direction,fontsize=9);axes[row,col].axis('off')
    fig.suptitle('NMC electrode: fixed central crop | simulated views | same display scale')
    fig.savefig(run/'nmc_comparison.png',dpi=150);plt.close(fig)
    np.savez_compressed(run/'review_slices.npz',**slices)
    report={'status':'fixed_nmc_simulated_transfer_evaluation','created_utc':datetime.now(timezone.utc).isoformat(),
            'checkpoint_sha256':CHECKPOINT_SHA,'checkpoint_epoch':8,'training_performed':False,
            'embedded_architecture_source_sha256':TRAINING_SOURCE_SHA,
            'geometry_report_sha256':sha(args.geometry_report),'geometry_source':g,
            'simulation_width':width,'nmc_physical_spacing_known':False,
            'crop':crop,'volume_shape':[SIZE]*3,'metrics':metrics,
            'raw_model_voxel_rmse_reduction_vs_sirt20_percent':gain('sirt20','model_raw'),
            'model_sirt3_voxel_rmse_reduction_vs_sirt23_percent':gain('sirt23','model_sirt3'),
            'protocol':{'input_views':75,'angles_degrees':(np.arange(0,1200,16)*.3).tolist(),
                        'noise':'none','sirt_min_constraint':0,'tiling_patch':80,'tiling_stride':40,
                        'tiles':tiles,'input_float16_roundtrip':True,'performance_based_selection':False},
            'total_seconds':perf_counter()-started,'torch_version':str(torch.__version__),
            'astra_version':astra.__version__,'script_sha256':sha(Path(__file__)),
            'limitations':['One central block from one NMC electrode stack, not multiple independent scans.',
                'Reference is a reconstructed TIFF volume, not independently measured ground truth.',
                'Projections are simulated from the reference with matching forward/inverse geometry and no noise.',
                'Intensity conversion and geometry are simulation assumptions, not calibrated NMC attenuation or voxel spacing.',
                'Fit projection RMSE uses reconstruction views; it is not a held-out projection metric.',
                'This checks transfer from cylindrical battery phantoms to electrode microstructure; degradation is possible.',
                'Results do not establish real acquisition performance, defect detection, battery health, or time-resolved 4D accuracy.',
                'This evaluated block is consumed test data; do not tune on it and report it as a fresh independent test.']}
    (run/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    bundle=run/'nmc_transfer_review.zip'
    with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ['report.json','nmc_comparison.png','review_slices.npz']:z.write(run/name,name)
        z.write(Path(__file__),'nmc_frozen_transfer_check.py')
    with zipfile.ZipFile(bundle) as z:
        if z.testzip() is not None:raise ValueError('ZIP verification failed.')
    print('\nNMC TRANSFER SUMMARY | simulated reconstruction, arbitrary intensity units')
    for name in METHODS:
        print(f"{name}: voxel RMSE {metrics[name]['voxel_rmse']:.6f}; fit projection RMSE {metrics[name]['fit_projection_rmse']:.6f}")
    print(f'Raw model vs SIRT20 voxel RMSE reduction: {gain("sirt20","model_raw"):.2f}%')
    print(f'Model + SIRT3 vs SIRT23 voxel RMSE reduction: {gain("sirt23","model_sirt3"):.2f}%')
    print('Positive = improvement; negative = worse. No real measured projection test was performed.')
    print('Upload ZIP: '+str(bundle.resolve()))
    if args.open and os.name=='nt':os.startfile(str((run/'nmc_comparison.png').resolve()))


if __name__=='__main__':
    try:main()
    except (OSError,ValueError,KeyError,RuntimeError,ImportError,zipfile.BadZipFile) as exc:
        print('NMC test stopped: '+str(exc),file=sys.stderr)
        raise SystemExit(1)
