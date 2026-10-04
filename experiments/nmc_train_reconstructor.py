"""One-scan NMC electrode pilot with fixed spatially separated regions.

Requires nmc_frozen_transfer_check.py alongside this file. No prior weights are
loaded. All projections are simulated noiseless views, not acquired X-rays.
Training, validation and testing share one specimen; this cannot establish
between-specimen generalization. The previously evaluated central crop is excluded.
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
SEED=20261004


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def regions():
    result=[]
    for split,xs in [('train',[256,448,640,832]),('validation',[1408]),('test',[1728])]:
        for y in [256,448,640,832,1024,1216]:
            for x in xs:
                result.append({'case_id':f'case_{len(result):03d}','split':split,
                               'z':[45,173],'y':[y,y+SIZE],'x':[x,x+SIZE]})
    check_regions(result)
    return result


def overlap(a,b):
    return all(max(a[k][0],b[k][0])<min(a[k][1],b[k][1]) for k in ['z','y','x'])


def check_regions(rows):
    excluded={'z':[45,173],'y':[960,1088],'x':[960,1088]}
    assert len(rows)==36
    assert {s:sum(r['split']==s for r in rows) for s in ['train','validation','test']}=={'train':24,'validation':6,'test':6}
    for i,a in enumerate(rows):
        if overlap(a,excluded):raise ValueError('Region overlaps the consumed central block.')
        for b in rows[i+1:]:
            if overlap(a,b):raise ValueError('Regions overlap.')
    # A minimum 320-pixel separation between training and validation strips,
    # and 192 pixels between validation and test strips.


def read_regions(root,rows):
    import tifffile
    files=[root/f'NMC_90wt_0bar_{i:03d}.tif' for i in range(1,219)]
    if not all(p.is_file() for p in files):raise ValueError('Expected all 218 NMC_90wt_0bar TIFFs.')
    arrays=np.empty((36,SIZE,SIZE,SIZE),dtype=np.float32)
    sources=[]
    for z,path in enumerate(files[45:173]):
        image=tifffile.imread(path)
        if image.shape!=(2048,2048) or image.dtype!=np.uint16:
            raise ValueError('Unexpected TIFF dimensions or dtype: '+path.name)
        for i,row in enumerate(rows):
            arrays[i,z]=image[row['y'][0]:row['y'][1],row['x'][0]:row['x'][1]].astype(np.float32)*(UNIT_SCALE/65535)
        sources.append({'name':path.name,'sha256':sha(path)})
        if (z+1)%32==0:print(f'Read {z+1}/128 source slices',flush=True)
    for row,a in zip(rows,arrays):
        if not np.isfinite(a).all() or float(a.std())<1e-8:
            raise ValueError('Fixed region is constant or invalid: '+row['case_id']+'. No automatic crop substitution.')
    return arrays,sources


def patches(a,b,rng,batch=2,augment=False):
    xs=[];ys=[]
    for _ in range(batch):
        pos=rng.integers(0,SIZE-PATCH+1,size=3)
        sl=tuple(slice(int(p),int(p)+PATCH) for p in pos)
        x=a[sl];y=b[sl]
        if augment:
            k=int(rng.integers(4));x=np.rot90(x,k,axes=(1,2));y=np.rot90(y,k,axes=(1,2))
            if rng.random()<.5:x=np.flip(x,0);y=np.flip(y,0)
        xs.append(np.ascontiguousarray(x,dtype=np.float32)/UNIT_SCALE)
        ys.append(np.ascontiguousarray(y,dtype=np.float32)/UNIT_SCALE)
    return np.stack(xs)[:,None],np.stack(ys)[:,None]


def loss(torch,pred,target):
    pred=pred.float();target=target.float()
    gradient=sum((torch.diff(pred,dim=d)-torch.diff(target,dim=d)).abs().mean() for d in (2,3,4))/3
    return (pred-target).abs().mean()+.1*gradient


def load_pair(run,row):
    with np.load(run/'pairs'/(row['case_id']+'.npz'),allow_pickle=False) as z:
        return z['input'].astype(np.float32),z['target'].astype(np.float32)


def validate(torch,model,run,rows):
    model.eval();values=[]
    with torch.inference_mode():
        for i,row in enumerate(rows):
            a,b=load_pair(run,row);rng=np.random.default_rng(SEED+50000+i)
            for _ in range(4):
                x,y=patches(a,b,rng)
                x=torch.from_numpy(x).cuda();y=torch.from_numpy(y).cuda()
                with torch.autocast(device_type='cuda',dtype=torch.float16):pred=model(x)
                value=float(loss(torch,pred,y).item())
                if not np.isfinite(value):raise ValueError('Nonfinite validation loss.')
                values.append(value)
    return float(np.mean(values))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--geometry-report',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--out',type=Path,default=Path.cwd()/'results'/'battery_extension')
    parser.add_argument('--open',action='store_true')
    args=parser.parse_args()
    if not 1<=args.epochs<=100:raise ValueError('Epoch count must be 1..100.')
    import astra
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import nmc_frozen_transfer_check as base
    if not astra.use_cuda() or not torch.cuda.is_available():raise ValueError('CUDA ASTRA and PyTorch are required.')
    if (base.SIZE,base.PATCH,base.UNIT_SCALE)!=(SIZE,PATCH,UNIT_SCALE):raise ValueError('NMC helper constants differ.')
    started=perf_counter()
    geometry_report=json.loads(args.geometry_report.read_text(encoding='utf-8'))
    if (geometry_report.get('candidate_iterations')!=20
        or geometry_report.get('fixed_iteration_counts')!=[3,10,20]):raise ValueError('Select the reviewed SIRT20 iteration report.')
    if astra.__version__!=geometry_report['astra_version']:raise ValueError('ASTRA version changed.')
    g=geometry_report['geometry'];pg,vg,width=base.make_geometry(astra,g)
    rows=regions();args.out.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run=args.out/f'nmc_training_{stamp}_{uuid.uuid4().hex[:6]}';run.mkdir();(run/'pairs').mkdir()
    protocol={'status':'preparation_started','regions':rows,'epochs':args.epochs,
              'excluded_consumed_central_block':{'z':[45,173],'y':[960,1088],'x':[960,1088]},
              'split_type':'spatial separation within one scan; not independent specimens',
              'views':75,'projection_noise':'none','intensity_rule':'uint16 / 65535 * 0.4',
              'geometry_source':g,'simulation_width':width,'physical_calibration_known':False,
              'script_sha256':sha(Path(__file__)),'helper_sha256':sha(Path(base.__file__)),
              'training_seed':SEED,'architecture':'BatteryResidualUNet1ch_12_24_36_zero_head',
              'initialization':'new zero correction head, no previous checkpoint loaded',
              'selection':'fixed validation patch L1 + 0.1 gradient L1 only; epoch 0 eligible'}
    write_json(run/'protocol.json',protocol)
    print('NMC PAIR PREPARATION | 24 train / 6 validation / 6 test regions | ONE SCAN',flush=True)
    targets,sources=read_regions(args.root,rows)
    protocol['source_files']=sources
    for i,(row,target) in enumerate(zip(rows,targets)):
        measured=base.project(astra,target,pg,vg)
        initial=base.reconstruct(astra,measured,pg,vg)
        # Match the storage-format input used by the preceding model probe.
        quantized=initial.astype(np.float16).astype(np.float32)
        if not np.isfinite(quantized).all():raise ValueError('Input quantization overflow.')
        path=run/'pairs'/(row['case_id']+'.npz')
        np.savez_compressed(path,input=quantized,target=target)
        row['pair_sha256']=sha(path)
        print(f'Prepared {i+1}/36 | {row["split"]}',flush=True)
    del targets
    protocol['status']='ready';write_json(run/'protocol.json',protocol)
    splits={s:[r for r in rows if r['split']==s] for s in ['train','validation','test']}
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    rng=np.random.default_rng(SEED+7)
    model=base.build_model(torch).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    best=validate(torch,model,run,splits['validation']);best_epoch=0
    history=[{'epoch':0,'train_loss':None,'validation_loss':best}]
    def save_best(epoch,val):
        torch.save({'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'epoch':epoch,'validation_loss':val,'protocol_sha256':sha(run/'protocol.json'),
                    'config':{'architecture':protocol['architecture'],'patch_size':PATCH,
                              'unit_scale':UNIT_SCALE,'seed':SEED,'mixed_precision':True},
                    'training_scope':'one NMC scan, spatially separated regions'},run/'best_model.pt')
    save_best(0,best)
    print(f'NMC TRAINING STARTED | new model | epoch 0 validation {best:.6f}',flush=True)
    for epoch in range(1,args.epochs+1):
        model.train();values=[]
        for n,index in enumerate(rng.permutation(24)):
            a,b=load_pair(run,splits['train'][int(index)])
            for _ in range(6):
                x,y=patches(a,b,rng,augment=True)
                x=torch.from_numpy(x).cuda();y=torch.from_numpy(y).cuda()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda',dtype=torch.float16):pred=model(x)
                value=loss(torch,pred,y)
                if not torch.isfinite(value).item():raise ValueError('Nonfinite training loss.')
                scaler.scale(value).backward();scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                scaler.step(optimizer);scaler.update();values.append(float(value.item()))
            if (n+1)%6==0:print(f'Epoch {epoch}/{args.epochs} | {n+1}/24 training regions',flush=True)
        val=validate(torch,model,run,splits['validation'])
        if val<best:best=val;best_epoch=epoch;save_best(epoch,val)
        history.append({'epoch':epoch,'train_loss':float(np.mean(values)),'validation_loss':val})
        write_json(run/'training_history.json',history)
        print(f'Epoch {epoch}/{args.epochs} | train {np.mean(values):.6f} | validation {val:.6f} | best {best_epoch}',flush=True)
    selected=torch.load(run/'best_model.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(selected['model_state'],strict=True);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    locked_hash=sha(run/'best_model.pt')
    def predict(patch):
        with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16):
            x=torch.from_numpy(np.ascontiguousarray(patch[None,None])).cuda()
            return model(x)[0,0].float().cpu().numpy()
    print('Checkpoint locked. Evaluating 6 spatially separate test regions...',flush=True)
    tests=[];first=None
    for row in splits['test']:
        a,b=load_pair(run,row)
        raw=base.tiled_predict(a,predict)
        measured=base.project(astra,b,pg,vg)
        physics23=base.reconstruct(astra,measured,pg,vg,a,3)
        corrected=base.reconstruct(astra,measured,pg,vg,raw,3)
        arrays={'sirt20':a,'model_raw':raw,'sirt23':physics23,'model_sirt3':corrected}
        metrics={name:base.rmse(v,b) for name,v in arrays.items()}
        tests.append({'case_id':row['case_id'],'voxel_rmse':metrics})
        print(row['case_id']+' | '+' | '.join(f'{k} {v:.6f}' for k,v in metrics.items()),flush=True)
        if first is None:first={'reference':b,**arrays}
    if sha(run/'best_model.pt')!=locked_hash:raise ValueError('Checkpoint changed during testing.')
    means={k:float(np.mean([r['voxel_rmse'][k] for r in tests])) for k in tests[0]['voxel_rmse']}
    def gain(a,b):return 100*(means[a]-means[b])/means[a] if means[a]>0 else None
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    ax.plot([r['epoch'] for r in history[1:]],[r['train_loss'] for r in history[1:]],'o-',label='Train')
    ax.plot([r['epoch'] for r in history],[r['validation_loss'] for r in history],'o-',label='Validation')
    ax.axvline(best_epoch,color='gray',linestyle='--',label=f'Selected epoch {best_epoch}')
    ax.set(xlabel='Epoch',ylabel='Normalized loss',title='NMC training | spatial split within one scan');ax.legend();ax.grid(alpha=.2)
    fig.savefig(run/'training_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(15,3.5),layout='constrained')
    low,high=np.percentile(first['reference'],[1,99]);slices={}
    for ax,(name,a) in zip(axes,first.items()):
        ax.imshow(a[64],cmap='gray',vmin=low,vmax=high);ax.set_title(name,fontsize=10);ax.axis('off')
        for direction,sl in [('axial',np.s_[64,:,:]),('coronal',np.s_[:,64,:]),('sagittal',np.s_[:,:,64])]:slices[name+'_'+direction]=a[sl]
    fig.suptitle('First fixed test region | simulated views | common display scale')
    fig.savefig(run/'test_comparison.png',dpi=150);plt.close(fig)
    np.savez_compressed(run/'review_slices.npz',**slices)
    report={'status':'nmc_within_scan_pilot_completed','best_epoch':best_epoch,'epochs':args.epochs,
            'best_validation_loss':best,'checkpoint_sha256':locked_hash,'test_cases':tests,
            'mean_test_voxel_rmse':means,'raw_model_reduction_vs_sirt20_percent':gain('sirt20','model_raw'),
            'model_sirt3_reduction_vs_sirt23_percent':gain('sirt23','model_sirt3'),
            'protocol_sha256':sha(run/'protocol.json'),'total_seconds':perf_counter()-started,
            'torch_version':str(torch.__version__),'astra_version':astra.__version__,
            'limitations':['Train, validation and test regions come from one scan, not independent specimens.',
               'Regions do not overlap and fixed XY gaps separate the split strips, but scan-specific correlations remain.',
               'The previously evaluated central block was excluded; this does not restore it as a fresh test.',
               'TIFF reconstructions are reference targets, not independent physical ground truth.',
               'Views are noiseless simulations with matching forward/inverse geometry; real acquired data may perform worse.',
               'Intensity units and geometry are assumptions, not NMC physical calibration.',
               'No battery defect labels, battery health prediction or time-resolved 4D reconstruction are evaluated.',
               'All six test regions are now consumed evaluation data; further tuning requires a new independent evaluation.']}
    write_json(run/'report.json',report)
    bundle=run/'nmc_training_review.zip'
    with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ['report.json','protocol.json','training_history.json','training_curves.png','test_comparison.png','review_slices.npz']:
            z.write(run/name,name)
        z.write(Path(__file__),'nmc_train_reconstructor.py')
    with zipfile.ZipFile(bundle) as z:
        if z.testzip() is not None:raise ValueError('Review ZIP verification failed.')
    print('\nNMC TRAINING SUMMARY | WITHIN ONE SCAN | SIMULATED VIEWS')
    print(f'Selected epoch: {best_epoch}/{args.epochs}')
    for k,v in means.items():print(f'{k}: mean test voxel RMSE {v:.6f}')
    print(f'Raw model reduction vs SIRT20: {gain("sirt20","model_raw"):.2f}%')
    print(f'Model + SIRT3 reduction vs SIRT23: {gain("sirt23","model_sirt3"):.2f}%')
    print('Positive = better; negative = worse. No between-specimen generalization established.')
    print('Checkpoint: '+str((run/'best_model.pt').resolve()))
    print('Upload ZIP: '+str(bundle.resolve()))
    if args.open and os.name=='nt':os.startfile(str((run/'test_comparison.png').resolve()))


if __name__=='__main__':
    try:main()
    except (OSError,ValueError,KeyError,RuntimeError,ImportError,zipfile.BadZipFile) as exc:
        print('NMC training stopped: '+str(exc),file=sys.stderr)
        raise SystemExit(1)
