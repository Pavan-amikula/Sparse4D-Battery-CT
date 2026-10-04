"""Actual frozen-model execution. No training or model updates."""
from pathlib import Path
import contextlib, importlib.util, json, re, time, struct, zlib
import numpy as np
import nmc_reference as ref

HERE=Path(__file__).resolve().parent
class Cancelled(Exception): pass

def check_cancel(event):
    if event.is_set(): raise Cancelled('Cancelled between stages or inference tiles.')

def write_json(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)

def natural_key(path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)',path.name)]

def crop_origin(value):
    if not isinstance(value,list) or len(value)!=3 or any(type(v) is not int or v<0 for v in value):
        raise ValueError('Crop origin must be three nonnegative integers in Z,Y,X order.')
    return tuple(value)

def validate_volume(array):
    if array.shape!=(128,128,128): raise ValueError('Supported crop is exactly 128 × 128 × 128 in Z,Y,X order.')
    if array.dtype.kind not in 'uif': raise ValueError('Volume must contain real numeric values.')
    if not np.isfinite(array).all(): raise ValueError('Volume contains NaN or infinity.')
    if float(array.std())<1e-8: raise ValueError('Constant crop is not supported. Choose a valid region explicitly.')
    return array

def scale_volume(array, rule, factor=1):
    if rule=='nmc_uint16':
        if array.dtype!=np.uint16: raise ValueError('NMC TIFF scaling requires uint16 input.')
        scaled=array.astype(np.float32)*(ref.UNIT_SCALE/65535)
    elif rule=='already_scaled': scaled=array.astype(np.float32)
    elif rule=='multiply':
        factor=float(factor)
        if not np.isfinite(factor) or factor<=0: raise ValueError('Scaling multiplier must be finite and positive.')
        scaled=array.astype(np.float32)*factor
    else: raise ValueError('Choose an explicit input scaling rule.')
    validate_volume(scaled)
    return np.ascontiguousarray(scaled)

def load_crop(path, origin, cancelled):
    path=Path(path).expanduser().resolve(); z,y,x=crop_origin(origin)
    files=[]
    if path.is_dir():
        import tifffile
        files=sorted([p for p in path.iterdir() if p.suffix.lower() in ('.tif','.tiff')],key=natural_key)
        if len(files)<z+128: raise ValueError('TIFF folder has too few slices for the requested crop.')
        selected=files[z:z+128]; result=[]
        for p in selected:
            check_cancel(cancelled)
            # A TIFF stack here must use single-page 2D files. Avoid loading oversized files before validation.
            with tifffile.TiffFile(p) as t:
                if len(t.pages)!=1 or len(t.pages[0].shape)!=2: raise ValueError('Folder mode expects one 2D page per TIFF.')
                shape=t.pages[0].shape
                if shape[0]<y+128 or shape[1]<x+128: raise ValueError('Crop exceeds TIFF dimensions.')
                if np.prod(shape)>32_000_000: raise ValueError('TIFF slice exceeds the supported size limit.')
                a=t.pages[0].asarray()
            result.append(a[y:y+128,x:x+128].copy())
        arr=np.stack(result); provenance=[{'name':p.name,'sha256':ref.sha(p)} for p in selected]
    elif path.suffix.lower()=='.npy':
        arr=np.load(path,mmap_mode='r',allow_pickle=False)
        if arr.ndim!=3: raise ValueError('NPY volume must have Z,Y,X dimensions.')
        if any(start+128>size for start,size in zip((z,y,x),arr.shape)): raise ValueError('Crop exceeds NPY volume dimensions.')
        arr=np.array(arr[z:z+128,y:y+128,x:x+128]); provenance=[{'name':path.name,'sha256':ref.sha(path)}]
    elif path.suffix.lower() in ('.tif','.tiff'):
        import tifffile
        with tifffile.TiffFile(path) as t:
            shape=t.series[0].shape
            if len(shape)!=3 or np.prod(shape)>64_000_000: raise ValueError('A single TIFF must be a 3D stack of at most 64 million voxels. Use a folder for larger scans.')
            arr=t.asarray()
        if any(start+128>size for start,size in zip((z,y,x),arr.shape)): raise ValueError('Crop exceeds TIFF stack dimensions.')
        arr=arr[z:z+128,y:y+128,x:x+128].copy(); provenance=[{'name':path.name,'sha256':ref.sha(path)}]
    else: raise ValueError('Provide a TIFF folder, a 3D TIFF stack or an NPY volume.')
    return validate_volume(arr), {'path':str(path),'crop_zyx':list(origin),'sources':provenance,'dtype':str(arr.dtype)}

def geometry_spec(g, allow_archive=False):
    if g.get('calibration_status') != 'user_supplied_measured' and not (allow_archive and g.get('calibration_status') == 'metadata_approximation' and g.get('experimental_archive_replay') is True and isinstance(g.get('assumptions'),list) and len(g['assumptions']) >= 3):
        raise ValueError('Geometry must be supplied from your scanner with calibration_status=user_supplied_measured. The simulation schema template cannot be used unchanged.')
    if g.get('type')!='circular_cone' or g.get('array_order')!='detector_rows,angles,detector_cols':
        raise ValueError('Supported geometry: circular_cone with explicit detector_rows,angles,detector_cols order.')
    for key in ('detector_rows','detector_cols'):
        if type(g.get(key)) is not int or not 16<=g[key]<=1024: raise ValueError('Detector dimensions must be integers from 16 to 1024.')
    for key in ('detector_spacing_x','detector_spacing_y','source_origin','source_detector','volume_width'):
        v=g.get(key)
        if not isinstance(v,(int,float)) or not np.isfinite(v) or v<=0: raise ValueError('Geometry '+key+' must be finite and positive.')
    if g['source_detector']<=g['source_origin']: raise ValueError('Source-detector distance must exceed source-origin distance.')
    if g.get('volume_shape')!=[128,128,128]: raise ValueError('Supported volume_shape is [128,128,128].')
    angles=np.asarray(g.get('angles_degrees',[]),dtype=float)
    if angles.ndim!=1 or not 16<=len(angles)<=1200 or not np.isfinite(angles).all(): raise ValueError('Provide 16–1200 finite angles.')
    unwrapped=np.unwrap(np.deg2rad(angles))
    if not np.allclose(np.diff(unwrapped),2*np.pi/len(angles),rtol=1e-5,atol=1e-6):
        raise ValueError('This application supports uniformly spaced full-rotation angles in increasing order only.')
    if g.get('centered_volume') is not True or g.get('detector_tilt_degrees')!=0:
        raise ValueError('Only centered volumes with zero detector tilt are implemented.')
    return dict(g)

def make_measured_geometry(astra,g):
    w=g['volume_width']
    vg=astra.create_vol_geom(128,128,128,-w/2,w/2,-w/2,w/2,-w/2,w/2)
    pg=astra.create_proj_geom('cone',g['detector_spacing_x'],g['detector_spacing_y'],g['detector_rows'],g['detector_cols'],np.deg2rad(g['angles_degrees']),g['source_origin'],g['source_detector']-g['source_origin'])
    return pg,vg

def load_projections(path,g):
    p=Path(path).expanduser().resolve()
    if p.suffix.lower()!='.npy': raise ValueError('Measured projections must be a numeric NPY array, not raw detector TIFFs.')
    a=np.load(p,mmap_mode='r',allow_pickle=False)
    expected=(g['detector_rows'],len(g['angles_degrees']),g['detector_cols'])
    if a.shape!=expected or a.dtype.kind!='f': raise ValueError('Projection array must be floating-point with shape '+str(expected))
    if np.prod(a.shape)>64_000_000 or not np.isfinite(a).all(): raise ValueError('Invalid projections or more than 64 million samples.')
    return np.ascontiguousarray(a,dtype=np.float32), {'path':str(p),'sha256':ref.sha(p),'shape':list(a.shape),'dtype':str(a.dtype)}

class Runtime:
    def __init__(self): self.model=None; self.torch=None; self.astra=None; self.last_load=0
    def ensure(self, need_astra, progress):
        start=time.perf_counter(); cold=self.model is None
        if cold:
            progress('Loading frozen epoch 12 and warming up the GPU',6)
            import torch
            if not torch.cuda.is_available(): raise ValueError('CUDA PyTorch is required. Use the existing GPU Python environment.')
            training=json.loads((HERE/'assets/report.json').read_text())
            if str(torch.__version__)!=training['torch_version']: raise ValueError('PyTorch version differs from the reviewed environment: expected '+training['torch_version'])
            checkpoint_path=HERE/'assets/best_model.pt'
            if ref.sha(checkpoint_path)!=ref.CHECKPOINT_SHA or ref.sha(HERE/'assets/protocol.json')!=ref.PROTOCOL_SHA: raise ValueError('Checkpoint or protocol integrity failed.')
            ck=torch.load(checkpoint_path,map_location='cpu',weights_only=True);cfg=ck['config']
            if (ck['epoch']!=12 or ck['protocol_sha256']!=ref.PROTOCOL_SHA or cfg['architecture']!='BatteryResidualUNet1ch_12_24_36_zero_head' or cfg['unit_scale']!=.4 or cfg['patch_size']!=80 or cfg['mixed_precision'] is not True):
                raise ValueError('Checkpoint configuration differs from the verified model.')
            self.torch=torch;model=ref.build_model(torch).cuda();model.load_state_dict(ck['model_state'],strict=True);model.eval()
            for p in model.parameters():p.requires_grad_(False)
            with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16,enabled=True):
                model(torch.zeros((1,1,80,80,80),device='cuda'))
            torch.cuda.synchronize();self.model=model
        if need_astra and self.astra is None:
            import astra
            training=json.loads((HERE/'assets/report.json').read_text())
            if not astra.use_cuda() or astra.__version__!=training['astra_version']: raise ValueError('CUDA ASTRA '+training['astra_version']+' is required.')
            self.astra=astra
        elapsed=time.perf_counter()-start
        return cold,elapsed
    def sync(self):
        if self.torch is not None:self.torch.cuda.synchronize()
    def predict(self,array,cancelled,progress):
        torch=self.torch;count=0
        def tile(patch):
            nonlocal count
            check_cancel(cancelled)
            with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16,enabled=True):
                x=torch.from_numpy(np.ascontiguousarray(patch[None,None])).cuda();v=self.model(x)[0,0].float().cpu().numpy()
            count+=1;progress(f'Neural correction · tile {count}/27',35+int(count/27*30));return v
        return ref.tiled_predict(array,tile)

def validate_request(req):
    if not isinstance(req,dict):raise ValueError('Job configuration must be an object.')
    mode=req.get('mode')
    if mode not in ('enhance','simulate','measured','selfcheck','archive'):raise ValueError('Choose a supported mode.')
    if not isinstance(req.get('path'),str) or not req['path'].strip():raise ValueError('Input path is required.')
    if mode not in ('measured','archive'):crop_origin(req.get('origin'))
    if mode=='enhance' and req.get('confirm_reconstruction') is not True:raise ValueError('Confirm that the input is an existing reconstruction in the model input domain.')
    if mode in ('measured','archive') and (req.get('confirm_preprocessing') is not True or req.get('confirm_domain') is not True):raise ValueError('Measured data requires explicit preprocessing and domain confirmations.')
    if mode=='simulate':
        if float(req.get('noise',0)) not in (0,.01,.03,.05):raise ValueError('Noise must be 0, 0.01, 0.03 or 0.05.')
    return req

def execute(req, runtime, output, cancel, progress):
    validate_request(req);started=time.perf_counter();times={};mode=req['mode'];output.mkdir(parents=True,exist_ok=False)
    write_json(output/'request.json',req)
    def stage(name,func):
        check_cancel(cancel);runtime.sync();t=time.perf_counter();v=func();runtime.sync();times[name]=time.perf_counter()-t;return v
    cold,load=runtime.ensure(mode!='enhance',progress);times['runtime_load_and_warmup']=load
    torch=runtime.torch;torch.cuda.reset_peak_memory_stats()
    protocol=json.loads((HERE/'assets/protocol.json').read_text());target=None;measurement=None;g=None
    progress('Loading and validating input data',15)
    if mode in ('measured','archive'):
        g=geometry_spec(json.loads(Path(req['geometry_path']).expanduser().read_text(encoding='utf-8-sig')), allow_archive=mode=='archive')
        if mode=='archive' and g.get('calibration_status')!='metadata_approximation':raise ValueError('Archive replay requires its explicitly approximate geometry file.')
        measurement,provenance=stage('input_loading',lambda:load_projections(req['path'],g));pg,vg=make_measured_geometry(runtime.astra,g)
        # Explicitly supplied line-integral units. No automatic intensity rescaling.
    else:
        origin=[45,256,1728] if mode=='selfcheck' else req['origin']
        a,provenance=stage('input_loading',lambda:load_crop(req['path'],origin,cancel))
        rule='nmc_uint16' if mode=='selfcheck' else req.get('scaling')
        target=stage('preprocessing',lambda:scale_volume(a,rule,req.get('factor',1)))
        del a
        if mode!='enhance':pg,vg,width=ref.make_geometry(runtime.astra,protocol['geometry_source']);g={'simulation_geometry':protocol['geometry_source'],'simulation_width':width,'views':75}
    if mode=='enhance':initial=target;target=None
    else:
        if measurement is None:
            progress('Generating simulated cone-beam projections',22)
            measurement=stage('forward_projection',lambda:ref.project(runtime.astra,target,pg,vg))
            noise=0 if mode=='selfcheck' else float(req.get('noise',0))
            if noise:
                rng=np.random.default_rng(202610041);sigma=noise*np.sqrt(np.mean(measurement.astype(np.float64)**2))
                measurement=(measurement.astype(np.float64)+rng.normal(0,sigma,measurement.shape)).astype(np.float32)
        progress('Physics reconstruction · FDK + SIRT20',28)
        initial=stage('fdk_sirt20',lambda:ref.reconstruct(runtime.astra,measurement,pg,vg))
    quantized=stage('input_quantization',lambda:initial.astype(np.float16).astype(np.float32));validate_volume(quantized)
    if mode=='enhance' and (float(quantized.min())<-.4 or float(quantized.max())>.8) and req.get('confirm_domain') is not True:
        raise ValueError('Input lies outside the broad expected simulation range [-0.4, 0.8]. Check units and explicitly acknowledge a domain shift before proceeding.')
    raw=stage('model_inference',lambda:runtime.predict(quantized,cancel,progress));validate_volume(raw)
    volumes={'input':quantized,'model_raw':raw}
    if mode!='enhance':
        progress('Matched physics baseline · SIRT23',70)
        physics=stage('extra_sirt_baseline',lambda:ref.reconstruct(runtime.astra,measurement,pg,vg,quantized,3))
        progress('Physics correction · model + SIRT3',80)
        hybrid=stage('extra_sirt_hybrid',lambda:ref.reconstruct(runtime.astra,measurement,pg,vg,raw,3))
        volumes={'sirt20':quantized,'model_raw':raw,'sirt23':physics,'model_sirt3':hybrid}
        if target is not None:volumes={'reference':target,**volumes}
    progress('Measuring results and saving output volumes',90)
    metrics={}
    def evaluate():
        for name,v in volumes.items():
            check_cancel(cancel)
            if name=='reference':continue
            m={}
            if target is not None:m['voxel_rmse']=ref.rmse(v,target)
            if measurement is not None:m['fit_projection_rmse']=ref.rmse(ref.project(runtime.astra,v,pg,vg),measurement)
            metrics[name]=m
    stage('metrics',evaluate)
    for name,v in volumes.items():validate_volume(v)
    checks=None
    if mode=='selfcheck':
        expected=json.loads((HERE/'assets/expected_2000bar.json').read_text());expected_sources=expected['source_files']
        if provenance['sources']!=expected_sources:raise ValueError('Self-check source TIFF hashes differ from the verified 2000bar stack.')
        recorded=expected['test_cases'][0]['metrics'];checks={}
        for name in recorded:
            delta=abs(metrics[name]['voxel_rmse']-recorded[name]['voxel_rmse']);checks[name]={'expected':recorded[name]['voxel_rmse'],'actual':metrics[name]['voxel_rmse'],'absolute_difference':delta,'passed':delta<=1e-5}
    t=time.perf_counter()
    for name,v in volumes.items():np.save(output/(name+'.npy'),v,allow_pickle=False)
    times['volume_saving']=time.perf_counter()-t;runtime.sync()
    basis=target if target is not None else quantized
    low,high=[float(x) for x in np.percentile(basis,[1,99])]
    if high<=low:low,high=float(basis.min()),float(basis.max())
    elapsed=time.perf_counter()-started
    report={'mode':mode,'checkpoint_sha256':ref.sha(HERE/'assets/best_model.pt'),'epoch':12,'training_performed':False,'request':req,'input_provenance':provenance,'geometry':g,'metrics':metrics,'timings_seconds':times,'processing_seconds':elapsed,'cold_model_load':cold,'latency_target_seconds':5,'processing_target_met':elapsed<=5,'peak_pytorch_gpu_allocated_mb':torch.cuda.max_memory_allocated()/1024**2,'torch_version':str(torch.__version__),'astra_version':runtime.astra.__version__ if runtime.astra is not None else None,'methods':list(volumes),'display_range':[low,high],'selfcheck':checks,'model_input_range':[float(quantized.min()),float(quantized.max())],'limits':['Processing latency excludes scanner acquisition, browser transfers and ZIP packaging.','The 5-second target is an engineering target, not a scanner real-time claim.','GPU memory metric includes PyTorch allocations only, not ASTRA or total device usage.','Model validation uses simulated NMC views. Measured-data generalization is unproven.','Projection RMSE uses reconstruction angles; it is not held-out validation.','Enhancement mode does not perform physics correction without projection data.','TIFF references are reconstructions, not independent physical ground truth.']}
    if mode=='archive':
        report['experimental_archive_replay']=True
        report['calibration_confirmed']=False
        report['limits'].extend(g['assumptions'])
        report['limits'].append('Archived acquisition replay; no live scanner connection or acquisition-to-display latency measurement.')
        report['limits'].append('128-cubed full-field volume differs from the earlier 256-cubed Waygate experiment; its old numerical results do not apply.')
    if report['checkpoint_sha256']!=ref.CHECKPOINT_SHA:raise ValueError('Checkpoint changed during execution.')
    write_json(output/'report.json',report);check_cancel(cancel)
    progress('Packaging actual outputs',96)
    import zipfile
    with zipfile.ZipFile(output/'results.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in output.iterdir():
            if p.name!='results.zip' and p.is_file():z.write(p,p.name)
    return report

def slice_png(volume,axis,index,low,high):
    if axis not in ('axial','coronal','sagittal') or not 0<=index<128:raise ValueError('Invalid slice selection.')
    a=volume[index,:,:] if axis=='axial' else volume[:,index,:] if axis=='coronal' else volume[:,:,index]
    gray=(np.clip((a-low)/max(high-low,1e-8),0,1)*255).astype(np.uint8)
    raw=b''.join(b'\0'+row.tobytes() for row in gray)
    def chunk(tag,data):return struct.pack('!I',len(data))+tag+data+struct.pack('!I',zlib.crc32(tag+data)&0xffffffff)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',128,128,8,0,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')
