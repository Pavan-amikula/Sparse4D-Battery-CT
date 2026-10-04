"""Prepare a reproducible experimental archive replay, not calibrated scanner data."""
from pathlib import Path
import argparse,configparser,datetime,hashlib,json,re,time
import numpy as np

BASE=Path(r'C:\Users\paam25\MLProjects\Sparse4D_Project\data\raw\waygate750')
HERE=Path(__file__).resolve().parent
ASSUMPTIONS=[
 'Detector dark/flat correction status is unknown. No additional dark/flat subtraction is applied.',
 'Historical provisional preprocessing: mean 3x3 intensities, floor 0.5, then -log(I/FreeRay=9851); negative values retained.',
 'Historical geometry assumption: exported 750-pixel detector pitch treated as 0.2 mm, then 0.6 mm after 3x3 reduction. The 1500-to-750 vendor ROI mapping may imply a different effective pitch; it is unresolved.',
 'Centered circular cone approximation: detector rotation zero; recorded vendor Tilt=-0.00044246 and CorrectionValue=1.095 are not interpreted or applied.',
 'Angles from the acquisition log are used in increasing order; scanner-to-ASTRA orientation has not been independently verified.',
 'Frozen NMC model is transferred to a different battery type without training or validated measured-data performance.',
 'Full 24.000082 mm field reconstructed as 128 cubed; voxel spacing is twice the NMC simulation spacing. This introduces a spatial-scale domain shift.',
]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def convert(a,free):
 if a.shape!=(750,750) or a.dtype!=np.uint16:raise ValueError('Expected one 750x750 uint16 projection.')
 mean=a.astype(np.float32).reshape(250,3,250,3).mean(axis=(1,3))
 return -np.log(np.maximum(mean,np.float32(.5))/np.float32(free)).astype(np.float32)
def parse_log(path):
 rows={}
 for line in path.read_text(encoding='utf-8-sig').splitlines():
  if not re.match(r'^\s*\d+\s+[\d.]+\s+',line):continue
  fields=line.split();n=int(fields[0])
  if n in rows:raise ValueError('Duplicate projection in acquisition log.')
  if len(fields)<12:raise ValueError('Incomplete acquisition row.')
  rows[n]={'angle':float(fields[1]),'use':int(fields[7]),'timestamp':fields[-2]+' '+fields[-1]}
 for n in range(1,1201):
  if n not in rows or rows[n]['use']!=1 or abs(rows[n]['angle']-(n-1)*.3)>1e-6:raise ValueError('The first 1200 log entries differ from the reviewed acquisition.')
 return rows

def main():
 import tifffile
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,default=BASE);args=ap.parse_args()
 base=args.base;acq=base/'acquisition';meta=base/'metadata'
 def find(name):
  for d in [acq,meta]:
   if (d/name).is_file():return d/name
  raise FileNotFoundError(name+' is missing from acquisition and metadata folders.')
 pca=find('Battery 750.pca');pcp=find('Battery 750.pcp')
 if sha(pca)!='d6d04a4948d92d46655b9207a34d378fd5e53c5c16cc8fef4d1586e645d21fbb' or sha(pcp)!='5c5bb5511ebcd6df06a10f92ab82acfe95f0e32f8d512ac0a8d594a7fbfefcab':raise ValueError('Metadata hashes differ from the reviewed Waygate archive.')
 c=configparser.ConfigParser();c.read(pca);rows=parse_log(pcp)
 free=float(c['Image']['FreeRay']);fod=float(c['Geometry']['FOD']);fdd=float(c['Geometry']['FDD'])
 expected=json.loads((HERE/'assets/waygate_history.json').read_text());oldhash=expected['input_info']['source_sha256']
 out=HERE/'prepared_waygate'
 if out.exists():raise FileExistsError('prepared_waygate already exists. Reuse it; this tool never overwrites prepared data.')
 started=time.perf_counter();views=list(range(1,1201,16));arrays=[];sources=[]
 print('Preparing 75 archived measured views. Calibration remains approximate.',flush=True)
 for i,n in enumerate(views):
  p=acq/f'Battery 750{n:05d}.tif';digest=sha(p)
  if digest!=oldhash.get(p.name):raise ValueError('Projection hash differs from earlier experiment: '+p.name)
  with tifffile.TiffFile(p) as tf:
   if len(tf.pages)!=1:raise ValueError('Expected single-page projection.')
   a=tf.asarray()
  arrays.append(convert(a,free));sources.append({'name':p.name,'sha256':digest,'angle_degrees':rows[n]['angle'],'timestamp':rows[n]['timestamp']})
  if (i+1)%15==0:print(f'Read {i+1}/75',flush=True)
 data=np.stack(arrays,axis=1)
 if data.shape!=(250,75,250) or not np.isfinite(data).all():raise ValueError('Invalid prepared projections.')
 width=24.000082165002823
 g={'calibration_status':'metadata_approximation','experimental_archive_replay':True,'type':'circular_cone','array_order':'detector_rows,angles,detector_cols','detector_rows':250,'detector_cols':250,'detector_spacing_x':.6,'detector_spacing_y':.6,'source_origin':fod,'source_detector':fdd,'volume_width':width,'volume_shape':[128,128,128],'angles_degrees':[rows[n]['angle'] for n in views],'centered_volume':True,'detector_tilt_degrees':0,'assumptions':ASSUMPTIONS,'source_metadata':{'pca_sha256':sha(pca),'pcp_sha256':sha(pcp)},'preprocessing_confirmed':False}
 out.mkdir()
 np.save(out/'projections.npy',data,allow_pickle=False)
 def write(name,value):(out/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
 write('geometry.json',g)
 first=datetime.datetime.fromisoformat(rows[1]['timestamp']);last=datetime.datetime.fromisoformat(rows[1200]['timestamp'])
 report={'status':'prepared_experimental_archive_replay','projection_sources':sources,'selected_image_numbers':views,'excluded_endpoint':1201,'projection_shape':list(data.shape),'projection_range':[float(data.min()),float(data.max())],'preparation_seconds':time.perf_counter()-started,'historical_full_scan_first_to_last_seconds':(last-first).total_seconds(),'selected_views_first_to_last_seconds':(datetime.datetime.fromisoformat(rows[views[-1]]['timestamp'])-first).total_seconds(),'calibration_confirmed':False,'live_acquisition':False,'assumptions':ASSUMPTIONS,'projection_sha256':sha(out/'projections.npy'),'geometry_sha256':sha(out/'geometry.json')}
 write('preparation_report.json',report)
 print('PREPARATION COMPLETE',flush=True);print('Folder:',out);print('Historical full scan span:',report['historical_full_scan_first_to_last_seconds'],'seconds');print('No calibrated accuracy or live acquisition claim.')
if __name__=='__main__':main()
