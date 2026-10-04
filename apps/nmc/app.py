"""Local HTTP interface. One GPU job at a time; model reused across jobs."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse, importlib.util, json, mimetypes, secrets, threading, time, traceback, urllib.parse, webbrowser
import numpy as np
import engine

HERE=Path(__file__).resolve().parent
class Jobs:
    def __init__(self,root):
        self.root=root;self.lock=threading.Lock();self.jobs={};self.current=None;self.runtime=engine.Runtime()
    def submit(self,req):
        engine.validate_request(req)
        with self.lock:
            if self.current and self.jobs[self.current]['status'] in ('running','cancelling'):raise ValueError('A GPU job is already running. Finish or cancel it first.')
            jobid=time.strftime('%Y%m%dT%H%M%S')+'_'+secrets.token_hex(3)
            event=threading.Event();job={'id':jobid,'status':'running','stage':'Starting','percent':0,'cancel':event,'started':time.time(),'report':None,'error':None,'folder':self.root/jobid}
            self.jobs[jobid]=job;self.current=jobid
            threading.Thread(target=self.run,args=(jobid,dict(req)),daemon=True).start()
        return jobid
    def run(self,jobid,req):
        job=self.jobs[jobid]
        def progress(stage,percent):
            with self.lock:job.update(stage=stage,percent=percent)
        try:
            report=engine.execute(req,self.runtime,job['folder'],job['cancel'],progress)
            with self.lock:job.update(status='complete',stage='Complete · actual outputs saved',percent=100,report=report)
        except engine.Cancelled as exc:
            with self.lock:job.update(status='cancelled',stage=str(exc),error=str(exc))
        except Exception as exc:
            traceback.print_exc()
            with self.lock:job.update(status='failed',stage='Stopped',error=str(exc))
            if job['folder'].is_dir():engine.write_json(job['folder']/'failure.json',{'error':str(exc),'request':req,'status':'failed'})
    def status(self):
        with self.lock:
            rows=[{k:v for k,v in j.items() if k not in ('cancel','folder')} for j in self.jobs.values()]
            return {'current':self.current,'jobs':rows[-20:],'model_loaded':self.runtime.model is not None,'dependencies':{k:importlib.util.find_spec(k) is not None for k in ('numpy','torch','astra','tifffile')}}
    def cancel(self,jobid):
        with self.lock:
            j=self.jobs.get(jobid)
            if not j:raise ValueError('Unknown job.')
            if j['status']=='running':j['cancel'].set();j['status']='cancelling';j['stage']='Cancellation requested; waiting for a safe stage boundary.'
    def completed(self,jobid):
        with self.lock:
            j=self.jobs.get(jobid)
            if not j or j['status']!='complete':raise ValueError('Completed job not found.')
            return j

class Handler(BaseHTTPRequestHandler):
    def log_message(self,fmt,*args):
        if '/api/status' not in (args[0] if args else ''):super().log_message(fmt,*args)
    def valid_host(self):
        return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')
    def reply(self,status,data,ctype='application/json',download=None):
        if isinstance(data,dict):data=json.dumps(data,allow_nan=False).encode()
        if isinstance(data,str):data=data.encode()
        self.send_response(status);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        if download:self.send_header('Content-Disposition','attachment; filename="'+download+'"')
        self.end_headers();self.wfile.write(data)
    def do_GET(self):
        if not self.valid_host():self.reply(403,{'error':'Localhost access only.'});return
        parsed=urllib.parse.urlparse(self.path);q=urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path=='/':
                html=(HERE/'web/index.html').read_text(encoding='utf-8').replace('__TOKEN__',self.server.token)
                self.reply(200,html,'text/html; charset=utf-8')
            elif parsed.path=='/evidence':self.reply(200,(HERE/'web/evidence.html').read_bytes(),'text/html; charset=utf-8')
            elif parsed.path=='/api/status':self.reply(200,self.server.jobs.status())
            elif parsed.path=='/api/slice':
                j=self.server.jobs.completed(q['job'][0]);method=q['method'][0]
                if method not in j['report']['methods']:raise ValueError('Unknown output method.')
                a=np.load(j['folder']/(method+'.npy'),mmap_mode='r',allow_pickle=False)
                lo,hi=j['report']['display_range'];png=engine.slice_png(a,q.get('axis',['axial'])[0],int(q.get('index',['64'])[0]),lo,hi)
                self.reply(200,png,'image/png')
            elif parsed.path=='/api/download':
                j=self.server.jobs.completed(q['job'][0]);name=q.get('file',['results.zip'])[0]
                allowed=['results.zip','report.json']+[m+'.npy' for m in j['report']['methods']]
                if name not in allowed:raise ValueError('Unknown output file.')
                self.reply(200,(j['folder']/name).read_bytes(),mimetypes.guess_type(name)[0] or 'application/octet-stream',name)
            elif parsed.path=='/geometry-template':self.reply(200,(HERE/'measured_geometry_TEMPLATE.json').read_bytes(),'application/json','measured_geometry_TEMPLATE.json')
            else:self.reply(404,{'error':'Not found.'})
        except (ValueError,KeyError,OSError) as exc:self.reply(400,{'error':str(exc)})
    def do_POST(self):
        if not self.valid_host() or self.headers.get('X-App-Token')!=self.server.token:self.reply(403,{'error':'Invalid local application token.'});return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if self.path=='/api/upload':
                if length<=0 or length>64*1024**2:raise ValueError('NPY uploads must be between 1 byte and 64 MiB. Use a local path for larger files.')
                folder=HERE/'uploads';folder.mkdir(exist_ok=True);p=folder/(secrets.token_hex(8)+'.npy');p.write_bytes(self.rfile.read(length))
                try:
                    a=np.load(p,mmap_mode='r',allow_pickle=False)
                    if a.ndim!=3 or a.dtype.kind not in 'uif':raise ValueError('Upload must be a real numeric 3D NPY volume or projection array.')
                except Exception:p.unlink(missing_ok=True);raise ValueError('Invalid numeric 3D NPY file.')
                self.reply(200,{'path':str(p.resolve()),'shape':list(a.shape),'dtype':str(a.dtype)});return
            if not 0<length<=16384:raise ValueError('Invalid request size.')
            body=json.loads(self.rfile.read(length))
            if self.path=='/api/run':self.reply(202,{'id':self.server.jobs.submit(body)})
            elif self.path=='/api/cancel':self.server.jobs.cancel(body['id']);self.reply(200,{'status':'requested'})
            else:self.reply(404,{'error':'Not found.'})
        except (ValueError,KeyError,OSError,json.JSONDecodeError) as exc:self.reply(400,{'error':str(exc)})

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765);p.add_argument('--no-browser',action='store_true');args=p.parse_args()
    if not 1024<=args.port<=65535:raise ValueError('Port must be 1024–65535.')
    output=HERE/'results';output.mkdir(exist_ok=True)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.token=secrets.token_urlsafe(32);server.jobs=Jobs(output)
    print(f'NMC Live App: http://127.0.0.1:{args.port}',flush=True)
    print('Frozen model. Results saved in '+str(output),flush=True)
    if not args.no_browser:webbrowser.open(f'http://127.0.0.1:{args.port}')
    try:server.serve_forever()
    except KeyboardInterrupt:print('Stopping interface. In-flight jobs may be interrupted; completed results stay saved.')
    finally:server.server_close()
if __name__=='__main__':main()
