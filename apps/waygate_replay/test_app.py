import ast,io,json,struct,tempfile,threading,unittest,zipfile,zlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
import engine,nmc_reference as ref

class CoreTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.event=threading.Event()
    def tearDown(self):self.temp.cleanup()
    def volume(self):return np.broadcast_to(np.linspace(0,.4,128,dtype=np.float32),(128,128,128)).copy()
    def test_checkpoint_and_protocol_identity(self):
        self.assertEqual(ref.sha(engine.HERE/'assets/best_model.pt'),ref.CHECKPOINT_SHA)
        self.assertEqual(ref.sha(engine.HERE/'assets/protocol.json'),ref.PROTOCOL_SHA)
    def test_npy_crop_and_provenance(self):
        a=np.zeros((130,131,132),np.float32);a[1:129,2:130,3:131]=self.volume();p=self.root/'v.npy';np.save(p,a)
        crop,provenance=engine.load_crop(p,[1,2,3],self.event);np.testing.assert_array_equal(crop,self.volume());self.assertEqual(provenance['crop_zyx'],[1,2,3]);self.assertEqual(provenance['sources'][0]['sha256'],ref.sha(p))
        with self.assertRaises(ValueError):engine.load_crop(p,[4,2,3],self.event)
    def test_validation_and_scaling(self):
        v=self.volume();self.assertAlmostEqual(float(engine.scale_volume(v,'multiply',2).max()),.8,places=6)
        for bad in [v[:127],np.full(v.shape,np.nan),np.zeros(v.shape),v.astype(complex)]:
            with self.assertRaises(ValueError):engine.validate_volume(bad)
        for factor in [0,-1,float('nan')]:
            with self.assertRaises(ValueError):engine.scale_volume(v,'multiply',factor)
        with self.assertRaises(ValueError):engine.scale_volume(v,'nmc_uint16')
        for origin in [[-1,0,0],[0.5,0,0],[0,0],[True,0,0]]:
            with self.assertRaises(ValueError):engine.crop_origin(origin)
    def test_tiling_preserves_volume_and_checks_predictions(self):
        v=self.volume();calls=[]
        def identity(x):calls.append(x.shape);return x
        actual=ref.tiled_predict(v,identity);np.testing.assert_allclose(actual,v,atol=1e-6);self.assertEqual(len(calls),27)
        with self.assertRaises(ValueError):ref.tiled_predict(v,lambda x:np.full(x.shape,np.nan))
    def spec(self):return {'calibration_status':'user_supplied_measured','type':'circular_cone','array_order':'detector_rows,angles,detector_cols','detector_rows':16,'detector_cols':16,'detector_spacing_x':.6,'detector_spacing_y':.6,'source_origin':129,'source_detector':806,'volume_width':12,'volume_shape':[128]*3,'angles_degrees':np.arange(0,360,4.8).tolist(),'centered_volume':True,'detector_tilt_degrees':0}
    def test_measured_inputs_require_explicit_supported_geometry(self):
        g=engine.geometry_spec(self.spec());p=self.root/'p.npy';np.save(p,np.ones((16,75,16),np.float32));a,meta=engine.load_projections(p,g);self.assertEqual(a.shape,(16,75,16))
        for key,value in [('source_detector',100),('detector_tilt_degrees',1),('array_order','angles,rows,cols'),('angles_degrees',[0,20,35])]:
            bad=self.spec();bad[key]=value
            with self.assertRaises(ValueError):engine.geometry_spec(bad)
        np.save(p,np.ones((75,16,16),np.float32))
        with self.assertRaises(ValueError):engine.load_projections(p,g)
        with self.assertRaises(ValueError):engine.validate_request({'mode':'measured','path':'x.npy'})
    def test_slice_png_orientation_and_format(self):
        v=self.volume();png=engine.slice_png(v,'sagittal',127,0,.4);self.assertEqual(png[:8],b'\x89PNG\r\n\x1a\n');pos=8;raw=b''
        while pos<len(png):
            n=struct.unpack('!I',png[pos:pos+4])[0];tag=png[pos+4:pos+8];data=png[pos+8:pos+8+n];crc=struct.unpack('!I',png[pos+8+n:pos+12+n])[0];self.assertEqual(crc,zlib.crc32(tag+data)&0xffffffff)
            if tag==b'IDAT':raw+=data
            pos+=12+n
        rows=zlib.decompress(raw);self.assertEqual(len(rows),128*129);self.assertTrue(all(rows[i*129]==0 for i in range(128)));self.assertEqual(rows[1],255)
        with self.assertRaises(ValueError):engine.slice_png(v,'axial',128,0,.4)
    def test_full_job_saves_actual_arrays_metrics_and_report(self):
        class FakeCuda:
            def reset_peak_memory_stats(self):pass
            def max_memory_allocated(self):return 1024
        class FakeTorch:cuda=FakeCuda();__version__='test-double'
        class FakeAstra:__version__='test-double'
        class FakeRuntime:
            torch=FakeTorch();astra=FakeAstra()
            def ensure(self,need_astra,progress):return False,0
            def sync(self):pass
            def predict(self,array,cancel,progress):return array*.98
        p=self.root/'volume.npy';v=self.volume();np.save(p,v);calls=[]
        def recon(astra,measurement,pg,vg,initial=None,iterations=20):
            calls.append((initial is None,iterations));return v*.9 if initial is None else initial*.99
        req={'mode':'simulate','path':str(p),'origin':[0,0,0],'scaling':'already_scaled','noise':0}
        out=self.root/'job'
        with patch.object(ref,'make_geometry',return_value=(None,None,12)),patch.object(ref,'project',side_effect=lambda a,v,p,g:v),patch.object(ref,'reconstruct',side_effect=recon):
            r=engine.execute(req,FakeRuntime(),out,self.event,lambda *a:None)
        self.assertEqual(calls,[(True,20),(False,3),(False,3)])
        actual=np.load(out/'model_raw.npy');np.testing.assert_allclose(actual,v.astype(np.float16).astype(np.float32)*.9*.98,atol=.0002)
        self.assertAlmostEqual(r['metrics']['model_raw']['voxel_rmse'],ref.rmse(actual,v))
        self.assertFalse(r['training_performed']);self.assertIsNone(r['selfcheck']);self.assertIn('model_inference',r['timings_seconds'])
        with zipfile.ZipFile(out/'results.zip') as z:self.assertIsNone(z.testzip());self.assertIn('model_raw.npy',z.namelist())
    def test_archive_reports_assumptions_and_has_no_voxel_accuracy(self):
        from types import SimpleNamespace
        import prepare_waygate
        g=self.spec();g.update(calibration_status='metadata_approximation',experimental_archive_replay=True,assumptions=prepare_waygate.ASSUMPTIONS)
        with self.assertRaises(ValueError):engine.geometry_spec(g)
        engine.geometry_spec(g,allow_archive=True)
        projections=np.ones((16,75,16),np.float32)
        source=self.root/'projections.npy';np.save(source,projections)
        geometry=self.root/'geometry.json';geometry.write_text(json.dumps(g))
        v=self.volume()
        cuda=SimpleNamespace(reset_peak_memory_stats=lambda:None,max_memory_allocated=lambda:1024)
        runtime=SimpleNamespace(torch=SimpleNamespace(cuda=cuda,__version__='test-double'),astra=SimpleNamespace(__version__='test-double'),ensure=lambda *args:(False,0),sync=lambda:None,predict=lambda a,*args:a*.98)
        req=dict(mode='archive',path=str(source),geometry_path=str(geometry),confirm_domain=True,confirm_preprocessing=True)
        out=self.root/'archive_job'
        with patch.object(engine,'make_measured_geometry',return_value=(None,None)),patch.object(ref,'reconstruct',side_effect=lambda a,m,p,g,initial=None,iterations=20:v if initial is None else initial*.99),patch.object(ref,'project',return_value=projections):
            report=engine.execute(req,runtime,out,self.event,lambda *args:None)
        self.assertTrue(report['experimental_archive_replay']);self.assertFalse(report['calibration_confirmed'])
        self.assertNotIn('reference',report['methods'])
        self.assertTrue(all('voxel_rmse' not in m for m in report['metrics'].values()))
        self.assertTrue(all(a in report['limits'] for a in prepare_waygate.ASSUMPTIONS))
        with zipfile.ZipFile(out/'results.zip') as z:self.assertIsNone(z.testzip())

    def test_cancelled_stage_does_not_publish_results(self):
        self.event.set()
        with self.assertRaises(engine.Cancelled):engine.check_cancel(self.event)

if __name__=='__main__':unittest.main(verbosity=2)
