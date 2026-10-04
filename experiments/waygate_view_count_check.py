"""Archived Waygate view-count check at 256 cubed: no model or training.
Run with the GPU Python used by NMC_Waygate_Replay. This is a development
check, not new held-out validation, calibrated accuracy, or live acquisition.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, time, uuid, zipfile
import numpy as np

APP = Path(r'C:\Users\paam25\Downloads\NMC_Waygate_Replay')
BASE = Path(r'C:\Users\paam25\MLProjects\Sparse4D_Project\data\raw\waygate750')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def convert(a):
    if a.shape != (750, 750) or a.dtype != np.uint16:
        raise ValueError('Expected a 750 x 750 uint16 projection.')
    mean = a.astype(np.float32).reshape(250, 3, 250, 3).mean(axis=(1, 3))
    return -np.log(np.maximum(mean, np.float32(.5)) / np.float32(9851)).astype(np.float32)

def parse_log(path):
    rows = {}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        f = line.split()
        if not f or not f[0].isdigit():
            continue
        if len(f) < 12:
            raise ValueError('Incomplete acquisition log row.')
        n = int(f[0])
        if n in rows:
            raise ValueError('Duplicate acquisition log row.')
        rows[n] = {'angle': float(f[1]), 'use': int(f[7]), 'timestamp': ' '.join(f[-2:])}
    for n in range(1, 1201):
        if n not in rows or rows[n]['use'] != 1 or abs(rows[n]['angle'] - (n-1)*.3) > 1e-6:
            raise ValueError('Unexpected acquisition trajectory.')
    return rows

def fdk(astra, data, angles, geometry, size):
    w = geometry['volume_width']
    vg = astra.create_vol_geom(size, size, size, -w/2, w/2, -w/2, w/2, -w/2, w/2)
    pg = astra.create_proj_geom('cone', .6, .6, 250, 250, np.deg2rad(angles),
                                geometry['source_origin'],
                                geometry['source_detector'] - geometry['source_origin'])
    pid = rid = aid = None
    try:
        pid = astra.data3d.create('-proj3d', pg, np.ascontiguousarray(data))
        rid = astra.data3d.create('-vol', vg, 0)
        cfg = astra.astra_dict('FDK_CUDA')
        cfg.update(ProjectionDataId=pid, ReconstructionDataId=rid)
        aid = astra.algorithm.create(cfg)
        astra.algorithm.run(aid)
        volume = astra.data3d.get(rid).astype(np.float32)
        if volume.shape != (size, size, size) or not np.isfinite(volume).all():
            raise ValueError('FDK produced an invalid volume.')
        return volume
    finally:
        if aid is not None:
            astra.algorithm.delete(aid)
        for ident in (pid, rid):
            if ident is not None:
                astra.data3d.delete(ident)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--app', type=Path, default=APP)
    ap.add_argument('--base', type=Path, default=BASE)
    args = ap.parse_args()
    import astra
    import tifffile
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not astra.use_cuda():
        raise RuntimeError('CUDA-enabled ASTRA is required. Use the project GPU Python.')
    prepared = args.app / 'prepared_waygate'
    prep = json.loads((prepared / 'preparation_report.json').read_text())
    g = json.loads((prepared / 'geometry.json').read_text())
    for name, key in [('projections.npy', 'projection_sha256'), ('geometry.json', 'geometry_sha256')]:
        if sha(prepared / name) != prep[key]:
            raise ValueError('Prepared data changed: ' + name)
    if prep['projection_sha256'] != '54bf2d838b45004eddea855c59bcd9ddc23891692546b42e90998277796a3670':
        raise ValueError('This is not the reviewed replay input.')
    expected = {'detector_spacing_x': .6, 'detector_spacing_y': .6,
                'source_origin': 129.03973962, 'source_detector': 806.495599,
                'volume_width': 24.000082165002823}
    for key, value in expected.items():
        if abs(g[key] - value) > 1e-8:
            raise ValueError('Geometry changed: ' + key)
    def metadata(name):
        for folder in [args.base / 'acquisition', args.base / 'metadata']:
            p = folder / name
            if p.is_file():
                return p
        raise FileNotFoundError(name)
    for suffix in ['pca', 'pcp']:
        if sha(metadata('Battery 750.' + suffix)) != g['source_metadata'][suffix + '_sha256']:
            raise ValueError('Scanner metadata changed.')
    rows = parse_log(metadata('Battery 750.pcp'))
    numbers = list(range(1, 1201, 16))
    if prep['selected_image_numbers'] != numbers or not np.allclose(g['angles_degrees'], [rows[n]['angle'] for n in numbers], atol=1e-8, rtol=0):
        raise ValueError('Sparse view selection changed.')
    sparse = np.load(prepared / 'projections.npy', allow_pickle=False)
    if sparse.shape != (250, 75, 250) or sparse.dtype != np.float32 or not np.isfinite(sparse).all():
        raise ValueError('Invalid sparse input.')
    print('WAYGATE VIEW COUNT CHECK | 256 cubed | FDK only | no training', flush=True)
    print('Fixed provisional geometry. Comparing 75, 150, 300, 600 and 1200 views.', flush=True)
    start = time.perf_counter()
    full = np.empty((250, 1200, 250), dtype=np.float32)
    sources = []
    sparse_hashes = {x['name']: x['sha256'] for x in prep['projection_sources']}
    for n in range(1, 1201):
        path = args.base / 'acquisition' / f'Battery 750{n:05d}.tif'
        digest = sha(path)
        if path.name in sparse_hashes and digest != sparse_hashes[path.name]:
            raise ValueError('Sparse projection changed: ' + path.name)
        with tifffile.TiffFile(path) as tf:
            if len(tf.pages) != 1:
                raise ValueError('Expected single-page projection: ' + path.name)
            full[:, n-1, :] = convert(tf.asarray())
        sources.append({'name': path.name, 'sha256': digest, **rows[n]})
        if n % 100 == 0:
            print(f'Read {n}/1200', flush=True)
    if not np.isfinite(full).all() or not np.array_equal(full[:, ::16, :], sparse):
        raise ValueError('All-view preparation does not reproduce the reviewed 75 views exactly.')
    read_seconds = time.perf_counter() - start
    out = args.app / 'diagnostics' / ('view_counts_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:6])
    out.mkdir(parents=True, exist_ok=False)
    slices, cases = {}, []
    full_angles = [rows[n]['angle'] for n in range(1, 1201)]
    size = 256
    counts = [75, 150, 300, 600, 1200]
    for count in counts:
        step = 1200 // count
        indices = np.arange(0, 1200, step)
        data = sparse if count == 75 else np.ascontiguousarray(full[:, indices, :])
        angles = [full_angles[int(i)] for i in indices]
        key = f'fdk_{count}views_{size}'
        print(f'Reconstructing {key}...', flush=True)
        started = time.perf_counter()
        volume = fdk(astra, data, angles, g, size)
        seconds = time.perf_counter() - started
        middle = size // 2
        slices[key + '_axial'] = volume[middle, :, :].copy()
        slices[key + '_coronal'] = volume[:, middle, :].copy()
        slices[key + '_sagittal'] = volume[:, :, middle].copy()
        cases.append({'method': 'FDK_CUDA', 'views': count, 'size': size,
                      'voxel_spacing_mm': g['volume_width'] / size,
                      'selected_image_numbers': [int(i) + 1 for i in indices],
                      'angle_step_degrees': step * .3,
                      'reconstruction_seconds': seconds,
                      'range': [float(volume.min()), float(volume.max())]})
        del volume
        print(f'Completed in {seconds:.2f} seconds', flush=True)
    # Difference from the full-view reconstruction on the same voxel grid.
    # This is a development comparison, not an independent accuracy metric.
    reference_axial = slices['fdk_1200views_256_axial']
    for case in cases:
        a = slices[f"fdk_{case['views']}views_256_axial"]
        case['middle_axial_rmse_vs_1200view_fdk'] = float(np.sqrt(np.mean((a.astype(np.float64) - reference_axial.astype(np.float64)) ** 2)))
    # Reference is optional, processed, and not scored or used for tuning.
    refs = sorted((args.base / 'reference').rglob('*375*.tif')) if (args.base / 'reference').is_dir() else []
    reference = None
    if len(refs) == 1:
        reference = tifffile.imread(refs[0])
        if reference.shape != (750, 750) or not np.isfinite(reference).all():
            raise ValueError('Unexpected vendor reference slice.')
        slices['vendor_reference_375'] = reference
    np.savez_compressed(out / 'review_slices.npz', **slices)
    fig, axes = plt.subplots(2, 3, figsize=(13, 9), facecolor='#f6f8fb')
    low, high = [float(x) for x in np.percentile(reference_axial, [1, 99])]
    for ax, count in zip(axes.flat, counts):
        a = slices[f'fdk_{count}views_256_axial']
        ax.imshow(a, cmap='gray', vmin=low, vmax=high, interpolation='nearest')
        ax.set_title(f'FDK: {count} measured views | 256 x 256')
    ax = axes.flat[5]
    if reference is not None:
        ax.imshow(reference, cmap='gray', vmin=np.percentile(reference, 1), vmax=np.percentile(reference, 99))
        ax.set_title('Vendor reference | 750 x 750')
    else:
        ax.text(.5, .5, 'Vendor slice not found\nDiagnostic still completed', ha='center', va='center', transform=ax.transAxes)
    for ax in axes.flat:
        ax.axis('off')
    fig.suptitle('View-count check at 256 cubed: fixed provisional geometry', fontsize=15)
    fig.text(.5, .02, 'All five FDK panels use the same brightness scale; vendor panel scaled separately.\nArchived data, approximate middle slices. No calibrated accuracy or real-time claim.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .94))
    fig.savefig(out / 'view_count_comparison.png', dpi=150)
    plt.close(fig)
    first = datetime.datetime.fromisoformat(rows[1]['timestamp'])
    last = datetime.datetime.fromisoformat(rows[1200]['timestamp'])
    report = {'purpose': 'Development diagnostic of view count at fixed 256-cubed voxel grid and provisional geometry',
              'calibration_confirmed': False, 'training_performed': False, 'model_used': False,
              'live_acquisition': False, 'held_out_validation': False,
              'geometry': g, 'diagnostic_volume_sizes': [256], 'diagnostic_view_counts': counts, 'fdk_display_range': [low, high],
              'detector_preprocessing': 'Same mean 3x3, floor 0.5, -log(I/9851) as reviewed replay',
              'sparse_input_sha256': prep['projection_sha256'], 'sparse_input_reproduced_exactly': True,
              'endpoint_1201_excluded': True, 'source_projections': sources,
              'input_loading_and_preprocessing_seconds': read_seconds,
              'historical_acquisition_span_seconds': (last-first).total_seconds(),
              'cases': cases, 'astra_version': astra.__version__,
              'vendor_reference': {'path': str(refs[0]), 'sha256': sha(refs[0])} if reference is not None else None,
              'limits': ['Geometry and correction status remain provisional.',
                         'All-view reconstruction uses previously reserved views and is now development work, not a new blind test.',
                         'No true voxel accuracy, defect detection, or real-time scanner performance is established.',
                         'Every 256-cubed grid covers the same 24.000082 mm width.',
                         'The recorded geometry volume_shape of 128 is overridden only by the explicit 256-cubed diagnostic grid.',
                         'FDK only, no SIRT iterations or learned model. No correction or geometry parameters are selected from this test.',
                         'Middle axial RMSE compares with the same-scan 1200-view FDK, not independent ground truth.',
                         'View counts sample the same archived 20-minute acquisition; they are not measured faster scans.',
                         'Reported reconstruction time excludes acquisition, preprocessing, plotting and packaging.']}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    archive = out / 'waygate_view_count_review.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ['report.json', 'review_slices.npz', 'view_count_comparison.png']:
            z.write(out / name, name)
    print('DIAGNOSTIC COMPLETE', flush=True)
    print('Upload ZIP:', archive, flush=True)
    print('This checks archived reconstruction behavior, not live scanner accuracy.', flush=True)

if __name__ == '__main__':
    main()
