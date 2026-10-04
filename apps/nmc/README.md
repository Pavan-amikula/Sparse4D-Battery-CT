# NMC Live Reconstruction App

This is a local application that runs the frozen epoch-12 model on input data. It is separate from the saved-results dashboard. No training is performed.

## Start on your Windows PC

1. Extract `NMC_Live_App.zip` into Downloads. The extracted folder is `NMC_Live_App`.
2. Run in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\paam25\Downloads\NMC_Live_App\Launch_NMC.ps1"
```

The launcher checks NumPy, PyTorch, ASTRA and tifffile in your existing GPU environment. It does not install or replace packages. Keep the PowerShell window open. The interface opens at http://127.0.0.1:8765.

If the port is occupied, pass `-Port 8766`. If your Python path changes, pass `-Python "C:\path\to\python.exe"`.

Default Python:
`C:\Users\paam25\MLProjects\Sparse4D_Project\.venv\Scripts\python.exe`

## First action: verify installation

Leave **Verify installation** selected. The default input is:
`C:\Users\paam25\MLProjects\Sparse4D_Project\data\raw\battery_nmc2000bar\grayscale`

Click **Run verification**. This reproduces the first fixed reviewed 2000bar region at Z=45, Y=256, X=1728, with a 128-cubed crop. This is a reproducibility check on consumed evaluation data, not a new generalization experiment.

The full 128 source file hash list must match the reviewed stack. All four voxel RMSE values must match the recorded report within 0.00001. Small GPU numeric variation is expected; this tolerance is a reproducibility check, not a new model selection criterion. A mismatch stays visible in the report. Do not tune the model or evaluation settings to chase the scores.

Approximate expected values:
- SIRT20: 0.00531956
- Raw model: 0.00323679
- SIRT23: 0.00513063
- Hybrid: 0.00306944

Run the same verification a second time to obtain warmed-runtime timings. The first run includes model loading and warm-up. Subsequent jobs reuse the model. Compare `cold_model_load` and `timings_seconds` in each report.

## Modes

### Simulation

Load a 3D NPY, a 3D TIFF, or a folder of single-page 2D TIFF files. Select a zero-based crop origin in Z,Y,X order. The program generates the existing 75 simulated cone-beam views, runs FDK+SIRT20, neural correction and the matched extra three SIRT steps.

Choose an explicit scaling rule. `NMC uint16 / 65535 × 0.4` applies only to uint16 input. Unknown materials or units are not automatically calibrated. The known geometry is used only in simulation mode and is not NMC scanner calibration.

Noise options use fixed Gaussian noise at 0, 1, 3 and 5 percent of clean projection RMS with seed 202610041. These are sensitivity simulations, not calibrated photon or scanner noise. User-selected crops are demonstrations, not pre-registered new evaluation results.

### Reconstruction enhancement

Supply an existing reconstruction comparable to the FDK+SIRT20 model input, in Z,Y,X orientation. Do not treat a raw high-quality TIFF reference as a validated input distribution. Choose the correct scaling and confirm that you understand the input domain.

This mode returns the input and actual raw model output. It does not run SIRT without projection data. It has no ground-truth reference and reports no voxel accuracy claim. Inputs outside the broad range [-0.4, 0.8] require explicit domain acknowledgement; that acknowledgement does not establish scientific validity.

### Measured projections: experimental integration

This mode has implementation support but has not been tested with measured NMC projections. It is not a demonstrated real-time scanner connection.

- Provide a **float32/float64 NPY** in `(detector_rows, angles, detector_cols)` order.
- Provide preprocessed attenuation line integrals, not raw counts. Flat/dark correction, logarithmic conversion, detector calibration and scanner-specific preprocessing are outside this app.
- Supply explicit scanner geometry in JSON using `measured_geometry_TEMPLATE.json` as a schema guide.
- The template's numerical values are simulation examples. It is deliberately rejected unchanged. Obtain scanner geometry and set `calibration_status` to `user_supplied_measured` only when those fields actually describe your data.
- All lengths must use consistent units. Projection and reconstruction units must also be consistent with each other. No measured-data intensity rescaling is silently applied.
- Supported: centered 128-cubed volume, circular cone trajectory, flat detector, zero tilt, increasing uniformly spaced angles covering a full rotation, 16–1200 views, detector dimensions 16–1024.
- Unsupported: arbitrary cone vectors, helical trajectories, detector tilt, off-center reconstruction, limited angle, incomplete rotation, nonuniform angles, cone calibration estimation or arbitrary acquisition formats.
- The model's learned prior remains simulation-domain-specific. It may worsen measured reconstructions even if the geometry is accepted.

Fit projection RMSE uses the reconstruction angles and is not a held-out projection metric. Voxel RMSE is intentionally omitted without a reference.

## Interface features

- Background processing and stage progress.
- One active GPU job at a time to avoid memory competition.
- Cancellation between stages and tiles. An ASTRA call already in progress must finish before cancellation is handled.
- Actual 3D slice scrolling in axial, coronal and sagittal planes.
- Draggable before/after comparison and volume selection.
- Download output NPY volumes, a JSON report or the complete results ZIP.
- Optional local-browser NPY upload up to 64 MiB; use local paths for larger arrays and TIFF folders.
- Linked access to the explanatory saved-results dashboard.

## Timing and output

Each job creates a unique directory under `NMC_Live_App/results`. Input source hashes, configuration, checkpoint identity, metrics, actual volume arrays and stage timings are saved there. Uploaded NPY files are stored in `NMC_Live_App/uploads`.

CUDA is synchronized around timed GPU stages. The displayed processing latency includes input reading, hashes, runtime initialization, inference, physics, metric evaluation and saving NPY outputs. It excludes scanner acquisition, browser transfer and ZIP packaging.

The **five-second target** is an engineering target for a supported 128-cubed crop. The app reports whether it was met; it does not promise it. Compare cold and warmed jobs before discussing performance. Peak GPU memory is PyTorch allocation only, not ASTRA or total device memory.

No performance number was measured on your PC while this package was built. Send the first verification `results.zip` to review reproducibility and timing.

## Frozen model identity

Epoch: 12

Checkpoint SHA256:
`85d942f6d2697ab57a91c2e3204b034817e5555603183ef9bf90cfc7eaba8750`

Protocol SHA256:
`c180e08642f780d85ac121be74732bfc492735d9e88691e7a9ed3e9326b33c62`

Reviewed GPU software: PyTorch 2.11.0+cu128 and ASTRA 2.5.0. The model uses float16 autocast, patch size 80, stride 40, 27 tiles, Hann blending, unit scale 0.4 and the reviewed float16 input round-trip. Exact architecture, tiling and physics functions are preserved in `nmc_reference.py`.

## Checks completed during packaging

- Checkpoint and protocol file identities.
- Exact model, tiling, geometry, reconstruction and metric function equivalence to the reviewed script.
- Eight CPU tests: crop/provenance, validation/scaling, tiling, measured geometry/input rejection, PNG format/orientation, cancellation and a full job's persistence/metric plumbing with test doubles.
- Local HTTP routes, token checks, invalid job rejection and asynchronous failure reporting.
- Frontend control and report rendering logic with a DOM test double.

**Not executed here:** CUDA inference, ASTRA reconstruction, actual Windows launch, GPU benchmark or measured-projection validation. There is no PyTorch/ASTRA GPU runtime in the build environment. The self-check on your PC is required. UI visual rendering has not been verified in a full browser in this environment.

Run CPU checks yourself, if desired:
```powershell
& "C:\Users\paam25\MLProjects\Sparse4D_Project\.venv\Scripts\python.exe" "C:\Users\paam25\Downloads\NMC_Live_App\test_app.py"
```

This app does not establish defect detection, battery health, between-specimen independence or time-resolved 4D reconstruction accuracy. Scanner integration still requires data access, correct calibration, acquisition-to-output timing and independent validation.
