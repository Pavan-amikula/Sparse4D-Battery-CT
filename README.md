# Sparse4D Battery CT

An interactive research prototype for reconstructing battery structure from sparse X-ray views. It combines GPU cone-beam reconstruction, a compact residual 3D U-Net, controlled simulations, archived measured-data replay, and readable result viewers.

**Status:** working research demonstration. Live scanner integration, physical calibration, specimen-independent generalization and defect detection remain future work. Sparse4D is the project name; the current evaluated outputs are static 3D volumes.

## Try the visual demo first

No Python or GPU is needed for the saved-results viewers:

1. Download this repository with **Code → Download ZIP**, then extract it.
2. Open `demo/waygate_interactive_review.html` in a browser.
3. Switch between 75, 150, 300, 600 and 1200 views. Compare the three middle slices, adjust brightness, and use the comparison overlay.
4. Open `demo/nmc_reconstruction_dashboard.html` for the NMC experiments.

These pages show actual saved results. They do not reconstruct a new scan or connect to a scanner. The 300-view Preview and 600-view Review choices are provisional demo settings, not validated inspection thresholds.

## Recorded results

| Experiment | Finding | Evaluation scope |
|---|---|---|
| NMC training | 37.64% lower mean hybrid voxel RMSE than SIRT23 | Six spatial test regions within one scan |
| Frozen 2000bar check | 35.38% lower hybrid voxel RMSE than SIRT23 | Separate stack; specimen independence unconfirmed |
| Gaussian sensitivity | Hybrid gain falls from 35.03% to 8.19% from 0% to 5% noise | Six regions; 60 simulated evaluations |
| Measured Waygate replay | Hybrid fit projection RMSE is **5.11% higher** than SIRT23 | Negative transfer under provisional geometry |
| Physics-only Waygate FDK | 300 views: 0.271 s; 1200 views: 0.534 s | Single-run reconstruction timings, 256-cubed grid |

Simulated voxel error, measured projection fit and slice difference are different metrics. The archived acquisition spans 1199 seconds; sub-second reconstruction timings do not demonstrate sub-second capture or end-to-end live performance. The earlier failed frozen transfer and evaluation limits are documented in the reports.

## Repository contents

| Folder | Purpose |
|---|---|
| `apps/nmc/` | Local simulation, enhancement and supported measured-projection interface, port 8765 |
| `apps/waygate_replay/` | Archived Waygate preparation and experimental replay, port 8766 |
| `experiments/` | NMC training, frozen checks, noise evaluation and FDK diagnostics |
| `demo/` | Self-contained offline interactive result viewers |
| `evidence/` | Saved protocols, training history and numerical reports |
| `docs/` | IEEE report, full project explanation and editable IEEE LaTeX source |

Each app includes the frozen epoch-12 checkpoint and the metadata required by its verification checks. Checkpoint SHA-256: `85d942f6d2697ab57a91c2e3204b034817e5555603183ef9bf90cfc7eaba8750`.

## Run the local GPU app on Windows

The original experiments used an NVIDIA RTX 2000 Ada Generation GPU. Use a working environment with NumPy, tifffile, matplotlib, CUDA-enabled PyTorch and ASTRA. `requirements.txt` lists dependencies, but does not select a CUDA driver or guarantee GPU compatibility. The recorded environment reports PyTorch 2.11.0+cu128 and ASTRA 2.5.0.

From the extracted repository root, paste this into PowerShell, changing the Python path to your own environment:

```powershell
powershell -ExecutionPolicy Bypass -File ".\apps\nmc\Launch_NMC.ps1" -Python "C:\Users\paam25\MLProjects\Sparse4D_Project\.venv\Scripts\python.exe"
```

Keep the terminal open. The browser opens at **http://127.0.0.1:8765**. Enter your own data paths in the interface. The original Windows defaults are retained in experiment files and app forms for historical reproducibility; they are not expected to exist on another computer.

Read [NMC app instructions](apps/nmc/README.md) before choosing an input mode. A reconstructed TIFF stack is a volume, not measured detector projections. The verification mode requires the exact reviewed 2000bar source slices. The app runs locally and has no live scanner feed.

## Archived Waygate replay

Raw datasets are downloaded separately. Arrange the measured projections and metadata in `acquisition/` and `metadata/` beneath your Waygate data folder. Prepare the reviewed archive from the repository root:

```powershell
& "C:\path\to\python.exe" ".\apps\waygate_replay\prepare_waygate.py" --base "C:\path\to\waygate750"
& "C:\path\to\python.exe" ".\apps\waygate_replay\app.py" --port 8766
```

Then open **http://127.0.0.1:8766**. Select archive replay and provide the full paths to the prepared `projections.npy` and `geometry.json`. Review the assumption acknowledgements. The preparation script checks the exact reviewed acquisition hashes and refuses to overwrite existing prepared data.

Detector pitch, correction status and scanner orientation remain unresolved. No extra dark/flat subtraction is applied to the exported TIFFs without a verified mapping. See [replay instructions](apps/waygate_replay/README.md).

For physics-only diagnostics, explicitly pass the repository app directory and your data directory:

```powershell
& "C:\path\to\python.exe" ".\experiments\waygate_view_count_check.py" --app ".\apps\waygate_replay" --base "C:\path\to\waygate750"
```

## Reproduction and testing

Use each experiment's `--help` to see its required arguments. Training needs the original data and geometry report. Frozen NMC checks need the training output directory with its locked checkpoint and protocol. The historical first transfer check requires its earlier checkpoint, which differs from the included epoch-12 NMC model.

CPU validation tests exercise input checking, tiling, geometry validation, reporting and HTTP job behavior with mocked GPU operations:

```powershell
Push-Location ".\apps\nmc"
& "C:\path\to\python.exe" -m unittest -v test_app
Pop-Location
Push-Location ".\apps\waygate_replay"
& "C:\path\to\python.exe" -m unittest -v test_app
Pop-Location
```

These tests do not independently validate scanner accuracy or execute the full GPU reconstruction. Keep consumed evaluation regions and views separate from genuinely new validation data.

## Reports

- [IEEE-style project report](docs/Sparse4D_IEEE_Report.pdf)
- [Complete project report](docs/Sparse4D_Complete_Project_Report.pdf)
- [IEEEtran compiled paper](docs/Sparse4D_IEEE_LaTeX.pdf)
- [Editable LaTeX main document](docs/latex/main.tex)

Upload the contents of `docs/latex/` to Overleaf and compile `main.tex` with pdfLaTeX. The title block has no date. Source references and figures are included.

## Data and references

- NMC stacks: [Battery 3D Images, Kaggle](https://www.kaggle.com/datasets/kmader/battery-3d-images).
- Measured battery archive: E. Zwanenburg and J. Warnett, [XCT datasets of battery for reconstruction, Zenodo](https://doi.org/10.5281/zenodo.14993742).
- GPU physics library: [ASTRA Toolbox](https://astra-toolbox.com/).
- Full numbered references and study limitations are in the reports.

Raw TIFF stacks, reconstructed arrays, virtual environments and generated run archives are excluded from Git. Public checkpoint and saved plots are provided for this research demonstration. Data and third-party dependencies retain their own terms. The bundled IEEEtran class retains its original license header.

Author: **Amikula Pavan Kumar Goud**, MSc Computer Science student, Blekinge Institute of Technology, Karlskrona, Sweden.
