# Experimental Waygate archive replay

This separate app preserves the original NMC app. Extract into Downloads so the folder is NMC_Waygate_Replay. Run Launch_Replay.ps1 with the same verified GPU Python environment. It prepares 75 actual archived projections, then runs the local viewer on port 8766. The existing NMC app can keep running on port 8765.

Select Waygate archive replay. The default projection and geometry paths assume C:\Users\paam25\Downloads\NMC_Waygate_Replay. If extracted elsewhere, enter that folder's prepared_waygate/projections.npy and geometry.json paths manually. Read and tick the two acknowledgements, then run. Inspect slices and download results.zip. No training occurs. Preparation reads the local raw archive and writes a new folder; it refuses to overwrite it.

## What is verified
Original PCA, PCP and all 75 selected TIFF hashes must match the previous Waygate experiment. TIFF intensities are averaged 3x3, floored at 0.5, then converted with -log(I/9851), retaining negatives. This reproduces the historical preprocessing recipe, not proof of correct detector calibration. The selected angles come from the original acquisition log.

## What remains approximate
Exported TIFFs have 750 pixels but metadata describes a 1500-pixel ROI. Effective exported detector spacing remains unresolved. The historical 0.2 mm spacing (0.6 mm after 3x3 reduction) is retained explicitly as an assumption. Detector correction status is unknown; no additional dark/flat correction is applied. Tilt and vendor correction are not interpreted. Scanner orientation is not independently checked.

The entire 24.000082 mm field is reconstructed on a 128-cubed grid. This is different from the original 256-cubed Waygate reconstruction and from NMC training voxel scale. Prior Waygate scores do not apply. The NMC model is transferred to another battery type, so improvement is unproven. The app shows fit projection error only, with no reference-based accuracy claim. Fit error does not establish structural truth or held-out performance.

## Timing
This is archived data, not a live scanner connection. The 1200-view historical scan took 1199 seconds between its first and last timestamps. Selecting 75 views from that archive does not reduce the original acquisition time. Reported processing excludes preparation, acquisition, browser transfer, ZIP packaging and display latency. It is not acquisition-to-display timing.

## Verification scope
CPU preparation, geometry validation and existing pipeline unit tests were checked during packaging. Actual GPU execution of this new archived-data replay must be verified on the user's PC. Keep all existing experiment outputs and models.
