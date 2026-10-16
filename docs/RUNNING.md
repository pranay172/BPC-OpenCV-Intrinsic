# Running it

[← Back to the README](../README.md) · [Technical notes](APPROACH.md) · [Checkpoints](../models/README.md)

This rebuilds the estimator the way it was submitted and runs it against the IPD validation split with the organizers' local tester. It's a 2025 competition environment, so expect some reconstruction work rather than a one-command install.

## 🧰 What you need

| | |
| --- | --- |
| **OS** | Linux with Docker, and your user in the `docker` group |
| **GPU** | NVIDIA, with the [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) installed. The submission targeted a 24 GB L4. |
| **Python** | Python 3 with `venv`, plus `7z` for unpacking the dataset |
| **Weights** | SAM ViT-H, DINOv2 ViT-L/14, SAM-6D PEM and MAE ViT-B. See [models/README.md](../models/README.md). |
| **Blender** | Blender **4.1.1** for Linux, used by BlenderProc to render templates |
| **Disk** | Plenty. The finished image is close to 20 GB. |

> [!NOTE]
> The image pins `torch==2.6.0` and compiles PointNet2 for compute capabilities up to 8.6 (+PTX). It ran on L4-class (Ada) GPUs. Very new cards, such as the RTX 50 series (Blackwell), need a newer PyTorch build. The full pipeline doesn't run on CPU.

## 1. Install the organizers' CLI

From the repository root:

```bash
python3 -m venv .venv && source .venv/bin/activate
(cd ibpc_py && pip install .)
bpc --help
```

## 2. Fetch the dataset

```bash
bpc fetch ipd
```

This puts the validation split and the CAD models in `./ipd`. Keep running commands from the repository root: `bpc test` mounts `ipd/models` into the estimator, and that's where the node finds the meshes and `models_info.json`.

## 3. Add the checkpoints

Put the four weight files under `models/checkpoints/` exactly as laid out in [models/README.md](../models/README.md). The Dockerfile copies `models/` into the image, and the node reads `models/checkpoints/` at startup.

You don't need to provide templates. They're rendered when the container starts.

## 4. Add Blender

Download [Blender 4.1.1 for Linux](https://download.blender.org/release/Blender4.1/) and extract it so the `blender` executable sits at:

```text
ibpc_pose_estimator_py/ibpc_pose_estimator_py/Render/blender/blender
```

This folder is ignored by Git but copied into the image.

On its first run, BlenderProc installs its own Python packages into `Render/blender/custom-python-packages/`. The submitted image shipped with that folder already populated, so the container didn't need network access to render. To do the same, render once on the host before building. This also doubles as a check that rendering works:

```bash
pip install blenderproc==2.8.0
blenderproc run --custom-blender-path ibpc_pose_estimator_py/ibpc_pose_estimator_py/Render/blender \
  ibpc_pose_estimator_py/ibpc_pose_estimator_py/Render/render_bop_templates.py \
  --models_dir ipd/models --output_dir /tmp/bpc_templates
```

## 5. Build the images

```bash
docker build -f Dockerfile.estimator -t bpc-sam6d .
docker build -f Dockerfile.tester -t bpc-tester .
```

Building the tester from this checkout keeps it in step with the estimator. Otherwise `bpc` pulls the organizers' latest published tester.

## 6. Run the validation harness

```bash
bpc test bpc-sam6d ipd --tester-image bpc-tester
```

`bpc` starts three containers: the estimator, the tester and a Zenoh router. The estimator renders templates for every part in `ipd/models`, loads the models, and starts serving. The tester then walks the validation split and writes `submission.csv` to the current directory. Pass `--result_directory` to write it somewhere else.

The CSV holds the predictions only. Scoring (mAP over MSSD thresholds from 2 to 20 mm) follows the [challenge's evaluation protocol](https://bpc.opencv.org/web/challenges/challenge-page/1/evaluation) and isn't computed here.

## 🩺 When things go wrong

| Symptom | Where to look |
| --- | --- |
| `Template rendering failed` at startup | Check the Blender path in step 4, and that `ipd/models` contains `models_info.json` and the meshes |
| Missing checkpoint | Compare the path in the error with [models/README.md](../models/README.md) |
| `CUDA out of memory` | Try the `oneshot` branch, which runs pose inference in batches of 12 |
| An object never shows up in the CSV | Errors are caught per object and logged by the estimator. Run `bpc test ... --debug-inside` for a shell |
| `apt` failures while building | `main` installs from the ROS 2 testing repository. The `oneshot` branch switches to stable packages |
