# Technical notes

[← Back to the README](../README.md) · [Run it](RUNNING.md)

The README tells the story. This page covers the details: what happens to a single request, which numbers were chosen, and the conventions you'd need to know if you reused the node elsewhere. Everything here describes the **`main`** branch, the code behind the testing-phase submission. The [last section](#the-oneshot-branch) covers what changed on `oneshot`.

## The service contract

The organizers' tester calls one ROS 2 service, `/get_pose_estimates` ([definition](../ibpc_interfaces/srv/GetPoseEstimates.srv)). Each request carries:

- a list of **object IDs** to look for,
- three polarization cameras (RGB, depth, AoLP, DoLP, intrinsics, pose),
- one **Photoneo** structured-light sensor (intensity image, depth, intrinsics, pose).

The response is a list of `(obj_id, score, pose)`. The [estimator node](../ibpc_pose_estimator_py/ibpc_pose_estimator_py/ibpc_pose_estimator.py) parses all four cameras, but **only the Photoneo frame is used**: one image, one depth map, one set of intrinsics. Fusing the other views is left for future work.

## At startup: rendering the templates

The harness mounts the dataset's CAD folder at `/opt/ros/underlay/install/3d_models`. Before the service comes up, [`TemplateRenderer`](../ibpc_pose_estimator_py/ibpc_pose_estimator_py/Render/template_renderer.py) runs [`render_bop_templates.py`](../ibpc_pose_estimator_py/ibpc_pose_estimator_py/Render/render_bop_templates.py) through BlenderProc with a bundled Blender 4.1.

For every object listed in `models_info.json`, the script:

- scales the mesh by `1 / diameter`, so every part fills the frame the same way,
- places the camera at the **42 viewpoints** of SAM-6D's level-0 view sphere (`cam_poses_level0.npy`),
- writes `rgb_i.png`, `mask_i.png` and `xyz_i.npy` (normalized object coordinates) into `model_dir/templates/obj_XXXXXX/`.

Rendering uses the GPU with one sample per pixel. The templates feed DINOv2 features and point sampling, so they don't need to look photorealistic.

## Per request: finding the part

For each requested object ID, [`ISM.run_inference`](../ibpc_pose_estimator_py/ibpc_pose_estimator_py/Instance_Segmentation_Model/run_inference_custom.py) does the following.

**1. Encode the templates.** It loads the 42 views, masks and crops them to 224 px, and computes DINOv2 ViT-L/14 features: one CLS token per view, plus masked patch tokens for the appearance check.

**2. Propose masks.** It runs SAM ViT-H's automatic mask generator on the Photoneo image, after a grayscale-to-RGB expansion if needed. The settings follow SAM-6D: 64 points per batch, predicted IoU ≥ 0.88, stability ≥ 0.85, box NMS at 0.7, and the image resized to 640 px wide.

**3. Score each proposal three ways.**

| Score | How it's computed |
| --- | --- |
| **Semantic** `s` | Cosine similarity between the proposal's CLS token and every template. The score is the mean of the **top 5** views. Proposals below **0.2** are discarded before the other scores are computed. |
| **Appearance** `a` | Masked patch-token similarity against the single best-matching template. |
| **Geometric** `g` | Rotate sampled CAD points by the best template's orientation. Translate them to the mean 3D position of the proposal's depth pixels, project them into the image, and take the IoU of that box with the proposal's box. |

The **visible ratio** `v` is the fraction of the template's patches that find a match (cosine > 0.5) in the proposal. It controls how much the geometric term counts:

```text
score = (s + a + g·v) / (2 + v)
```

Masks scoring **≥ 0.4** survive. They're passed on as COCO-style RLE dictionaries, in memory, without writing anything to disk.

## Per request: recovering the pose

[`PEM.run_inference`](../ibpc_pose_estimator_py/ibpc_pose_estimator_py/Pose_Estimation_Model/run_inference_custom.py) wraps SAM-6D's Pose Estimation Model. It works in four steps.

1. **Build the observed point cloud.** For each mask, it intersects the mask with valid depth and back-projects the pixels. Points farther than 1.2 × the model radius from the centroid are dropped, and the rest are resampled to **2,048 points**. Masks with ≤ 32 valid pixels are skipped.
2. **Encode the model.** It samples **1,024 points** from the CAD mesh and loads template features from the 42 rendered views.
3. **Match.** Coarse point matching gives an initial pose. Fine point matching refines it.
4. **Score.** The pose network's own confidence is multiplied by the mask score from the previous stage. Poses below **0.4** are dropped.

All surviving masks for an object go through the network as **one batch**. On the `oneshot` branch this became batches of 12.

## Conventions worth knowing

| What | Convention |
| --- | --- |
| Depth | The node assumes a depth scale of 0.1: `raw × 0.1` gives millimeters, which are divided by 1,000 internally to get metres. |
| CAD, templates | Millimetres on disk, divided by 1,000 internally. |
| Output translation | Multiplied back by 1,000, so the response is in **millimeters**. |
| Output frame | The **Photoneo camera frame**. No camera-to-world transform is applied. |
| Output rotation | Quaternion in ROS order `(x, y, z, w)`. |

## Where the GPU memory went

The submission had to fit SAM ViT-H, DINOv2 ViT-L and the pose model on a single 24 GB L4. The fixes accumulated one stage at a time:

- **The mask-center calculation runs on the CPU, in chunks of 8 masks.** The original code multiplied every proposal mask by the full-resolution depth image on the GPU in one go, which builds an N × H × W tensor.
- `gc.collect()` and `torch.cuda.empty_cache()` run between every stage, and intermediate tensors are deleted explicitly.
- Everything runs under `torch.no_grad()`. SAM and DINOv2 run under `torch.amp.autocast`.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces fragmentation across requests of different sizes.
- Converting the models wholesale to half precision was tried and reverted.

Per-object failures are caught and logged, and the node moves on to the next object. One bad object costs its own predictions, not the whole image.

## Known limitations

- **Single view.** Three cameras' worth of RGB and polarization data goes unused.
- **Templates are re-encoded on every call.** The DINOv2 template features for an object are recomputed for each request, not cached. That's an easy speed-up left on the table.
- **Fixed paths.** The node expects the container layout from the Dockerfile, including `BASE_DIR` and the 3D-model mount. It isn't a drop-in ROS package for other systems.
- **Fixed thresholds.** The same 0.4 / 0.4 cut-offs apply to every part, regardless of how distinctive it is.

## The `oneshot` branch

`oneshot` adds one commit on top of `main` with the work that followed the testing-phase submission:

| Change | Why |
| --- | --- |
| Pose inference in **batches of 12** | Template features are replicated per instance, so a crowded scene could exhaust memory in a single forward pass. |
| Counters for ISM/PEM errors, CUDA OOMs and empty estimates, logged every 3 images | Visibility into failure modes during long unattended runs. |
| Thresholds lowered to **0.35** (mask) and **0.3** (pose) | Trades some precision for recall. |
| Dockerfiles use stable ROS 2 packages, with `apt upgrade` in place of the `ros2-testing` source | Builds against stable releases, not the testing repository. |
| FastSAM wrapper updated for `ultralytics` 8.3 | Keeps the alternative segmentor buildable. The node still uses SAM. |

## Foundations

The segmentation and pose networks, their weights and most of the vendored utilities come from **[SAM-6D](https://github.com/JiehongLin/SAM-6D)** (Lin et al., *SAM-6D: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation*, CVPR 2024). That work builds on [CNOS](https://github.com/nv-nguyen/cnos), [SAM](https://github.com/facebookresearch/segment-anything) and [DINOv2](https://github.com/facebookresearch/dinov2). The rendering script follows SAM-6D's own template renderer.
