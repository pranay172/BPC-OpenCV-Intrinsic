# Checkpoints

[← Back to the README](../README.md) · [Run it](../docs/RUNNING.md)

The pretrained weights aren't in Git. Put them here before building. `Dockerfile.estimator` copies this whole folder to `/opt/ros/underlay/install/models`, and the node looks for them under `checkpoints/`:

```text
models/
└── checkpoints/
    ├── segment-anything/
    │   └── sam_vit_h_4b8939.pth          SAM ViT-H: mask proposals
    ├── dinov2/
    │   └── dinov2_vitl14_pretrain.pth    DINOv2 ViT-L/14: template matching
    ├── sam-6d-pem-base.pth               SAM-6D pose estimation model
    └── mae_pretrain_vit_base.pth         MAE ViT-B: the pose model's image backbone
```

| File | Where to get it |
| --- | --- |
| `sam_vit_h_4b8939.pth` | [Segment Anything](https://github.com/facebookresearch/segment-anything#model-checkpoints), or SAM-6D's `download_sam.py` |
| `dinov2_vitl14_pretrain.pth` | [DINOv2](https://github.com/facebookresearch/dinov2#pretrained-models), or SAM-6D's `download_dinov2.py` |
| `sam-6d-pem-base.pth` | [SAM-6D pose estimation model](https://github.com/JiehongLin/SAM-6D/tree/main/SAM-6D/Pose_Estimation_Model) |
| `mae_pretrain_vit_base.pth` | [MAE](https://github.com/facebookresearch/mae), as referenced by SAM-6D's setup |

Two things you **don't** need to add:

- **CAD models.** The harness mounts `ipd/models` into the container.
- **Templates.** The node renders them into `templates/obj_XXXXXX/` here when it starts.

The FastSAM wrapper in the code expects `checkpoints/FastSAM/FastSAM-x.pt`, but the node runs SAM, so it's optional.

Each set of weights stays under its authors' license.
