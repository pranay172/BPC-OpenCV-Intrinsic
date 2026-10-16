import os
import gc
import math
import numpy as np
import torch
from PIL import Image
import logging
import glob
import trimesh
import torchvision.transforms as T

from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from utils.bbox_utils import CropResizePad, compute_iou, xyxy_to_xywh, force_binary_mask
from model.utils import Detections, mask_to_rle

# Reduce VRAM usage by reducing fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class ISM:
    def __init__(self, checkpoints_path, segmentor_model="sam", device=None):
        """
        Initialize the SAM6D model for instance segmentation and pose estimation.
        
        Args:
            checkpoints_path (str): Path to the directory containing model checkpoints
                                   with 'dinov2', 'FastSAM', and 'segment-anything' subdirectories
            segmentor_model (str): Type of segmentation model to use ("sam" or "fastsam")
            device (str, optional): Device to run the model on (will use CUDA if available when None)
        """
        logging.basicConfig(level=logging.INFO)
        self.checkpoints_path = checkpoints_path
        self.segmentor_model_type = segmentor_model
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Model parameters
        self.visible_thred = 0.5
        self.pointcloud_sample_num = 2048
        self.matching_confidence_thresh = 0.2
        self.ref_data = {}
        
        # Transform for visualization
        self.inv_rgb_transform = T.Compose([
            T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        ])
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize segmentation and descriptor models."""
        logging.info("Initializing models")
        
        # Initialize descriptor model (DINOv2)
        self._initialize_descriptor_model()
        
        # Initialize segmentor model (SAM or FastSAM)
        self._initialize_segmentor_model()
        
        logging.info(f"Models initialized and moved to {self.device}")
        
    def _initialize_descriptor_model(self):
        """Initialize the descriptor model (DINOv2)."""
        from model.dinov2 import CustomDINOv2
        
        self.descriptor_model = CustomDINOv2(
            model_name="dinov2_vitl14",
            token_name="x_norm_clstoken",
            image_size=224,
            chunk_size=16,
            descriptor_width_size=640,
            checkpoint_dir=f"{self.checkpoints_path}/dinov2/",
            validpatch_thresh=0.5
        )
        
        self.descriptor_model.model = self.descriptor_model.model.to(self.device)
        self.descriptor_model.model.eval()
        self.descriptor_model.model.device = self.device

        logging.info(f"Descriptor Model DINOv2 initialized")
        
    def _initialize_segmentor_model(self):
        """Initialize the segmentation model (SAM or FastSAM)."""
        if self.segmentor_model_type == "sam":
            from model.sam import CustomSamAutomaticMaskGenerator, load_sam
            
            sam = load_sam(
                model_type="vit_h",
                checkpoint_dir=f"{self.checkpoints_path}/segment-anything/"
            )
            
            self.segmentor_model = CustomSamAutomaticMaskGenerator(
                sam,
                points_per_batch=64,
                stability_score_thresh=0.85,
                box_nms_thresh=0.7,
                min_mask_region_area=0,
                crop_overlap_ratio=512/1500,
                pred_iou_thresh=0.88,
                segmentor_width_size=640
            )
            
            self.segmentor_model.predictor.model = self.segmentor_model.predictor.model.to(self.device)
            self.segmentor_model.predictor.model.eval()
            logging.info(f"Segmenter Model SAM initialized")
            
        elif self.segmentor_model_type == "fastsam":
            from model.fast_sam import FastSAM
            
            config = {
                "iou_threshold": 0.9,
                "conf_threshold": 0.05,
                "max_det": 200
            }
            
            self.segmentor_model = FastSAM(
                checkpoint_path=f"{self.checkpoints_path}/FastSAM/FastSAM-x.pt",
                config=config,
                segmentor_width_size=640,
                device=self.device
            )
            
            self.segmentor_model.model.setup_model(device=self.device, verbose=True)
            logging.info(f"Segmenter Model FastSAM initialized")
            
        else:
            raise ValueError(f"Unsupported segmentor model: {self.segmentor_model_type}. Use 'sam' or 'fastsam'.")
        
        # Initialize matching metric
        from model.loss import PairwiseSimilarity
        self.matching_metric = PairwiseSimilarity(metric="cosine", chunk_size=16)
        
    def run_inference(self, output_dir, cad_path, rgb, depth, cam_info, seg_score_thresh=0.4):
        """
        Run inference on the provided inputs.
        
        Args:
            output_dir (str): Path to save output results
            cad_path (str): Path to the CAD model
            rgb_path (str): Path to the RGB image
            depth_path (str): Path to the depth image
            cam_path (str): Path to the camera parameters
            
        Returns:
            str: Path to the generated detection results
        """
        with torch.no_grad():
            gc.collect()
            torch.cuda.empty_cache()

            # Load and process templates
            logging.info("Initializing templates")
            self._load_templates(output_dir, cad_path)

            gc.collect()
            torch.cuda.empty_cache()

            # Convert grayscale to RGB if needed
            image = np.array(rgb)
            if len(image.shape) == 2:
                image = np.stack([image, image, image], axis=2)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                image = np.concatenate([image, image, image], axis=2)

            logging.info("Generating masks")
            with torch.amp.autocast('cuda'):
                detections = self.segmentor_model.generate_masks(image)
                detections = Detections(detections)
                query_descriptors, query_appe_descriptors = self.descriptor_model(image, detections)

            gc.collect()
            torch.cuda.empty_cache()

            logging.info("Computing semantic score")
            # Match descriptors
            (
                idx_selected_proposals,
                pred_idx_objects,
                semantic_score,
                best_template,
            ) = self._compute_semantic_score(query_descriptors)

            del query_descriptors

            # Filter detections
            detections.filter(idx_selected_proposals)
            query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

            logging.info("Computing appearance score")
            # Compute appearance score
            appe_scores, ref_aux_descriptor = self._compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)
            
            gc.collect()
            torch.cuda.empty_cache()

            logging.info("Computing geometric score")
            # Compute geometric score
            batch = self._batch_input_data(depth, cam_info)
            self._prepare_geometric_score_computation(cad_path)
            image_uv = self._project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)
            geometric_score, visible_ratio = self._compute_geometric_score(
                image_uv, detections, query_appe_descriptors, ref_aux_descriptor
            )

            # Calculate final score
            final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)

            gc.collect()
            torch.cuda.empty_cache()
            
            logging.info("Finalizing ISM results")
            # Update and save detections
            detections.add_attribute("scores", final_score)
            detections.add_attribute("object_ids", torch.zeros_like(final_score))
            detections.to_numpy()
            
            results = self._detections_to_json(detections, 0, 0, 0, seg_score_thresh=seg_score_thresh)
            
            return results
        
    def log_memory_usage(self, stage_name):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
            max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)
            logging.info(f"Memory {stage_name}: Allocated: {allocated:.2f}GB, Max Allocated: {max_allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    def _detections_to_json(self, detections, scene_id, frame_id, runtime, seg_score_thresh):
        """
        Convert Detections object directly to JSON format without saving to disk
        """
        boxes = xyxy_to_xywh(detections.boxes)
        results = []
        
        for idx_det in range(len(boxes)):
            if float(detections.scores[idx_det]) < seg_score_thresh:
                continue
            # Determine category_id based on dataset
            category_id = int(detections.object_ids[idx_det] + 1)
                
            # Convert mask to RLE format directly
            mask_rle = mask_to_rle(force_binary_mask(detections.masks[idx_det]))
            
            # Create the result object
            result = {
                "scene_id": int(scene_id),
                "image_id": int(frame_id),
                "category_id": category_id,
                "bbox": boxes[idx_det].tolist(),
                "score": float(detections.scores[idx_det]),
                "time": float(runtime),
                "segmentation": mask_rle
            }
            results.append(result)
            
        return results
        
    def _load_templates(self, output_dir, cad_path):
        """Load and process templates for the given CAD model."""
        template_dir = os.path.join(output_dir, 'templates', cad_path.split('/')[-1].split('.')[0])
        num_templates = len(glob.glob(f"{template_dir}/*.npy"))
        
        boxes, masks, templates = [], [], []
        for idx in range(num_templates):
            image = Image.open(os.path.join(template_dir, 'rgb_'+str(idx)+'.png'))
            mask = Image.open(os.path.join(template_dir, 'mask_'+str(idx)+'.png'))
            boxes.append(mask.getbbox())

            image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
            mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
            image = image * mask.unsqueeze(-1)
            templates.append(image)
            masks.append(mask.unsqueeze(-1))
            
        templates = torch.stack(templates).permute(0, 3, 1, 2)
        masks = torch.stack(masks).permute(0, 3, 1, 2)
        boxes = torch.tensor(np.array(boxes))
        
        # Process templates and compute features
        proposal_processor = CropResizePad(target_size=224)
        templates = proposal_processor(images=templates, boxes=boxes).to(self.device)
        masks_cropped = proposal_processor(images=masks, boxes=boxes).to(self.device)

        self.ref_data["descriptors"] = self.descriptor_model.compute_features(
                        templates, token_name="x_norm_clstoken"
                    ).unsqueeze(0).data
        self.ref_data["appe_descriptors"] = self.descriptor_model.compute_masked_patch_feature(
                        templates, masks_cropped[:, 0, :, :]
                    ).unsqueeze(0).data
                    
    def _batch_input_data(self, depth, cam_info):
        """Prepare batch input data from depth and camera information."""
        batch = {}
        depth = np.array(depth).astype(np.int32)
        cam_K = np.array(cam_info['cam_K']).reshape((3, 3))
        depth_scale = np.array(cam_info['depth_scale'])

        batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(self.device)
        batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(self.device)
        batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(self.device)
        return batch
    
    def _prepare_geometric_score_computation(self, cad_path):
        """Prepare data for geometric score computation."""
        template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
        template_poses[:, :3, 3] *= 0.4
        poses = torch.tensor(template_poses).to(torch.float32).to(self.device)
        self.ref_data["poses"] = poses[load_index_level_in_level2(0, "all"), :, :]

        mesh = trimesh.load_mesh(cad_path)
        model_points = mesh.sample(self.pointcloud_sample_num).astype(np.float32) / 1000.0
        self.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(self.device)
    
    def _compute_semantic_score(self, proposal_descriptors):
        """Compute semantic score for the proposals."""
        # Compute matching scores for each proposal
        scores = self.matching_metric(
            proposal_descriptors, self.ref_data["descriptors"]
        )  # N_proposals x N_objects x N_templates
        
        # Aggregate scores using top-5 average
        score_per_proposal_and_object = torch.topk(scores, k=5, dim=-1)[0]
        score_per_proposal_and_object = torch.mean(score_per_proposal_and_object, dim=-1)
        
        # Assign each proposal to the object with highest score
        score_per_proposal, assigned_idx_object = torch.max(
            score_per_proposal_and_object, dim=-1
        )
        
        # Select proposals with confidence above threshold
        idx_selected_proposals = torch.arange(
            len(score_per_proposal), device=score_per_proposal.device
        )[score_per_proposal > self.matching_confidence_thresh]
        
        pred_idx_objects = assigned_idx_object[idx_selected_proposals]
        semantic_score = score_per_proposal[idx_selected_proposals]
        
        # Find the best template view
        filtered_scores = scores[idx_selected_proposals, ...]
        best_template = self._best_template_pose(filtered_scores, pred_idx_objects)
        
        return idx_selected_proposals, pred_idx_objects, semantic_score, best_template
    
    def _best_template_pose(self, scores, pred_idx_objects):
        """Find the best template pose for each proposal."""
        _, best_template_idxes = torch.max(scores, dim=-1)
        N_query, N_object = best_template_idxes.shape[0], best_template_idxes.shape[1]
        pred_idx_objects = pred_idx_objects[:, None].repeat(1, N_object)

        assert N_query == pred_idx_objects.shape[0], "Prediction num != Query num"

        best_template_idx = torch.gather(best_template_idxes, dim=1, index=pred_idx_objects)[:, 0]
        return best_template_idx
    
    def _compute_appearance_score(self, best_pose, pred_objects_idx, query_appe_descriptors):
        """Compute appearance score based on the best template."""
        con_idx = torch.cat((pred_objects_idx[None, :], best_pose[None, :]), dim=0)
        ref_appe_descriptors = self.ref_data["appe_descriptors"][con_idx[0, ...], con_idx[1, ...], ...]

        from model.loss import MaskedPatch_MatrixSimilarity
        aux_metric = MaskedPatch_MatrixSimilarity(metric="cosine", chunk_size=64)
        appe_scores = aux_metric.compute_straight(query_appe_descriptors, ref_appe_descriptors)

        return appe_scores, ref_appe_descriptors
    
    def _project_template_to_image(self, best_pose, pred_object_idx, batch, proposals):
        """Project the reference pointclouds to query image."""
        pose_R = self.ref_data["poses"][best_pose, 0:3, 0:3]  # N_query x 3 x 3
        select_pc = self.ref_data["pointcloud"][pred_object_idx, ...]  # N_query x N_pointcloud x 3
        (N_query, N_pointcloud, _) = select_pc.shape

        # Transform pointcloud
        posed_pc = torch.matmul(pose_R, select_pc.permute(0, 2, 1)).permute(0, 2, 1)
        translate = self._calculate_query_translation(proposals, batch["depth"][0], batch["cam_intrinsic"][0], batch['depth_scale'])
        posed_pc = posed_pc + translate[:, None, :].repeat(1, N_pointcloud, 1)

        # Project pointcloud to image
        cam_intrinsic = batch["cam_intrinsic"][0][None, ...].repeat(N_query, 1, 1).to(torch.float32)
        image_homo = torch.bmm(cam_intrinsic, posed_pc.permute(0, 2, 1)).permute(0, 2, 1)
        image_vu = (image_homo / image_homo[:, :, -1][:, :, None])[:, :, 0:2].to(torch.int)
        
        # Clamp coordinates to image boundaries
        (imageH, imageW) = batch["depth"][0].shape
        image_vu[:, :, 0].clamp_(min=0, max=imageW - 1)
        image_vu[:, :, 1].clamp_(min=0, max=imageH - 1)

        return image_vu
    
    # def _calculate_query_translation(self, proposal, depth, cam_intrinsic, depth_scale):
    #     """Calculate translation from object origin to camera coordinate system."""
    #     (N_query, imageH, imageW) = proposal.squeeze_().shape
    #     masked_depth = proposal * (depth[None, ...].repeat(N_query, 1, 1))
    #     translate = depth_image_to_pointcloud_translate_torch(
    #         masked_depth, depth_scale, cam_intrinsic
    #     )
    #     return translate.to(torch.float32)

    # def _calculate_query_translation(self, proposal, depth, cam_intrinsic, depth_scale):
    #     """Calculate translation from object origin to camera coordinate system using CPU."""
    #     # Move tensors to CPU for processing
    #     proposal_cpu = proposal.squeeze_().cpu()
    #     depth_cpu = depth.cpu()
    #     cam_intrinsic_cpu = cam_intrinsic.cpu()
    #     depth_scale_cpu = depth_scale.cpu() if isinstance(depth_scale, torch.Tensor) else depth_scale
        
    #     (N_query, imageH, imageW) = proposal_cpu.shape
        
    #     # Calculate translations on CPU
    #     translations = []
    #     for query_idx in range(N_query):
    #         # Process one mask at a time to avoid large tensor allocations
    #         masked_depth_single = proposal_cpu[query_idx] * depth_cpu
            
    #         # Calculate mean position on CPU
    #         valid = masked_depth_single > 0
    #         valid_num = torch.count_nonzero(valid) + 1e-8
            
    #         # Create pixel coordinate grid on CPU (only once)
    #         if query_idx == 0:
    #             u = torch.arange(0, imageW, device='cpu')
    #             v = torch.arange(0, imageH, device='cpu')
    #             u, v = torch.meshgrid(u, v, indexing="xy")
            
    #         # Convert depth to metric
    #         Z = masked_depth_single * depth_scale_cpu / 1000
    #         X = (u - cam_intrinsic_cpu[0, 2]) * Z / cam_intrinsic_cpu[0, 0]
    #         Y = (v - cam_intrinsic_cpu[1, 2]) * Z / cam_intrinsic_cpu[1, 1]
            
    #         # Mask invalid points
    #         X = X * valid
    #         Y = Y * valid
    #         Z = Z * valid
            
    #         # Calculate average position
    #         avg_X = torch.sum(X) / valid_num
    #         avg_Y = torch.sum(Y) / valid_num
    #         avg_Z = torch.sum(Z) / valid_num
            
    #         translations.append(torch.tensor([avg_X, avg_Y, avg_Z], device='cpu'))
        
    #     # Stack results and move back to original device if needed
    #     translate = torch.stack(translations)
        
    #     # Move result back to original device
    #     device = proposal.device
    #     return translate.to(device).to(torch.float32)

    def _calculate_query_translation(self, proposal, depth, cam_intrinsic, depth_scale, chunk_size=8):
        """
        Calculate translation from object origin to camera coordinate system.
        Processes masks in chunks to balance memory usage and computational efficiency.
        """
        # Move tensors to CPU for processing
        proposal_cpu = proposal.squeeze_().cpu()
        depth_cpu = depth.cpu()
        cam_intrinsic_cpu = cam_intrinsic.cpu()
        depth_scale_cpu = depth_scale.cpu() if isinstance(depth_scale, torch.Tensor) else depth_scale
        
        N_query, imageH, imageW = proposal_cpu.shape
        
        # Create pixel coordinate grid on CPU (only once)
        u = torch.arange(0, imageW, device='cpu')
        v = torch.arange(0, imageH, device='cpu')
        u, v = torch.meshgrid(u, v, indexing="xy")
        
        # Process masks in chunks
        translations = []
        for start_idx in range(0, N_query, chunk_size):
            end_idx = min(start_idx + chunk_size, N_query)
            chunk_proposals = proposal_cpu[start_idx:end_idx]
            
            # Apply masks to depth image
            chunk_masked_depths = chunk_proposals * depth_cpu.unsqueeze(0).expand(chunk_proposals.shape[0], -1, -1)
            
            # Convert to metric 3D coordinates
            Z = chunk_masked_depths * depth_scale_cpu / 1000
            X = (u.unsqueeze(0) - cam_intrinsic_cpu[0, 2]) * Z / cam_intrinsic_cpu[0, 0]
            Y = (v.unsqueeze(0) - cam_intrinsic_cpu[1, 2]) * Z / cam_intrinsic_cpu[1, 1]
            
            # Create validity mask
            valid = Z > 0
            
            # Calculate translation for each mask in chunk
            chunk_translations = []
            for i in range(len(chunk_proposals)):
                mask_valid = valid[i]
                # When we calculate valid_count, we should check if it's zero
                valid_count = torch.count_nonzero(mask_valid)
                if valid_count > 0:
                    avg_X = torch.sum(X[i] * mask_valid) / valid_count
                    avg_Y = torch.sum(Y[i] * mask_valid) / valid_count
                    avg_Z = torch.sum(Z[i] * mask_valid) / valid_count
                else:
                    # Handle the case where no valid points are found (use zeros or another fallback)
                    avg_X, avg_Y, avg_Z = 0.0, 0.0, 0.0
                
                chunk_translations.append(torch.tensor([avg_X, avg_Y, avg_Z], device='cpu'))
            
            translations.extend(chunk_translations)
            
            # Free memory
            del chunk_proposals, chunk_masked_depths, Z, X, Y, valid
        
        # Stack results and move back to original device
        translate = torch.stack(translations)
        device = proposal.device
        
        return translate.to(device).to(torch.float32)
    
    def _compute_geometric_score(self, image_uv, proposals, appe_descriptors, ref_aux_descriptor):
        """Compute geometric score based on projected pointcloud."""
        from model.loss import MaskedPatch_MatrixSimilarity
        aux_metric = MaskedPatch_MatrixSimilarity(metric="cosine", chunk_size=64)
        visible_ratio = aux_metric.compute_visible_ratio(appe_descriptors, ref_aux_descriptor, self.visible_thred)
        
        # Calculate IoU between projected pointcloud box and detection box
        y1x1 = torch.min(image_uv, dim=1).values
        y2x2 = torch.max(image_uv, dim=1).values
        xyxy = torch.concatenate((y1x1, y2x2), dim=-1)
        iou = compute_iou(xyxy, proposals.boxes)

        return iou, visible_ratio