import gorilla
import os
import gc
import numpy as np
import random
import json
import torch
import torchvision.transforms as transforms
import cv2
from Pose_Estimation_Model.utils.data_utils import (
    load_im,
    get_bbox,
    get_point_cloud_from_depth,
    get_resize_rgb_choose,
)
import logging
from Pose_Estimation_Model.model.pose_estimation_model import Net
import pycocotools.mask as cocomask
import trimesh

# Reduce VRAM usage by reducing fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(level=logging.INFO)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class PEM:
    def __init__(self, checkpoints_path, device, gpus="0", config="config/base.yaml", rd_seed=1):
        """
        Initialize the pose estimator with models and configuration.
        """
        self.checkpoints_path = checkpoints_path
        self.gpus = gpus
        self.config_path = os.path.join(BASE_DIR, config)
        self.rd_seed = rd_seed
        
        # RGB transform for preprocessing
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Set GPU devices
        if device == "cuda":
            gorilla.utils.set_cuda_visible_devices(gpu_ids=self.gpus)
        
        # Set random seeds
        random.seed(self.rd_seed)
        torch.manual_seed(self.rd_seed)
        
        # Load configuration
        self.cfg = gorilla.Config.fromfile(self.config_path)
        
        # Initialize model
        logging.info("=> Creating PEM model...")
        self.cfg.model.feature_extraction.checkpoint_dir = self.checkpoints_path
        self.model = Net(self.cfg.model)
        if device == "cuda":
            self.model = self.model.cuda()
        self.model.eval()
        
        logging.info("loading PEM model checkpoint...")
        # Load model checkpoint
        checkpoint_path = os.path.join(self.checkpoints_path, 'sam-6d-pem-base.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        
        logging.info(f"=> Loading checkpoint from {checkpoint_path}")
        gorilla.solver.load_checkpoint(model=self.model, filename=checkpoint_path)
        
        # Ensure MAE pretrained model is available
        vit_checkpoint = os.path.join(self.checkpoints_path, 'mae_pretrain_vit_base.pth')
        if not os.path.exists(vit_checkpoint):
            raise FileNotFoundError(f"MAE pretrained model not found at {vit_checkpoint}")
        
        logging.info("=> PEM Model initialization complete")

    def _get_template(self, path, cfg, tem_index=1):
        """
        Extract template data from a specific template index.
        """
        rgb_path = os.path.join(path, f'rgb_{tem_index}.png')
        mask_path = os.path.join(path, f'mask_{tem_index}.png')
        xyz_path = os.path.join(path, f'xyz_{tem_index}.npy')

        rgb = load_im(rgb_path).astype(np.uint8)
        xyz = np.load(xyz_path).astype(np.float32) / 1000.0  
        mask = load_im(mask_path).astype(np.uint8) == 255

        bbox = get_bbox(mask)
        y1, y2, x1, x2 = bbox
        mask = mask[y1:y2, x1:x2]

        rgb = rgb[:,:,::-1][y1:y2, x1:x2, :]
        if cfg.rgb_mask_flag:
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)

        rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = self.rgb_transform(np.array(rgb))

        choose = (mask>0).astype(np.float32).flatten().nonzero()[0]
        if len(choose) <= cfg.n_sample_template_point:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point, replace=False)
        choose = choose[choose_idx]
        xyz = xyz[y1:y2, x1:x2, :].reshape((-1, 3))[choose, :]

        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)
        return rgb, rgb_choose, xyz

    def get_templates(self, path, cfg):
        """
        Extract all templates from a template directory.
        """
        n_template_view = cfg.n_template_view
        all_tem = []
        all_tem_choose = []
        all_tem_pts = []

        total_nView = 42
        for v in range(n_template_view):
            i = int(total_nView / n_template_view * v)
            tem, tem_choose, tem_pts = self._get_template(path, cfg, i)
            all_tem.append(torch.FloatTensor(tem).unsqueeze(0).cuda())
            all_tem_choose.append(torch.IntTensor(tem_choose).long().unsqueeze(0).cuda())
            all_tem_pts.append(torch.FloatTensor(tem_pts).unsqueeze(0).cuda())
        return all_tem, all_tem_pts, all_tem_choose

    def get_test_data(self, rgb, depth, cam_info, cad_path, seg_data, cfg):
        """
        Process test data for inference.
        """
        dets = seg_data  # keys: scene_id, image_id, category_id, bbox, score, segmentation
        K = np.array(cam_info['cam_K']).reshape(3, 3)

        whole_image = rgb.astype(np.uint8)
        if len(whole_image.shape)==2:
            whole_image = np.concatenate([whole_image[:,:,None], whole_image[:,:,None], whole_image[:,:,None]], axis=2)
        whole_depth = depth.astype(np.float32) * cam_info['depth_scale'] / 1000.0
        whole_pts = get_point_cloud_from_depth(whole_depth, K)

        mesh = trimesh.load_mesh(cad_path)
        model_points = mesh.sample(cfg.n_sample_model_point).astype(np.float32) / 1000.0
        radius = np.max(np.linalg.norm(model_points, axis=1))

        all_rgb = []
        all_cloud = []
        all_rgb_choose = []
        all_score = []
        all_dets = []
        for inst in dets:
            seg = inst['segmentation']
            score = inst['score']

            # mask
            h, w = seg['size']
            try:
                rle = cocomask.frPyObjects(seg, h, w)
            except:
                rle = seg
            mask = cocomask.decode(rle)
            mask = np.logical_and(mask > 0, whole_depth > 0)
            if np.sum(mask) > 32:
                bbox = get_bbox(mask)
                y1, y2, x1, x2 = bbox
            else:
                continue
            mask = mask[y1:y2, x1:x2]
            choose = mask.astype(np.float32).flatten().nonzero()[0]

            # pts
            cloud = whole_pts.copy()[y1:y2, x1:x2, :].reshape(-1, 3)[choose, :]
            center = np.mean(cloud, axis=0)
            tmp_cloud = cloud - center[None, :]
            flag = np.linalg.norm(tmp_cloud, axis=1) < radius * 1.2
            if np.sum(flag) < 4:
                continue
            choose = choose[flag]
            cloud = cloud[flag]

            if len(choose) <= cfg.n_sample_observed_point:
                choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point)
            else:
                choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point, replace=False)
            choose = choose[choose_idx]
            cloud = cloud[choose_idx]

            # rgb
            rgb = whole_image.copy()[y1:y2, x1:x2, :][:,:,::-1]
            if cfg.rgb_mask_flag:
                rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
            rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
            rgb = self.rgb_transform(np.array(rgb))
            rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)

            all_rgb.append(torch.FloatTensor(rgb))
            all_cloud.append(torch.FloatTensor(cloud))
            all_rgb_choose.append(torch.IntTensor(rgb_choose).long())
            all_score.append(score)
            all_dets.append(inst)

        ret_dict = {}
        ret_dict['pts'] = torch.stack(all_cloud).cuda()
        ret_dict['rgb'] = torch.stack(all_rgb).cuda()
        ret_dict['rgb_choose'] = torch.stack(all_rgb_choose).cuda()
        ret_dict['score'] = torch.FloatTensor(all_score).cuda()

        ninstance = ret_dict['pts'].size(0)
        ret_dict['model'] = torch.FloatTensor(model_points).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
        ret_dict['K'] = torch.FloatTensor(K).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
        return ret_dict, whole_image, whole_pts.reshape(-1, 3), model_points, all_dets
    
    def log_memory_usage(self, stage_name):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
            max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)
            logging.info(f"Memory {stage_name}: Allocated: {allocated:.2f}GB, Max Allocated: {max_allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    def run_inference(self, output_dir, cad_path, rgb, depth, cam_info, seg_data, est_score_thresh=0.2):
        """
        Run inference on input data.
        """
        gc.collect()
        torch.cuda.empty_cache()

        logging.info("=> Extracting templates...")
        tem_path = os.path.join(output_dir, 'templates', cad_path.split('/')[-1].split('.')[0])
        all_tem, all_tem_pts, all_tem_choose = self.get_templates(tem_path, self.cfg.test_dataset)

        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():
            all_tem_pts, all_tem_feat = self.model.feature_extraction.get_obj_feats(all_tem, all_tem_pts, all_tem_choose)

        logging.info("=> Loading input data...")
        input_data, img, whole_pts, model_points, detections = self.get_test_data(
            rgb, depth, cam_info, cad_path, seg_data, self.cfg.test_dataset
        )
        ninstance = input_data['pts'].size(0)

        del img, whole_pts, model_points, all_tem_choose, all_tem
        gc.collect()
        torch.cuda.empty_cache()

        logging.info("=> Running pem model...")
        with torch.no_grad():
            input_data['dense_po'] = all_tem_pts.repeat(ninstance, 1, 1)
            input_data['dense_fo'] = all_tem_feat.repeat(ninstance, 1, 1)
            out = self.model(input_data)

        if 'pred_pose_score' in out.keys():
            pose_scores = out['pred_pose_score'] * out['score']
        else:
            pose_scores = out['score']
        pose_scores = pose_scores.detach().cpu().numpy()
        pred_rot = out['pred_R'].detach().cpu().numpy()
        pred_trans = out['pred_t'].detach().cpu().numpy() * 1000

        del all_tem_pts, all_tem_feat, input_data
        gc.collect()
        torch.cuda.empty_cache()

        logging.info("=> filtering & returning detections...")
        filtered_detections = []
        for idx, det in enumerate(detections):
            if pose_scores[idx] < est_score_thresh:
                continue
            det['score'] = float(pose_scores[idx])
            det['R'] = list(pred_rot[idx].tolist())
            det['t'] = list(pred_trans[idx].tolist())
            filtered_detections.append(det)

        gc.collect()
        torch.cuda.empty_cache()
        
        return filtered_detections