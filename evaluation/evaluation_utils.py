import os
import csv
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import gc
from torchvision.transforms import PILToTensor


def _get_region_masks(mask_folder, n_regions, device='cuda'):
    orig_files = []
    for f in os.listdir(mask_folder):
        if f.startswith("mask_orig__") and f.endswith(".png"):
            idx_str = f.split("__")[1].split(".png")[0]
            if idx_str.isdigit():
                orig_files.append((int(idx_str), os.path.join(mask_folder, f)))
    proc_files = []
    for f in os.listdir(mask_folder):
        if f.startswith("mask_proc__") and f.endswith(".png"):
            idx_str = f.split("__")[1].split(".png")[0]
            if idx_str.isdigit():
                proc_files.append((int(idx_str), os.path.join(mask_folder, f)))

    orig_files_sorted = sorted(orig_files, key=lambda x: x[0])[:n_regions]
    proc_files_sorted = sorted(proc_files, key=lambda x: x[0])[:n_regions]

    if len(orig_files_sorted) != n_regions or len(proc_files_sorted) != n_regions:
        raise FileNotFoundError("Insufficient number of masks")

    original_masks = []
    for _, path in orig_files_sorted:
        mask = torch.tensor(np.array(Image.open(path).convert('L')), device=device) / 255.0
        mask[mask > 0] = 1.0
        original_masks.append(mask)

    dragged_masks = []
    for _, path in proc_files_sorted:
        mask = torch.tensor(np.array(Image.open(path).convert('L')), device=device) / 255.0
        mask[mask > 0] = 1.0
        dragged_masks.append(mask)

    return original_masks, dragged_masks


def _preprocess_image(image, device):
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0)
    return image.to(device)


def _tensor_to_image(tensor):
    img_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = (img_np + 1) * 127.5
    return Image.fromarray(img_np.astype(np.uint8))


def _get_mask_bbox(mask):
    nonzero_idx = torch.nonzero(mask)
    if nonzero_idx.shape[0] == 0:
        raise ValueError("No valid target area in the mask")
    y_min = int(nonzero_idx[:, 0].min().item())
    y_max = int(nonzero_idx[:, 0].max().item())
    x_min = int(nonzero_idx[:, 1].min().item())
    x_max = int(nonzero_idx[:, 1].max().item())
    return y_min, y_max, x_min, x_max


def _create_mask(handle_pts, target_pts, img_size):
    """
	Create masks based on pixel distances to point pairs.
	Args:
		handle_pts: Handle point coordinates
		target_pts: Target point coordinates
		img_size: Image dimensions (H,W)
	Returns:
		torch.Tensor: Binary mask
	"""
    handle_pts, target_pts = handle_pts.float(), target_pts.float()
    h, w = img_size

    min_dist = ((handle_pts - target_pts).norm(dim=1) / 2 ** 0.5).clamp(min=5)
    y_grid, x_grid = torch.meshgrid(
        torch.arange(h, device=handle_pts.device),
        torch.arange(w, device=handle_pts.device),
        indexing="ij"
    )

    y_grid = y_grid.expand(len(handle_pts), -1, -1)
    x_grid = x_grid.expand(len(handle_pts), -1, -1)

    handle_dist = (
            (x_grid - handle_pts[:, None, None, 0]) ** 2 + (y_grid - handle_pts[:, None, None, 1]) ** 2).sqrt()
    target_dist = (
            (x_grid - target_pts[:, None, None, 0]) ** 2 + (y_grid - target_pts[:, None, None, 1]) ** 2).sqrt()

    return (handle_dist < min_dist[:, None, None]) | (target_dist < min_dist[:, None, None])


def _nn_get_matches(src_featmaps, trg_featmaps, query, mask=None):
    """
	Find nearest neighbor matches between source and target feature maps.
	Args:
		src_featmaps: Source feature maps
		trg_featmaps: Target feature maps
		query: Query points
		l2_norm: Whether to apply L2 normalization
		mask: Optional mask for valid matches
	Returns:
		torch.Tensor: Matched point coordinates
	"""
    _, c, h, w = src_featmaps.shape
    query = query.long()
    src_feat = src_featmaps[0, :, query[:, 1], query[:, 0]]
    src_feat = F.normalize(src_feat, p=2, dim=0)

    trg_featmaps = F.normalize(trg_featmaps, p=2, dim=1)
    trg_featmaps = trg_featmaps.view(c, -1)
    similarity = torch.mm(src_feat.t(), trg_featmaps)

    if mask is not None:
        similarity = torch.where(
            mask.view(-1, h * w),
            similarity,
            torch.full_like(similarity, -torch.inf)
        )
    best_idx = similarity.argmax(dim=-1)
    y_coords = best_idx // w
    x_coords = best_idx % w

    return torch.stack((x_coords, y_coords), dim=1).float()


def reclaim_memory():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for gpu_id in range(num_gpus):
            try:
                torch.cuda.set_device(gpu_id)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except:
                pass
        gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
            torch.mps.empty_cache()
            gc.collect()
        except:
            pass


class RDragBenchmarker:
    @staticmethod
    def compute_LPIPS_loss(feat1, feat2, loss_fn_alex):
        with torch.no_grad():
            feat1_resized = F.interpolate(feat1, (224, 224), mode='bilinear')
            feat2_resized = F.interpolate(feat2, (224, 224), mode='bilinear')
            return loss_fn_alex(feat1_resized, feat2_resized).cpu().item()

    @staticmethod
    def compute_IF_bg_score(
            original_image,
            dragged_image,
            mask,
            loss_fn_alex,
            debug_dir,
            original_image_PIL,
            dragged_image_PIL,
            debug_enable,
            device
    ):
        C = original_image.shape[1]
        editable_mask_expand = mask.unsqueeze(0).unsqueeze(0).repeat(1, C, 1, 1)
        masked_mask_expand = 1 - editable_mask_expand
        orig_masked_feat = original_image * masked_mask_expand
        dragged_masked_feat = dragged_image * masked_mask_expand

        lpips_loss = RDragBenchmarker.compute_LPIPS_loss(orig_masked_feat, dragged_masked_feat, loss_fn_alex)
        if_msk_score = round(1 - lpips_loss, 3)

        if debug_enable:
            msk_dir = os.path.join(debug_dir, "IF_msk")
            os.makedirs(msk_dir, exist_ok=True)

            # Save the feature map of the masked region (transparent background)
            orig_masked_img = _tensor_to_image(orig_masked_feat)
            orig_masked_img_rgba = orig_masked_img.convert("RGBA")
            alpha = Image.fromarray(((1 - mask).cpu().numpy() * 255).astype(np.uint8)).convert("L")
            orig_masked_img_rgba.putalpha(alpha)
            orig_masked_img_rgba.save(os.path.join(msk_dir, "original_masked_region.png"))

            dragged_masked_img = _tensor_to_image(dragged_masked_feat)
            dragged_masked_img_rgba = dragged_masked_img.convert("RGBA")
            dragged_masked_img_rgba.putalpha(alpha)
            dragged_masked_img_rgba.save(os.path.join(msk_dir, "dragged_masked_region.png"))

            # Mark the bounding box of the masked area (orange)
            y_min, y_max, x_min, x_max = _get_mask_bbox(1 - mask)
            orig_img_with_msk = original_image_PIL.copy().convert("RGBA")
            draw_orig = ImageDraw.Draw(orig_img_with_msk)
            draw_orig.rectangle([(x_min, y_min), (x_max, y_max)], outline="orange", width=3)
            orig_img_with_msk.save(os.path.join(msk_dir, "original_with_masked_bbox.png"))

            drag_img_with_msk = dragged_image_PIL.copy().convert("RGBA")
            draw_drag = ImageDraw.Draw(drag_img_with_msk)
            draw_drag.rectangle([(x_min, y_min), (x_max, y_max)], outline="orange", width=3)
            drag_img_with_msk.save(os.path.join(msk_dir, "dragged_with_masked_bbox.png"))
            print(f"\t\t> IF_bg visualization images are saved to: {msk_dir}")
        return if_msk_score


    @staticmethod
    def compute_IF_s2t_score(
        instruction,
        original_image,
        dragged_image,
        original_reg_masks,
        dragged_reg_masks,
        loss_fn_alex,
        debug_dir,
        original_image_PIL,
        dragged_image_PIL,
        debug_enable,
        device,
        region_idx_list
    ):
        C = original_image.shape[1]
        lpips_losses = []
        s2t_dir = os.path.join(debug_dir, "IF_s2t") if debug_enable else None
        if debug_enable:
            os.makedirs(s2t_dir, exist_ok=True)

        for region_idx, (original_reg_mask, dragged_reg_mask) in zip(region_idx_list, zip(original_reg_masks, dragged_reg_masks)):
            region_init = original_reg_mask.unsqueeze(0).unsqueeze(0).repeat(1, C, 1, 1)
            dragged_mask_expand = dragged_reg_mask.unsqueeze(0).unsqueeze(0).repeat(1, C, 1, 1)

            curr_operation = instruction["region_operations"][str(region_idx)]
            original_image_affined = DynamicRegionEstimatorLT.estimate_affine_state(
                operation=curr_operation,
                region_init=region_init,
                original_image=original_image,
                device=device
            )
            original_feat = original_image_affined * dragged_mask_expand
            dragged_feat = dragged_image * dragged_mask_expand

            if original_feat.shape[2:] != dragged_feat.shape[2:]:
                max_h = max(original_feat.shape[2], dragged_feat.shape[2])
                max_w = max(original_feat.shape[3], dragged_feat.shape[3])
                original_feat = F.pad(original_feat, (0, max_w - original_feat.shape[3], 0, max_h - original_feat.shape[2]))
                dragged_feat = F.pad(dragged_feat, (0, max_w - dragged_feat.shape[3], 0, max_h - dragged_feat.shape[2]))

            lpips_loss = RDragBenchmarker.compute_LPIPS_loss(original_feat, dragged_feat, loss_fn_alex)
            lpips_losses.append(lpips_loss)

            if debug_enable:
                def get_min_rotated_bbox(mask):
                    mask_np = mask.cpu().numpy().astype(np.uint8)
                    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) == 0:
                        return []
                    largest_contour = max(contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(largest_contour)
                    return [(int(p[0]), int(p[1])) for p in cv2.boxPoints(rect)]

                alpha_orig = Image.fromarray((original_reg_mask.cpu().numpy() * 255).astype(np.uint8)).convert("L")
                alpha_drag = Image.fromarray((dragged_reg_mask.cpu().numpy() * 255).astype(np.uint8)).convert("L")

                # Original handle area features
                orig_feat_img = _tensor_to_image(original_feat)
                orig_feat_img_rgba = orig_feat_img.convert("RGBA")
                orig_feat_img_rgba.putalpha(alpha_orig)
                orig_feat_img_rgba.save(os.path.join(s2t_dir, f"{region_idx + 1}_original.png"))

                # Edited target region features
                dragged_feat_img = _tensor_to_image(dragged_feat)
                dragged_feat_img_rgba = dragged_feat_img.convert("RGBA")
                dragged_feat_img_rgba.putalpha(alpha_drag)
                dragged_feat_img_rgba.save(os.path.join(s2t_dir, f"{region_idx + 1}_dragged.png"))

                # Original region after affine transformation
                affined_feat_img = _tensor_to_image(original_image_affined)
                affined_feat_img_rgba = affined_feat_img.convert("RGBA")
                affined_feat_img_rgba.putalpha(alpha_drag)
                affined_feat_img_rgba.save(os.path.join(s2t_dir, f"{region_idx + 1}_affined.png"))

                # Label the minimum bounding rotation matrix (green=original, red=edited)
                orig_box = get_min_rotated_bbox(original_reg_mask)
                orig_img = original_image_PIL.copy().convert("RGBA")
                draw_orig = ImageDraw.Draw(orig_img)
                if orig_box:
                    draw_orig.polygon(orig_box, outline="green", width=3)
                orig_img.save(os.path.join(s2t_dir, f"{region_idx + 1}_original.png"))

                drag_box = get_min_rotated_bbox(dragged_reg_mask)
                drag_img = dragged_image_PIL.copy().convert("RGBA")
                draw_drag = ImageDraw.Draw(drag_img)
                if drag_box:
                    draw_drag.polygon(drag_box, outline="red", width=3)
                drag_img.save(os.path.join(s2t_dir, f"{region_idx + 1}_dragged.png"))

                drag_box = get_min_rotated_bbox(dragged_reg_mask)
                affine_img = _tensor_to_image(original_image_affined).convert("RGBA")
                draw_affine = ImageDraw.Draw(affine_img)
                if drag_box:
                    draw_affine.polygon(drag_box, outline="purple", width=3)
                affine_img.save(os.path.join(s2t_dir, f"{region_idx + 1}_affined.png"))

        avg_lpips = sum(lpips_losses) / len(lpips_losses) if lpips_losses else 0.0
        if_s2t_score = round(1 - avg_lpips, 3)
        if debug_enable:
            print(f"\t\t> IF_s2t visualization images are saved to: {s2t_dir}")
        return if_s2t_score


    @staticmethod
    def compute_IF_s2s_score(
        original_image,
        dragged_image,
        original_masks,
        loss_fn_alex,
        debug_dir,
        original_image_PIL,
        dragged_image_PIL,
        debug_enable,
        device,
        region_idx_list
    ):
        C = original_image.shape[1]
        lpips_losses = []

        s2s_dir = os.path.join(debug_dir, "IF_s2s") if debug_enable else None
        if debug_enable:
            os.makedirs(s2s_dir, exist_ok=True)

        for region_idx, orig_mask in zip(region_idx_list, original_masks):
            orig_mask_expand = orig_mask.unsqueeze(0).unsqueeze(0).repeat(1, C, 1, 1)
            orig_b_feat = original_image * orig_mask_expand
            dragged_b_feat = dragged_image * orig_mask_expand

            if orig_b_feat.shape[2:] != dragged_b_feat.shape[2:]:
                max_h = max(orig_b_feat.shape[2], dragged_b_feat.shape[2])
                max_w = max(orig_b_feat.shape[3], dragged_b_feat.shape[3])
                orig_b_feat = F.pad(
                    orig_b_feat,
                    (0, max_w - orig_b_feat.shape[3], 0, max_h - orig_b_feat.shape[2])
                )
                dragged_b_feat = F.pad(
                    dragged_b_feat,
                    (0, max_w - dragged_b_feat.shape[3], 0, max_h - dragged_b_feat.shape[2])
                )

            lpips_loss = RDragBenchmarker.compute_LPIPS_loss(orig_b_feat, dragged_b_feat, loss_fn_alex)
            lpips_losses.append(lpips_loss)

            if debug_enable:
                alpha = Image.fromarray((orig_mask.cpu().numpy() * 255).astype(np.uint8)).convert("L")
                # Original handle area features
                orig_b_img = _tensor_to_image(orig_b_feat)
                orig_b_img_rgba = orig_b_img.convert("RGBA")
                orig_b_img_rgba.putalpha(alpha)
                orig_b_img_rgba.save(os.path.join(s2s_dir, f"{region_idx + 1}_original_rg.png"))

                # Edited handle area features
                dragged_b_img = _tensor_to_image(dragged_b_feat)
                dragged_b_img_rgba = dragged_b_img.convert("RGBA")
                dragged_b_img_rgba.putalpha(alpha)
                dragged_b_img_rgba.save(os.path.join(s2s_dir, f"{region_idx + 1}_dragged_rg.png"))

                # Mark the bounding box (purple)
                y_min, y_max, x_min, x_max = _get_mask_bbox(orig_mask)
                orig_img_with_b = original_image_PIL.copy().convert("RGBA")
                draw_orig = ImageDraw.Draw(orig_img_with_b)
                draw_orig.rectangle([(x_min, y_min), (x_max, y_max)], outline="purple", width=3)
                orig_img_with_b.save(os.path.join(s2s_dir, f"{region_idx + 1}_original.png"))

                drag_img_with_b = dragged_image_PIL.copy().convert("RGBA")
                draw_drag = ImageDraw.Draw(drag_img_with_b)
                draw_drag.rectangle([(x_min, y_min), (x_max, y_max)], outline="purple", width=3)
                drag_img_with_b.save(os.path.join(s2s_dir, f"{region_idx + 1}_dragged.png"))

        avg_lpips = sum(lpips_losses) / len(lpips_losses) if lpips_losses else 0.0
        if_s2s_score = round(1 - avg_lpips, 3)
        if debug_enable:
            print(f"\t\t> IF_s2s visualization images are saved to: {s2s_dir}")
        return if_s2s_score


    @staticmethod
    def compute_MD_1_score(
        dift_model_MD_1,
        instruction,
        editable_mask,
        original_image_norm,
        dragged_image_norm,
        debug_dir,
        debug_enable=False,
        device="cuda"
    ):
        _, H, W = original_image_norm.shape
        ft_source = dift_model_MD_1.forward(original_image_norm, prompt="", t=261, up_ft_index=1, ensemble_size=8, device=device)
        ft_source = F.interpolate(ft_source, (H, W), mode='bilinear')
        ft_dragged = dift_model_MD_1.forward(dragged_image_norm, prompt="", t=261, up_ft_index=1, ensemble_size=8, device=device)
        ft_dragged = F.interpolate(ft_dragged, (H, W), mode='bilinear')

        if editable_mask.shape != (H, W):
            editable_mask = F.interpolate(
                editable_mask.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
        editable_mask = (editable_mask > 0.5).float()

        cos = nn.CosineSimilarity(dim=1)
        distances, begin_centroids, target_centroids, match_centroids = [], [], [], []
        for op_idx in instruction["region_operations"].keys():
            begin_centroid, target_centroid = instruction["region_operations"][op_idx]["centroids"]
            begin_centroid, target_centroid = torch.tensor(begin_centroid, device=device), torch.tensor(target_centroid, device=device)
            num_channel = ft_source.size(1)
            src_vec = ft_source[0, :, begin_centroid[1], begin_centroid[0]].view(1, num_channel, 1, 1)

            cos_map = cos(src_vec, ft_dragged).cpu().numpy()[0]
            cos_map = np.where(editable_mask.cpu().numpy() == 1, cos_map, -1e9)
            max_rc = np.unravel_index(cos_map.argmax(), cos_map.shape)
            max_rc = (max_rc[1], max_rc[0])

            dist = (target_centroid - torch.tensor(max_rc, device=device)).float().norm()
            distances.append(dist.mean().item())
            if debug_enable:
                begin_centroids.append(begin_centroid)
                target_centroids.append(target_centroid)
                match_centroids.append(max_rc)
            del cos_map, src_vec

        mean_dist = float(np.round(np.mean(distances), 3))

        if debug_enable:
            unm_dir = os.path.join(debug_dir, "MD_1")
            os.makedirs(unm_dir, exist_ok=True)

            mask_alpha = Image.fromarray((editable_mask.cpu().numpy() * 255).astype(np.uint8)).convert("L")
            original_image_PIL = _tensor_to_image(original_image_norm)
            dragged_image_PIL = _tensor_to_image(dragged_image_norm)

            orig_img_rgba = original_image_PIL.convert("RGBA")
            orig_img_rgba.putalpha(mask_alpha)
            orig_img_rgba.save(os.path.join(unm_dir, "original_with_mask.png"))
            dragged_img_rgba = dragged_image_PIL.convert("RGBA")
            dragged_img_rgba.putalpha(mask_alpha)
            dragged_img_rgba.save(os.path.join(unm_dir, "dragged_with_mask.png"))

            drag_with_points = dragged_image_PIL.copy()
            draw_drag = ImageDraw.Draw(drag_with_points)
            for idx in range(len(begin_centroids)):
                target = target_centroids[idx]
                match = match_centroids[idx]
                begin = begin_centroids[idx]

                # Draw green target
                draw_drag.ellipse(
                    [(target[0] - 5, target[1] - 5), (target[0] + 5, target[1] + 5)],
                    fill="green", outline="green", width=2
                )
                # Draw red match
                draw_drag.ellipse(
                    [(match[0] - 5, match[1] - 5), (match[0] + 5, match[1] + 5)],
                    fill="red", outline="red", width=2
                )
                # From blue dot (start) to green dot (target)
                draw_drag.ellipse(
                    [(begin[0] - 5, begin[1] - 5), (begin[0] + 5, begin[1] + 5)],
                    fill="blue", outline="blue", width=2)
                draw_drag.line(
                    [(begin[0], begin[1]), (match[0], match[1])],
                    fill="white", width=3
                )
                draw_drag.line(
                    [(begin[0], begin[1]), (target[0], target[1])],
                    fill="white", width=3
                )
            drag_with_points.save(os.path.join(unm_dir, "dragged_with_centroids.png"))
            print(f"\t\t> MD_1 visualization images are saved to: {unm_dir}")
        return mean_dist

    @torch.no_grad()
    @staticmethod
    def compute_MD_2_score(
        dift_model_MD_2,
        original_image_np,
        dragged_image_np,
        instruction,
        device="cuda",
        dtype=torch.float16
    ):
        begin_centroids, target_centroids = [], []
        for op_id in instruction["region_operations"].keys():
            begin_centroids.append(instruction["region_operations"][op_id]["centroids"][0])
            target_centroids.append(instruction["region_operations"][op_id]["centroids"][1])

        def preprocess_image(image: np.ndarray) -> torch.Tensor:
            image = torch.from_numpy(np.array(image)).float() / 127.5 - 1
            image = image.unsqueeze(0).permute(0, 3, 1, 2)
            return image.to(device).to(dtype)

        begin_centroids = torch.tensor(np.array(begin_centroids), device=device, dtype=torch.long)
        target_centroids = torch.tensor(np.array(target_centroids), device=device, dtype=torch.long)

        # Handle image size mismatch
        if original_image_np.shape != dragged_image_np.shape:
            orig_h, orig_w = original_image_np.shape[:2]
            edit_h, edit_w = dragged_image_np.shape[:2]
            dragged_image_np = cv2.resize(dragged_image_np, (orig_w, orig_h))
            target_centroids = target_centroids * torch.tensor([orig_w, orig_h], device=device)
            target_centroids = (target_centroids / torch.tensor([edit_w, edit_h], device=device)).long()

        image_h, image_w = original_image_np.shape[:2]
        orig_img = F.interpolate(preprocess_image(original_image_np), size=(768, 768)).to(dift_model_MD_2.device)
        edit_img = F.interpolate(preprocess_image(dragged_image_np), size=(768, 768)).to(dift_model_MD_2.device)

        # Extract and process features
        orig_feat = F.interpolate(dift_model_MD_2(orig_img, prompt=""), size=(image_h, image_w))
        edit_feat = F.interpolate(dift_model_MD_2(edit_img, prompt=""), size=(image_h, image_w))

        mask = _create_mask(begin_centroids, target_centroids, (image_h, image_w))
        matched_pts = _nn_get_matches(orig_feat, edit_feat, begin_centroids, mask=mask)

        # Calculate distance metric
        dist = target_centroids - matched_pts
        dist = dist.float() / torch.tensor([image_w, image_h], device=device)
        mean_dist = dist.norm(dim=-1).mean().item() * 100.0
        return mean_dist


    @staticmethod
    def store_outcomes(
        image_name,
        notes,
        if_bg_score,
        if_s2t_score,
        if_s2s_score,
        md_1_score,
        md_2_score,
        csv_output_path="./eval_scores.csv"
    ):
        os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
        file_exists = os.path.isfile(csv_output_path)
        with open(csv_output_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)

            if not file_exists:
                writer.writerow([
                    "image_name",
                    "notes",
                    "IF_bg  ⬆️",
                    "IF_s2t ⬆️",
                    "IF_s2s ⬇️",
                    "MD_1   ⬇️",
                    "MD_2   ⬇️",
                ])
            writer.writerow([
                image_name,
                notes,
                round(if_bg_score, 4),
                round(if_s2t_score, 4),
                round(if_s2s_score, 4),
                round(md_1_score, 3),
                round(md_2_score, 3),
            ])
        print(f"\t> Score Records Cached -> `{csv_output_path}`.")


    @staticmethod
    def score_image(
        image_name,
        dataset_dir,
        edited_image_path,
        loss_fn_alex,
        dift_model_MD_1,
        dift_model_MD_2,
        devices,
        notes="-",
        debug_enable=True,
        debug_path=None,
        csv_output_path=None
    ):
        """ Loading Data """
        instruction_path = os.path.join(dataset_dir, image_name, 'instruction.json')
        with open(instruction_path, 'r') as f:
            instruction = json.load(f)
        region_operations = instruction["region_operations"]
        n_regions = len(region_operations)
        if n_regions == 0:
            raise ValueError("No regional operation information")
        region_idx_list = sorted(map(int, region_operations.keys()))

        # Read editable region mask
        editable_mask_path = os.path.join(dataset_dir, image_name, "mask.png")
        if not os.path.exists(editable_mask_path):
            raise FileNotFoundError(f"Editable region mask does not exist: {editable_mask_path} (White area indicates valid area);")
        editable_mask = torch.tensor(
            np.array(Image.open(editable_mask_path).convert('L')),
            device=devices[0], dtype=torch.float32
        ) / 255.0
        editable_mask[editable_mask > 0.0] = 1.0

        # Read the handle area mask (original + after dragging)
        original_reg_masks, dragged_reg_masks = _get_region_masks(
            mask_folder=os.path.join(dataset_dir, image_name, "temp"),
            n_regions=n_regions,
            device=devices[0]
        )

        original_image_PIL = Image.open(os.path.join(dataset_dir, image_name, 'original_image.png')).convert('RGB')
        original_image_NUMPY = np.array(original_image_PIL)
        original_image = _preprocess_image(original_image_NUMPY, devices[0])
        original_image_norm = (PILToTensor()(original_image_PIL) / 255.0 - 0.5) * 2

        dragged_image_PIL = Image.open(edited_image_path).convert('RGB')
        dragged_image_NUMPY = np.array(dragged_image_PIL)
        dragged_image = _preprocess_image(dragged_image_NUMPY, devices[0])
        dragged_image_norm = (PILToTensor()(dragged_image_PIL) / 255.0 - 0.5) * 2

        """ Computing Scores """

        print(f"\n>> Evaluating IF Scores:")
        print(f"\n\t> Evaluating IF_bg...")
        if_bg_score = RDragBenchmarker.compute_IF_bg_score(
            original_image=original_image,
            dragged_image=dragged_image,
            mask=editable_mask,
            loss_fn_alex=loss_fn_alex,
            debug_dir=debug_path,
            original_image_PIL=original_image_PIL,
            dragged_image_PIL=dragged_image_PIL,
            debug_enable=debug_enable,
            device=devices[0]
        )

        print(f"\n\t> Evaluating IF_s2t...")
        if_s2t_score = RDragBenchmarker.compute_IF_s2t_score(
            instruction=instruction,
            original_image=original_image,
            dragged_image=dragged_image,
            original_reg_masks=original_reg_masks,
            dragged_reg_masks=dragged_reg_masks,
            loss_fn_alex=loss_fn_alex,
            debug_dir=debug_path,
            original_image_PIL=original_image_PIL,
            dragged_image_PIL=dragged_image_PIL,
            debug_enable=debug_enable,
            device=devices[0],
            region_idx_list=region_idx_list
        )

        print(f"\n\t> Evaluating IF_s2s...")
        if_s2s_score = RDragBenchmarker.compute_IF_s2s_score(
            original_image=original_image,
            dragged_image=dragged_image,
            original_masks=original_reg_masks,
            loss_fn_alex=loss_fn_alex,
            debug_dir=debug_path,
            original_image_PIL=original_image_PIL,
            dragged_image_PIL=dragged_image_PIL,
            debug_enable=debug_enable,
            device=devices[0],
            region_idx_list=region_idx_list
        )

        print(f"\n>> Evaluating MD Scores:")
        print(f"\n\t> Evaluating MD_1...")
        md_1_score = RDragBenchmarker.compute_MD_1_score(
            dift_model_MD_1=dift_model_MD_1,
            instruction=instruction,
            editable_mask=editable_mask.to(dift_model_MD_1.pipe.device),
            original_image_norm=original_image_norm.to(dift_model_MD_1.pipe.device),
            dragged_image_norm=dragged_image_norm.to(dift_model_MD_1.pipe.device),
            debug_dir=debug_path,
            debug_enable=debug_enable,
            device=devices[1]
        )

        print(f"\n\t> Evaluating MD_2...")
        md_2_score = RDragBenchmarker.compute_MD_2_score(
            dift_model_MD_2=dift_model_MD_2,
            original_image_np=original_image_NUMPY,
            dragged_image_np=dragged_image_NUMPY,
            instruction=instruction,
            device=devices[0]
        )

        del original_image_PIL, original_image, original_image_norm, dragged_image_PIL, dragged_image, (
            dragged_image_norm), original_reg_masks, dragged_reg_masks
        reclaim_memory()


        print(f"\n\n" + "=" * 90)
        print(f"📊 Image `{image_name}` - Benchmark Evaluation Outcomes")
        print(f"=" * 90)
        print(f"  • IF_bg:  {if_bg_score:.3f} ⬆")
        print(f"  • IF_s2t:  {if_s2t_score:.3f} ⬆")
        print(f"  • IF_s2s:  {if_s2s_score:.3f} ⬇")
        print("·" * 90)
        print(f"  • MD_1:  {md_1_score:.3f} ⬇")
        print(f"  • MD_2:  {md_2_score:.3f} ⬇")
        print(f"=" * 90)

        if csv_output_path is not None:
            RDragBenchmarker.store_outcomes(
                image_name=image_name,
                notes=notes,
                if_bg_score=if_bg_score,
                if_s2t_score=if_s2t_score,
                if_s2s_score=if_s2s_score,
                md_1_score=md_1_score,
                md_2_score=md_2_score,
                csv_output_path=csv_output_path
            )


class DynamicRegionEstimatorLT:
    @staticmethod
    def _process_rotation(region, anchor, degree_delta, weight=1.0):
        _, _, H, W = region.shape
        device = region.device
        dtype = region.dtype

        theta = -torch.deg2rad(torch.tensor(degree_delta * weight, dtype=dtype, device=device))
        cos_a = torch.cos(theta)
        sin_a = torch.sin(theta)

        R = torch.tensor([
            [cos_a, -sin_a, 0.0],
            [sin_a,  cos_a, 0.0],
            [0.0,    0.0,   1.0]
        ], dtype=dtype, device=device)

        cx_pix = anchor[0]
        cy_pix = anchor[1]

        T_to = torch.tensor([
            [1.0, 0.0, -cx_pix],
            [0.0, 1.0, -cy_pix],
            [0.0, 0.0, 1.0]
        ], dtype=dtype, device=device)

        T_back = torch.tensor([
            [1.0, 0.0, cx_pix],
            [0.0, 1.0, cy_pix],
            [0.0, 0.0, 1.0]
        ], dtype=dtype, device=device)

        S_n2p = torch.tensor([
            [W/2.0, 0.0,   W/2.0 - 0.5],
            [0.0,   H/2.0, H/2.0 - 0.5],
            [0.0,   0.0,   1.0]
        ], dtype=dtype, device=device)

        S_p2n = torch.tensor([
            [2.0/W, 0.0,   -(W - 1.0)/W],
            [0.0,   2.0/H, -(H - 1.0)/H],
            [0.0,   0.0,   1.0]
        ], dtype=dtype, device=device)

        affine_matrix = S_p2n @ (T_back @ (R @ (T_to @ S_n2p)))
        return affine_matrix[:2, :].unsqueeze(0)


    @staticmethod
    def _process_transformation(region, mapping_delta, weight=1.0):
        N, C, H, W = region.shape
        device = region.device
        dtype = region.dtype

        weighted_offset = [mapping_delta[0] * weight, mapping_delta[1] * weight]
        dx_norm = -2.0 * weighted_offset[0] / W
        dy_norm = -2.0 * weighted_offset[1] / H

        affine_matrix = torch.tensor([
            [1.0, 0.0, dx_norm],
            [0.0, 1.0, dy_norm]
        ], dtype=dtype, device=device).unsqueeze(0).expand(N, -1, -1)
        return affine_matrix


    @staticmethod
    def estimate_affine_state(operation, region_init, original_image, device="cuda:0"):
        assert region_init.dim() == 4, "\t\t* RegionDimNotFitError"
        _, _, H, W = region_init.shape
        begin_pt, target_pt = operation["centroids"]
        anchor_pt = operation["anchors"] if operation["task"] == "rotation" else None

        if operation["task"] == "rotation":
            assert anchor_pt is not None and len(anchor_pt) == 2, "\t* RotationAnchorNotFoundError"
            dx_begin = torch.tensor(begin_pt[0] - anchor_pt[0], device=device)
            dy_begin = torch.tensor(begin_pt[1] - anchor_pt[1], device=device)
            dx_target = torch.tensor(target_pt[0] - anchor_pt[0], device=device)
            dy_target = torch.tensor(target_pt[1] - anchor_pt[1], device=device)
            angle_begin = torch.atan2(dy_begin, dx_begin)
            angle_target = torch.atan2(dy_target, dx_target)
            full_delta = torch.rad2deg(angle_target - angle_begin).item()

            affine_matrix = DynamicRegionEstimatorLT._process_rotation(
                region=region_init,
                anchor=anchor_pt,
                degree_delta=full_delta,
            )
        elif operation["task"] in ["transformation", "deformation"]:
            dx_full = target_pt[0] - begin_pt[0]
            dy_full = target_pt[1] - begin_pt[1]
            full_delta = (dx_full, dy_full)

            affine_matrix = DynamicRegionEstimatorLT._process_transformation(
                region=region_init,
                mapping_delta=full_delta,
            )
        else:
            raise ValueError(f"\t* UnsupportedTaskTagError: {operation['task']}")

        grid = F.affine_grid(affine_matrix, region_init.shape, align_corners=False)
        region_curr = F.grid_sample(
            original_image,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return region_curr
