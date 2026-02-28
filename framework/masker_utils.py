import cv2
import numpy as np
import os
import torch
from einops import rearrange
from dragger_utils import DynamicRegionEstimator
import numpy as np
from scipy.ndimage import shift


class AdaptiveMaskEstimator:
    @staticmethod
    def _get_independent_regions(mask_array) -> tuple:
        contours, _ = cv2.findContours(
            mask_array,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        region_masks, valid_contours = [], []
        for cnt in contours:
            if cv2.contourArea(cnt) < 5:
                continue
            mask = np.zeros_like(mask_array)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            region_masks.append(mask)
            valid_contours.append(cnt)
        return region_masks, valid_contours


    @staticmethod
    def _translate_region(region_mask, target_pt) -> tuple:
        y_coords, x_coords = np.where(region_mask == 255)
        if not y_coords.size:
            return np.zeros_like(region_mask), np.zeros_like(region_mask), np.zeros_like(region_mask)

        orig_center = (int(np.mean(x_coords)), int(np.mean(y_coords)))
        dx, dy = target_pt[0] - orig_center[0], target_pt[1] - orig_center[1]
        original_only_mask = np.zeros_like(region_mask)
        original_only_mask[y_coords, x_coords] = 255
        processed_only_mask = np.zeros_like(original_only_mask)

        H, W = region_mask.shape[:2]
        for y, x in zip(y_coords, x_coords):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                processed_only_mask[ny, nx] = 255

        return processed_only_mask, original_only_mask


    @staticmethod
    def _anchor_rotate_region(region_mask, target_pt, anchor_pt) -> tuple:
        y_coords, x_coords = np.where(region_mask == 255)
        if not y_coords.size:
            return np.zeros_like(region_mask), np.zeros_like(region_mask)

        original_only_mask = np.zeros_like(region_mask)
        original_only_mask[y_coords, x_coords] = 255

        # begin -> anchor, begin -> target
        orig_center = (np.mean(x_coords), np.mean(y_coords))
        v1 = np.array([orig_center[0] - anchor_pt[0], orig_center[1] - anchor_pt[1]])
        v2 = np.array([target_pt[0] - anchor_pt[0], target_pt[1] - anchor_pt[1]])

        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        theta = np.arctan2(cross, dot)
        angle_deg = -np.degrees(theta)

        M = cv2.getRotationMatrix2D((anchor_pt[0], anchor_pt[1]), angle_deg, 1.0)
        rotated_only_mask = cv2.warpAffine(
            original_only_mask, M, (region_mask.shape[1], region_mask.shape[0]),
            flags=cv2.INTER_NEAREST, borderValue=0
        )
        return rotated_only_mask, original_only_mask


    @staticmethod
    def _get_combined_rotated_rect(original_mask, copied_mask) -> dict:
        combined_mask = cv2.bitwise_or(original_mask, copied_mask)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("\t\t* NoContourFoundWarning")
            return None

        all_points = []
        for cnt in contours:
            all_points.extend(cnt.squeeze().tolist())
        all_points = np.array(all_points, dtype=np.int32)

        rect = cv2.minAreaRect(all_points)
        (cx, cy), (width, height), angle = rect
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        return {
            "center": (int(cx), int(cy)),
            "size": (int(width), int(height)),
            "angle": angle,
            "box": box,
            "area": int(width * height)
        }

    @staticmethod
    def create_adaptive_mask(operation_region_filepath, instruction, device, dtype, output_path=None, debug_mode=False, use_goldens_centroids=True):
        print(f"\n>> Creating Adaptive Mask...")

        op_idxes, begin_pts, target_pts, anchor_pts, operation_tasks = [], [], [], [], []
        for op_idx in sorted(instruction["region_operations"].keys()):
            op_idxes.append(op_idx)
            begin_pts.append(instruction["region_operations"][op_idx]["centroids"][0])
            target_pts.append(instruction["region_operations"][op_idx]["centroids"][1])
            if "anchors" in instruction["region_operations"][op_idx]:
                anchor_pts.append(instruction["region_operations"][op_idx]["anchors"])
            else:
                anchor_pts.append(None)
            operation_tasks.append(instruction["region_operations"][op_idx]["task"])

        # Recognize Regions based on begin_pts:
        original_mask = cv2.imread(operation_region_filepath, 0)
        region_mask_map, contours = AdaptiveMaskEstimator._get_independent_regions(original_mask)
        print(f"\t> Find {len(region_mask_map)} independent white operation regions.")

        point_region_map = {}
        for pt in begin_pts:
            x, y = pt
            for idx, cnt in enumerate(contours):
                result = cv2.pointPolygonTest(cnt, (x, y), False)
                if result >= 0:
                    point_region_map[tuple(pt)] = idx
                    break
        if len(point_region_map) != len(begin_pts):
            print(f"\t\t* RegionUnmatchError: only {len(point_region_map)}/{len(begin_pts)} matched;")

        # Get the expected after-drag regions, then the combined masks:
        merged_source_mask = np.zeros_like(original_mask) if debug_mode else None
        merged_rotated_mask = np.zeros_like(original_mask)
        for idx, (op_idx, begin_pt, target_pt, op_task, anchor_pt) in enumerate(zip(op_idxes, begin_pts, target_pts, operation_tasks, anchor_pts)):
            print(f"\t\t> idx={idx} | begin_pt={begin_pt} | target_pt={target_pt} | op_task={op_task}")
            indep_mask = region_mask_map[point_region_map[tuple(begin_pt)]]

            if op_task in ("transformation", "deformation"):
                processed_only_mask, original_only_mask = AdaptiveMaskEstimator._translate_region(indep_mask, target_pt)
                if debug_mode and merged_source_mask is not None:
                    merged_source_mask = cv2.bitwise_or(merged_source_mask, processed_only_mask)
            elif op_task == "rotation" and anchor_pt is not None:
                print(f"\t\t> anchor_pt={anchor_pt}")
                processed_only_mask, original_only_mask = AdaptiveMaskEstimator._anchor_rotate_region(indep_mask, target_pt, anchor_pt)
                if debug_mode and merged_source_mask is not None:
                    merged_source_mask = cv2.bitwise_or(merged_source_mask, cv2.bitwise_or(processed_only_mask, original_only_mask))
            else:
                print(f"\t\t* Skip idx={idx}: unsupported operation {op_task}")
                continue

            original_tensor = torch.from_numpy(original_only_mask).float() / 255.0
            processed_tensor = torch.from_numpy(processed_only_mask).float() / 255.0
            if debug_mode and (output_path is not None):
                cv2.imwrite(os.path.join(output_path, "mask_orig__.png"), (original_tensor * 255.0).cpu().numpy())
                cv2.imwrite(os.path.join(output_path, "mask_proc__.png"), (processed_tensor * 255.0).cpu().numpy())
            original_tensor = rearrange(original_tensor, "h w -> 1 1 h w").to(device, dtype)
            processed_tensor = rearrange(processed_tensor, "h w -> 1 1 h w").to(device, dtype)

            if (not use_goldens_centroids) or (not instruction["region_operations"][op_idx].get("centroids")):
                original_centroid = DynamicRegionEstimator.compute_centroid(original_tensor)
                processed_centroid = DynamicRegionEstimator.compute_centroid(processed_tensor)
                amended_begin_pt = np.around([original_centroid[0].item(), original_centroid[1].item()], decimals=3)
                amended_target_pt = np.around([processed_centroid[0].item(), processed_centroid[1].item()], decimals=3)
                if debug_mode:
                    golden_centroids = instruction["region_operations"][op_idx]["centroids"]
                    print(f"\n> [Centroids] "
                          f"\ngolden_centroids: (b) {golden_centroids[0]} -> (t) {golden_centroids[1]};"
                          f"\ncompute_centroids: (b) {amended_begin_pt} -> (t) {amended_target_pt};")
                    if ((abs(golden_centroids[0][0] - amended_begin_pt[0]) < 1 and abs(golden_centroids[0][1] - amended_begin_pt[1]) < 1) and
                        (abs(golden_centroids[1][0] - amended_target_pt[0]) < 1 and abs(golden_centroids[1][1] - amended_target_pt[1]) < 1)):
                        amended_begin_pt, amended_target_pt = golden_centroids[0], golden_centroids[1]
                        print(f"\n\t* CentroidUnmatchWarnning: tried to use compute_centroids but inaccurate, golden_centroids applied: (begin) {golden_centroids[0]} vs {amended_begin_pt}; (target) {golden_centroids[1]} vs {amended_target_pt}")
                instruction["region_operations"][op_idx]["centroids"] = torch.tensor(np.array([amended_begin_pt, amended_target_pt]), dtype=dtype, device=device)

            print(f"\n> [Centroids] (b) {instruction['region_operations'][op_idx]['centroids'][0]} -> (t) {instruction['region_operations'][op_idx]['centroids'][1]};")
            set_rrect_mask = AdaptiveMaskEstimator._get_combined_rotated_rect(original_only_mask, processed_only_mask)
            if set_rrect_mask:
                cv2.drawContours(merged_rotated_mask, [set_rrect_mask["box"]], -1, 255, -1)
                print(f"\t\t> Processed Mask idx={idx + 1}: {begin_pt} -> {target_pt}, with Rotated Rect Size: {set_rrect_mask['size']}, Angle: {set_rrect_mask['angle']:.1f}°;")

        # Save to locals
        if output_path is not None:
            if debug_mode and merged_source_mask is not None:
                cv2.imwrite(os.path.join(output_path, "mask_source.png"), merged_source_mask)
            cv2.imwrite(os.path.join(output_path, "mask.png"), merged_rotated_mask)

        # Send to dragger
        merged_source_mask = torch.tensor(merged_source_mask, device=device, dtype=dtype) if debug_mode else None
        merged_rotated_mask = torch.tensor(merged_rotated_mask, device=device, dtype=dtype)
        merged_rotated_mask = merged_rotated_mask.float() / 255.0
        merged_rotated_mask[merged_rotated_mask > 0.0] = 1.0
        merged_rotated_mask = rearrange(merged_rotated_mask, "h w -> 1 1 h w")
        instruction["source_mask"] = merged_source_mask if debug_mode else None
        instruction["mask"] = merged_rotated_mask
        return instruction