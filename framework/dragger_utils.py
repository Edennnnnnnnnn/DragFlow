import torch.nn.functional as F
from dashboard_utils import *


class DynamicRegionEstimator:
    @staticmethod
    def _process_rotation(region, anchor, degree_delta, weight=1.0):
        _, _, H, W = region.shape
        device = region.device
        dtype = region.dtype

        theta = -torch.deg2rad(torch.tensor(degree_delta * weight, dtype=dtype, device=device))  # 采样用逆变换
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
    def _get_progressive_weight(instruction, timestep_count, timestep_max, dragging_count, dragging_max):
        progressive_weight = (timestep_count * dragging_max + dragging_count) / (timestep_max * dragging_max)
        progressive_weight = max(0.0, min(1.0, progressive_weight))
        step_weight = round(progressive_weight - float(instruction.get("progressive_weight", 0.0)), 3)
        if step_weight < 0.0:
            step_weight = 0.0
        instruction["progressive_weight"] = progressive_weight
        print(f"\t\t* [Weights] Progressive: {progressive_weight} | Step: {step_weight} ")
        return instruction


    @staticmethod
    def compute_centroid(region):
        assert region.dim() == 4
        mask = (region > 0.5).float()
        sum_mask = mask.sum()
        if sum_mask == 0:
            print("\t\t* NoOperationRegionDetectedWarning: op_region has all <0.5, no mask")
            return torch.tensor((0.0, 0.0), device=region.device)
        H, W = region.shape[-2:]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=region.device),
            torch.arange(W, device=region.device),
            indexing='ij'
        )
        centroid_x = (grid_x.float() * mask[0]).sum() / sum_mask
        centroid_y = (grid_y.float() * mask[0]).sum() / sum_mask
        centroid = torch.tensor([centroid_x, centroid_y], device=region.device)
        return centroid.round().long()


    @staticmethod
    def estimate_inprocessing_state(operation, progressive_weight, is_last_operationidx=False):
        assert operation["region_init"].dim() == 4, "\t\t* RegionDimNotFitError"
        _, _, H, W = operation["region_init"].shape
        begin_pt, curr_pt, target_pt = operation["points_fit"]
        anchor_pt = operation["anchors_fit"][0] if "anchors_fit" in operation else None

        if operation["task"] == "rotation":
            if not operation.get("full_delta"):
                assert anchor_pt is not None and len(anchor_pt) == 2, "\t* RotationAnchorNotFoundError"
                dx_begin = begin_pt[0] - anchor_pt[0]
                dy_begin = begin_pt[1] - anchor_pt[1]
                dx_target = target_pt[0] - anchor_pt[0]
                dy_target = target_pt[1] - anchor_pt[1]
                angle_begin = torch.atan2(dy_begin, dx_begin)
                angle_target = torch.atan2(dy_target, dx_target)
                raw_delta = angle_target - angle_begin
                raw_delta_deg = torch.rad2deg(raw_delta).item()
                delta = (raw_delta_deg + 180) % 360 - 180
                operation["full_delta"] = delta

            affine_matrix = DynamicRegionEstimator._process_rotation(
                region=operation["region_init"],
                anchor=anchor_pt,
                degree_delta=operation["full_delta"],
                weight=progressive_weight
            )
        elif operation["task"] in ["transformation", "deformation"]:
            if not operation.get("full_delta"):
                dx_full = target_pt[0] - begin_pt[0]
                dy_full = target_pt[1] - begin_pt[1]
                operation["full_delta"] = (dx_full, dy_full)

            affine_matrix = DynamicRegionEstimator._process_transformation(
                region=operation["region_init"],
                mapping_delta=operation["full_delta"],
                weight=progressive_weight
            )
        else:
            raise ValueError(f"\t* UnsupportedTaskTagError: {operation['task']}")

        grid = F.affine_grid(affine_matrix, operation["region_init"].shape, align_corners=False)
        operation["region_curr"] = F.grid_sample(
            operation["region_init"],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        operation["points_fit"][1] = DynamicRegionEstimator.compute_centroid(operation["region_curr"])
        if is_last_operationidx:
            operation["full_grid"] = grid

        return operation, grid
