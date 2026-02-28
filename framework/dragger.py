from copy import deepcopy
import torch
from tqdm import tqdm
from accelerate import Accelerator
from pipeline_flux import FluxDragEditPipeline
from overrider_DiT import Override_FluxTransformer2DModel
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from hookhub import KVHookHub
from dragger_utils import *
from dashboard_utils import *


class Dragger:
    def __init__(self, conf, dtype):
        self.conf = conf
        self.device_0 = "cuda:0"
        self.device_1 = "cuda:1"
        self.dtype = dtype
        self.quant_dtype = qint8
        self.pipeline = None
        self.raw_shape = None
        self.full_shape = None
        self.cut_shape = None
        self.total_operation_count = 0
        self.step_recorder = []
        self.hook_hub = KVHookHub(
            mode=self.conf["mode"],
            eta=self.conf["eta"],
            for_keys=self.conf.get("use_hook__for_keys"),
            for_values=self.conf.get("use_hook__for_values"),
            for_doubles=self.conf.get("use_hook__for_doubles"),
            for_singles=self.conf.get("use_hook__for_singles"),
            for_downs=self.conf.get("use_hook__for_downs"),
            for_mid=self.conf.get("use_hook__for_mid"),
            for_ups=self.conf.get("use_hook__for_ups"),
        ) if self.conf["use_kv_hook"] else None
        self.debug_dir = None


    def set_conf(self, conf):
        self.conf = conf


    def load_pipeline(self):
        transformer = Override_FluxTransformer2DModel.from_pretrained(
            self.conf["model_path_flux"],
            subfolder="transformer",
            torch_dtype=self.dtype
        )
        quantize(transformer, weights=self.quant_dtype)
        freeze(transformer)
        transformer.enable_gradient_checkpointing()
        transformer = transformer.to(self.device_0)
        for param in transformer.parameters():
            param.requires_grad = False

        self.pipeline = FluxDragEditPipeline(transformer=transformer).to(self.device_0)
        if self.conf["use_adapter"]:
            self.pipeline.init_adapter(
                image_encoder_path=self.conf["encoder_path_1"],
                image_encoder_2_path=self.conf["encoder_path_2"],
                subject_ipadapter_cfg=dict(subject_ip_adapter_path=self.conf["adapter_path"], nb_token=1024),
                subject_scale=self.conf["adapter_subject_scale"],
            )


    def process_inversion(self, source_inputs):
        timesteps = schedule(
            num_steps=self.conf["inversion_step_num"],
            image_seq_len=source_inputs["img"].shape[1],
            shift=(self.conf["model_path_flux"] != "black-forest-labs/FLUX.1-schnell")
        )
        z_T = self.pipeline.inverse_fireflow(
            source_inputs=source_inputs,
            timesteps=timesteps,
            inverse=True,
            skip_step_num=self.conf["skip_step_num"],
            guidance=self.conf["inversion_guidance_scale"],
            device=self.device_0,
            return_intermediates=False,
            need_adapt=self.conf["use_adapter"]
        )
        del timesteps
        print(f"\t> Inversion Accomplished.")
        reclaim_memory()
        return z_T


    def process_ennoising(self, image, ts=None):
        def get_noise(num_samples, height, width, device_0="cuda:0", dtype=torch.float32, seed=42):
            return torch.randn(
                num_samples,
                16,
                2 * math.ceil(height / 16),
                2 * math.ceil(width / 16),
                device=device_0,
                dtype=dtype,
                generator=torch.Generator(device=device_0).manual_seed(seed)
            )
        def get_noise_encoded(num_samples, seq_len, dim, device_0="cuda:0", dtype=torch.float32, seed=42):
            return torch.randn(
                num_samples,
                seq_len,
                dim,
                device=device_0,
                dtype=dtype,
                generator=torch.Generator(device=device_0).manual_seed(seed)
            )

        if ts is not None:
            noise = get_noise_encoded(num_samples=image.shape[0], seq_len=image.shape[1], dim=image.shape[2], dtype=self.dtype)
            image = image.to(noise.dtype)

            timesteps = None
            t_prev = torch.tensor(ts[0], device=noise.device, dtype=noise.dtype)
            t_curr = torch.tensor(ts[1], device=noise.device, dtype=noise.dtype)

            z_T = image + (t_curr - t_prev) / 2 * noise
        else:
            noise = get_noise(num_samples=1, height=self.full_shape[0], width=self.full_shape[1], dtype=self.dtype)
            image = image.to(noise.dtype)

            timesteps = schedule(
                num_steps=self.conf["ennoising_step_num"],
                image_seq_len=image.shape[1],
                shift=(self.conf["model_path_flux"] != "black-forest-labs/FLUX.1-schnell")
            )
            t_idx = int((1 - self.conf["img2img_strength"]) * self.conf["ennoising_step_num"])
            t = timesteps[t_idx]
            timesteps = timesteps[t_idx:]

            z_T = (1.0 - t) * image + t * noise
        return z_T, timesteps


    def _extract_latent_features(self, timestep, z, target_inputs):
        z = z.to(self.device_0)
        timestep_vec = torch.full((z.shape[0],), timestep, dtype=z.dtype, device=self.device_0)
        txt = target_inputs["txt"].to(self.device_0)
        vec = target_inputs["vec"].to(self.device_0)
        img_ids = target_inputs["img_ids"].view(-1, target_inputs["img_ids"].size(-1)).to(self.device_0)
        txt_ids = target_inputs["txt_ids"].view(-1, target_inputs["txt_ids"].size(-1)).to(self.device_0)
        guidance_vec = torch.full([1], self.conf["sampling_guidance_scale"], device=self.device_0, dtype=z.dtype)
        guidance_vec = guidance_vec.expand(z.shape[0])

        joint_attention_kwargs = self.pipeline.preprocess_adapter(info_inputs=target_inputs, t_vec=timestep_vec) \
            if self.conf["use_adapter"] else None

        outputs = self.pipeline.transformer(
            hidden_states=z,
            encoder_hidden_states=txt,
            pooled_projections=vec,
            timestep=timestep_vec,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance_vec,
            joint_attention_kwargs=joint_attention_kwargs,
            return_intermediates=True,
        )
        noise_prediction, intermediate_features, _ = outputs[0][0], outputs[1], outputs[2]

        target_features = []
        for target_id in self.conf["target_block_feature_ids_flux"]:
            feature = intermediate_features.get(target_id)
            [feature] = decode_for_calculation(zs=[feature], full_shape=self.full_shape)

            feature = F.interpolate(feature, (self.cut_shape[0], self.cut_shape[1]), mode='bilinear')
            target_features.append(feature)
        return_features = torch.cat(target_features, dim=1)
        del txt, vec, intermediate_features, timestep_vec, img_ids, txt_ids, guidance_vec
        reclaim_memory()
        return noise_prediction, return_features


    def _process_step_images(self, timesteps, target_inputs, full_shape):
        print(f"\t> Processing Step Images...")
        step_images = {}
        for step_record in tqdm(self.step_recorder):
            z_record, timeidx_record, track_count_record, drag_count_record, timeidx_real = step_record
            if (self.conf["show_step_images"] == 2) or (self.conf["show_step_images"] == 3):
                x_record = decode(
                    img=z_record.to(self.device_0),
                    full_height=full_shape[0],
                    full_width=full_shape[1],
                    device=self.device_0,
                    dtype=self.dtype
                )
                step_images[f"**{timeidx_record}-{track_count_record}-{drag_count_record}"] = x_record

            if (self.conf["show_step_images"] == 1) or (self.conf["show_step_images"] == 3):
                v_record = None
                is_first_step = True
                for timeidx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
                    if timeidx < timeidx_record:
                        continue
                    z_mid_record = None
                    if is_first_step:
                        z_mid_record = self.pipeline.sampling_velocity_fireflow(
                            i_real=timeidx_real,
                            t_curr=t_curr,
                            t_prev=t_prev,
                            img=z_record.to(self.device_0),
                            target_inputs=target_inputs,
                            guidance=self.conf["sampling_guidance_scale"],
                            device=self.device_0,
                            dtype=self.dtype,
                            need_adapt=self.conf["use_adapter"]
                        )
                        is_first_step = False

                    z_record, v_record = self.pipeline.sampling_step_fireflow(
                        i_real=timeidx_real,
                        t_curr=t_curr,
                        t_prev=t_prev,
                        img=z_record.to(self.device_0),
                        img_mid=z_mid_record,
                        velocity=v_record,
                        target_inputs=target_inputs,
                        return_intermediates=False,
                        guidance=self.conf["sampling_guidance_scale"],
                        device=self.device_0,
                        dtype=self.dtype,
                        need_adapt=self.conf["use_adapter"],
                        timeidx=timeidx,
                        full_shape=self.full_shape,
                        cut_shape=self.cut_shape,
                    )
                del v_record
                reclaim_memory()

                x_record = decode(
                    img=z_record,
                    full_height=full_shape[0],
                    full_width=full_shape[1],
                    device=self.device_0,
                    dtype=self.dtype
                )
                step_images[f"*{timeidx_record}-{track_count_record}-{drag_count_record}"] = x_record
        print(f"\t> Step Images Decoded")
        del self.step_recorder
        reclaim_memory()
        return step_images


    @staticmethod
    def _graphic_visualizer(operations, save_path):
        canvas, draw = None, None
        unmasked_color = (255, 255, 255)
        r = 2
        for op_id, operation in operations.items():
            curr_region = operation["region_curr"]
            begin_pt, curr_pt, target_pt = operation["points_fit"]

            if canvas is None:
                H, W = curr_region.shape[2], curr_region.shape[3]
                canvas = Image.new('RGB', (W, H), color=(0, 0, 0))
                draw = ImageDraw.Draw(canvas)

            region_mask = curr_region[0, 0].cpu().numpy()
            binary_mask = (region_mask > 0.5).astype(np.uint8) * 255
            mask_img = Image.fromarray(binary_mask, mode='L')
            white_img = Image.new('RGB', (W, H), unmasked_color)
            canvas.paste(white_img, mask=mask_img)

            if (operation["task"] == "rotation") and (operation.get("anchors_fit") is not None):
                anchor_pt = operation["anchors_fit"][0]
                x_a, y_a = float(anchor_pt[0]), float(anchor_pt[1])
                draw.ellipse((x_a - r, y_a - r, x_a + r, y_a + r), outline=(255, 0, 0), width=2)
            x_b, y_b = float(begin_pt[0]), float(begin_pt[1])
            draw.ellipse((x_b - r, y_b - r, x_b + r, y_b + r), outline=(0, 0, 255), width=2)
            x_c, y_c = float(curr_pt[0]), float(curr_pt[1])
            draw.ellipse((x_c - r, y_c - r, x_c + r, y_c + r), outline=(128, 0, 128), width=2)
            x_t, y_t = float(target_pt[0]), float(target_pt[1])
            draw.ellipse((x_t - r, y_t - r, x_t + r, y_t + r), outline=(0, 255, 0), width=2)
            draw.line((x_c, y_c, x_t, y_t), fill=(255, 255, 0), width=1)
        canvas.save(save_path)


    @torch.no_grad()
    def _combine_latents(self, mask, z1, z2, is_z1_decoded=False, is_z2_decoded=False, needs_encoding=True):
        [z1_decoded] = decode_for_calculation(zs=[z1], full_shape=self.full_shape) if not is_z1_decoded else z1
        [z2_decoded] = decode_for_calculation(zs=[z2], full_shape=self.full_shape) if not is_z2_decoded else z2

        z_combined = (z1_decoded * (1 - mask) + z2_decoded * mask)
        if needs_encoding:
            [z_combined] = encode_for_calculation(
                zs=[z_combined],
                full_shape=self.full_shape,
                device=self.device_0,
                dtype=self.dtype
            )
        print(f"\t> [Mask] Percentage of Editable Region: {mask.mean().item():.2%}")
        del z1_decoded, z2_decoded
        return z_combined


    def dragger_step(
        self,
        t_curr,
        t_prev,
        z_drag,
        z_orig,
        target_inputs,
        instruction,
        roundidx,
        image_name=None
    ):
        print("\nt_curr-operate:", t_curr)
        z_drag_ = z_drag.clone().detach()
        z_drag_.requires_grad_(True)

        if self.conf["use_optimizer"]:
            optimizer = torch.optim.SGD([z_drag_], lr=self.conf["lr"])
            accelerator = Accelerator(gradient_accumulation_steps=1, device_placement=False)
            z_drag_, self.pipeline.transformer, optimizer = accelerator.prepare(z_drag_, self.pipeline.transformer, optimizer)
        for param in self.pipeline.transformer.parameters():
            param.requires_grad_(False)

        # Forward F_orig
        with torch.no_grad():
            np_orig, F_orig = self._extract_latent_features(
                z=z_orig,
                target_inputs=target_inputs,
                timestep=t_curr,
            )
            z_orig_ = z_orig + (t_prev - t_curr) / 2 * np_orig
            [z_orig_] = decode_for_calculation(zs=[z_orig_], full_shape=self.full_shape)
            del np_orig, z_orig
            reclaim_memory()

        # K-loop
        stage_scope = range(int(self.conf["max_dragging_num"] + self.conf["max_intensify_num"]))
        for operationidx in stage_scope:
            stage_mode = "TRANSPORT" if operationidx < self.conf["max_dragging_num"] else "INTENSIFY"

            print("\n")
            for op_id, op in instruction["region_operations"].items():
                print(f"\t> [ROUND {roundidx}:{operationidx}] | Set {op_id} | {op['task']} ({stage_mode}) | "
                    f"b={np.around(op['points_fit'][0].tolist(), decimals=0).tolist()} -> "
                    f"c={np.around(op['points_fit'][1].tolist(), decimals=0).tolist()} -> "
                    f"t={np.around(op['points_fit'][2].tolist(), decimals=0).tolist()}")
                if op["task"] == "rotation":
                    print(f"\twhere a={np.around(op['anchors_fit'][0].tolist(), decimals=0).tolist()}")
            print("\n")

            if self.conf["use_optimizer"]:
                optimizer.zero_grad()
            else:
                if z_drag_.grad is not None:
                    z_drag_.grad.zero_()

            # Forward F_drag
            np_drag, F_drag = self._extract_latent_features(z=z_drag_, target_inputs=target_inputs, timestep=t_curr)
            F_drag = F_drag.to(self.device_0)
            del np_drag
            reclaim_memory()

            if stage_mode == "TRANSPORT":
                if "region_curr" not in instruction["region_operations"]["0"]:
                    for operation_id in instruction["region_operations"].keys():
                        operation = instruction["region_operations"][operation_id]
                        if "region_curr" not in operation:
                            operation["region_curr"] = F.interpolate(
                                operation["region"],
                                size=F_orig.shape[2:],
                                mode="bilinear",
                                align_corners=False
                            )
                        operation["region_init"] = operation["region_curr"].detach().clone()
                        centroid = DynamicRegionEstimator.compute_centroid(region=operation["region_curr"])
                        operation["points_fit"][0], operation["points_fit"][1] = centroid, copy.deepcopy(centroid)
                    instruction["progressive_weight"] = 0.0

                    if self.conf["use_affine_visualization"]:
                        self.debug_dir = f"./debug/{image_name}"
                        os.makedirs(self.debug_dir, exist_ok=True)
                        Dragger._graphic_visualizer(instruction["region_operations"], os.path.join(self.debug_dir, f"0.png"))

                instruction = DynamicRegionEstimator._get_progressive_weight(
                    instruction=instruction,
                    timestep_count=roundidx,
                    timestep_max=self.conf["max_operation_num"],
                    dragging_count=operationidx+1,
                    dragging_max=self.conf['max_dragging_num']
                )
            elif stage_mode == "INTENSIFY":
                instruction["progressive_weight"] = 1.0


            operation_loss = 0.0
            for operation_id in instruction["region_operations"].keys():
                operation = instruction["region_operations"][operation_id]

                if stage_mode == "TRANSPORT":
                    operation, grid_estimated = DynamicRegionEstimator.estimate_inprocessing_state(
                        operation=operation,
                        progressive_weight=instruction["progressive_weight"],
                        is_last_operationidx=(operationidx == (self.conf["max_dragging_num"] - 1))
                    )
                else:
                    grid_estimated = operation["full_grid"].clone().detach()

                F_orig_estimated = F.grid_sample(
                    F_orig,
                    grid_estimated.to(F_orig.device),
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                ).to(self.device_1)

                p_curr = (operation["region_curr"] > 0.5).float()
                valid_drag_feat = F_drag.to(self.device_0) * p_curr.to(self.device_0)
                valid_orig_feat = F_orig_estimated * p_curr.to(self.device_1)

                del F_orig_estimated, p_curr, grid_estimated
                reclaim_memory()

                F_drag = F_drag.to(self.device_1)
                operation_set_loss = F.l1_loss(valid_drag_feat.to(self.device_0), valid_orig_feat.to(self.device_0))
                operation_loss = operation_loss + operation_set_loss * operation["norm_weight"]
                del valid_drag_feat, valid_orig_feat, operation_set_loss, operation
                reclaim_memory()

            del F_drag
            reclaim_memory()

            if instruction.get("mask_fit") is None:
                if instruction.get("mask").shape != z_orig_.shape[2:]:
                    instruction["mask"] = F.interpolate(
                        instruction["mask"],
                        size=z_orig_.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                instruction["mask_fit"] = (instruction["mask"] > 0.5).float()

            total_loss = operation_loss
            print(f'\n\t> [ROUND {roundidx}:{operationidx}] Total Loss: {total_loss}:')
            print(f'\t\t* Operation Loss => {operation_loss};')

            # GD
            if self.conf["use_optimizer"]:
                accelerator.backward(total_loss)
                grad = z_drag_.grad
                optimizer.step()
            else:
                grad = torch.autograd.grad(
                    outputs=total_loss,
                    inputs=z_drag_,
                    create_graph=False,
                )[0]
                if grad is not None:
                    with torch.no_grad():
                        if self.conf["use_grad_mask"]:
                            if stage_mode == "INTENSIFY":
                                self.conf["lr"] = 1000.0
                            z_drag_updt = z_drag_ - self.conf["lr"] * grad
                            z_drag_new = self._combine_latents(
                                mask=instruction["mask_fit"],
                                z1=z_orig_,
                                z2=z_drag_updt,
                                is_z1_decoded=True,
                            )
                        else:
                            z_drag_new = z_drag_ - self.conf["lr"] * grad

                        z_drag_.copy_(z_drag_new)
                    z_drag_.requires_grad_(True)
                    print(f"\n\t> [GD] Gradient stats - Max: {self.conf['lr'] * grad.max()}, Min: {self.conf['lr'] * grad.min()}, Avg: {self.conf['lr'] * grad.mean().item()}, NaN: {torch.isnan(grad).any()}\n")
                del grad, z_drag_updt, z_drag_new
                reclaim_memory()

            if self.conf["use_affine_visualization"]:
                Dragger._graphic_visualizer(instruction["region_operations"], os.path.join(self.debug_dir, f"{roundidx}:{operationidx}.png"))

        z_drag = z_drag_.detach()
        z_drag.requires_grad_(False)

        del z_drag_, z_orig_, F_orig, target_inputs
        reclaim_memory()
        return z_drag


    def compute_adaptive_scaler(self, instruction):
        d_golden = self.conf["golden_distance"]
        if self.conf["use_adap_scale"]:
            ds = []
            for op_id in instruction["region_operations"].keys():
                (begin_centroid, target_centroid) = instruction["region_operations"][op_id]["centroids"]
                d_x = begin_centroid[0] - target_centroid[0]
                d_y = begin_centroid[1] - target_centroid[1]
                d = math.hypot(d_x, d_y)
                ds.append(d)
            d_mean = sum(ds) / len(ds)
            print(f"\n> [Adaptive Scaler] dists={ds} (mean={d_mean});")
            if d_mean < d_golden:
                scale_level = 4 if ((avg_size := sum(self.full_shape) / 2.) > 1000) else 3 if (avg_size > 600) else 2
            else:
                scale_level = d_golden / d_mean
        else:
            scale_level = 4 if ((avg_size := sum(self.full_shape) / 2.) > 1000) else 3 if (avg_size > 600) else 2
        print(f"> [Adaptive Scaler] use_adaptives={self.conf['use_adap_scale']}; scale_level={scale_level};\n")
        return scale_level


    def __call__(self, raw_image, instruction, image_name=None):
        print(f"\n>> Prepare Embeddings...")
        raw_shape = (raw_image.size[1], raw_image.size[0])
        full_width = raw_shape[1] if raw_shape[1] % 16 == 0 else raw_shape[1] - raw_shape[1] % 16
        full_height = raw_shape[0] if raw_shape[0] % 16 == 0 else raw_shape[0] - raw_shape[0] % 16
        full_id_image = raw_image.resize((full_width, full_height), Image.BICUBIC)
        full_image = np.array(full_id_image)
        full_image = torch.from_numpy(full_image).permute(2, 0, 1).float() / 127.5 - 1
        full_image = full_image.unsqueeze(0)
        full_image = full_image.to(self.device_0, self.dtype)

        self.raw_shape = raw_shape
        self.full_shape = (full_height, full_width)
        scale_level = self.compute_adaptive_scaler(instruction)
        self.cut_shape = (int(self.full_shape[0] // scale_level), int(self.full_shape[1] // scale_level))
        print(f"\n> scale_level={scale_level}, cut_shape=[{self.cut_shape}]), w/ full_shape=[{self.full_shape}]")
        self.total_operation_count = 0
        self.step_recorder = []

        if self.conf["use_kv_hook"]:
            self.hook_hub.clear_hooks()
            self.hook_hub.set_operation_blocks(combo=self.conf['hook_block_combo'])
            self.hook_hub.register_dit_hooks(dit=self.pipeline.transformer, do='CAPTURE')

        # Encoding
        source_image = encode(
            img=full_image,
            device=self.device_0,
            dtype=self.dtype,
            do_preprocess=True
        )
        print(f"\t> Image Encoded.")
        del full_image

        # Ennoisying
        if self.conf["forward_diffusion_mode"] == "EN":
            source_image, timesteps_EN = self.process_ennoising(image=source_image)
            print(f"\t> Image Ennoised.")

        # Embedding
        source_inputs, target_inputs = prepare_forFlux(
            repo_id=self.conf["model_path_flux"],
            source_image=source_image,
            source_prompt=instruction["background_prompt"],
            target_prompt=instruction["background_prompt"] + instruction["editing_prompt"],
            device=self.device_0,
            quant_dtype=self.quant_dtype,
            dtype=self.dtype,
        )
        del source_image

        if self.conf["use_adapter"]:
            full_id_image = full_id_image.resize((max(full_id_image.size), max(full_id_image.size)))
            subject_image_embeds_dict = self.pipeline.encode_image_emb(siglip_image=full_id_image, dtype=self.dtype)
            target_inputs["s_img_ids"] = subject_image_embeds_dict
            source_inputs["s_img_ids"] = subject_image_embeds_dict
        del full_id_image
        reclaim_memory()
        print(f"\t> Prompts Prepared.")


        # Inversion
        if self.conf["forward_diffusion_mode"] == "IN":
            print(f"\n>> Start Inversion...")
            z_drag = self.process_inversion(source_inputs=source_inputs)
            print(f"\t> Inversion Accomplished.")
        else:
            z_drag = source_inputs["img"]

        z_orig = z_drag.detach().clone()
        target_inputs["img_ids"] = deepcopy(source_inputs["img_ids"])
        del source_inputs
        reclaim_memory()

        # Timestep Update
        print(f"\n>> Prepare Timesteps...")
        if self.conf["forward_diffusion_mode"] == "EN":
            timesteps = timesteps_EN
        else:
            timesteps = schedule(
                num_steps=self.conf["sampling_step_num"],
                image_seq_len=z_drag.shape[1],
                shift=(self.conf["model_path_flux"] != "black-forest-labs/FLUX.1-schnell")
            )
        timesteps = timesteps[self.conf["skip_step_num"]:] if self.conf["skip_step_num"] else timesteps
        operation_steps = timesteps[self.conf["kappa_skip_step_num"]:] if self.conf["kappa_skip_step_num"] else timesteps
        operation_steps = operation_steps[:self.conf["max_operation_num"]] if self.conf["max_operation_num"] else operation_steps
        do_operation = [True if step in operation_steps else False for step in timesteps]
        print(f"\t> Timesteps Acquired.")


        # Hook Update
        if self.conf["use_kv_hook"]:
            print(f"\n>> Prepare Hooks...")
            self.hook_hub.switch_hooks(unet_or_dit=self.pipeline.transformer)
            self.hook_hub.set_operation_timesteps(
                combo=self.conf['hook_timestep_combo'],
                do_operation=do_operation,
            )
            print(f"\t> Hooks Ready for Injection:")
            print(f"\t\t* Timestep Combo => {self.conf['hook_timestep_combo']};")
            print(f"\t\t* Block Combo => {self.conf['hook_block_combo']};")


        print(f"\n>> Preprocess Coordinates...")
        instruction = scale_coordinates(
            instruction=instruction,
            raw_shape=self.raw_shape,
            cut_shape=self.cut_shape
        )
        print(f"\t> Location Coordinates Scaled.")

        print(f"\n>> Start Dragging...")
        v_drag, v_orig = None, None
        for roundidx, (t_curr, t_prev) in enumerate(tqdm(zip(timesteps[:-1], timesteps[1:]), desc="Sampling for DragEdit")):
            timeidx = self.conf["skip_step_num"] + roundidx
            if self.conf["show_step_images"] and (roundidx == 0):
                self.step_recorder.append([z_drag.clone().detach().cpu(), roundidx, 0, 0, timeidx])
            if (self.conf["use_kv_hook"]) and (roundidx > 0):
                self.hook_hub.countdown_hooks()

            z_mid_drag, z_mid_orig = None, None
            if do_operation[roundidx]:
                z_drag = self.dragger_step(
                    t_curr=t_curr,
                    t_prev=t_prev,
                    z_drag=z_drag,
                    z_orig=z_orig,
                    target_inputs=target_inputs,
                    instruction=instruction,
                    roundidx=roundidx,
                    image_name=image_name
                )

            if (roundidx == 0) or (do_operation[roundidx]):
                z_mid_drag = self.pipeline.sampling_velocity_fireflow(
                    i_real=timeidx,
                    t_curr=t_curr,
                    t_prev=t_prev,
                    img=z_drag,
                    target_inputs=target_inputs,
                    guidance=self.conf["sampling_guidance_scale"],
                    device=self.device_0,
                    dtype=self.dtype,
                    need_adapt=self.conf["use_adapter"]
                )

            if roundidx == 0:
                z_mid_orig = self.pipeline.sampling_velocity_fireflow(
                    i_real=timeidx,
                    t_curr=t_curr,
                    t_prev=t_prev,
                    img=z_orig,
                    target_inputs=target_inputs,
                    guidance=self.conf["inversion_guidance_scale"],
                    device=self.device_0,
                    dtype=self.dtype,
                    need_adapt=self.conf["use_adapter"]
                )

            if (self.conf["use_kv_hook"]) and (roundidx == 0):
                self.hook_hub.countdown_hooks()

            print("\nt_curr-step => ", t_curr)
            with torch.no_grad():
                z_drag, v_drag = self.pipeline.sampling_step_fireflow(
                    i_real=timeidx,
                    t_curr=t_curr,
                    t_prev=t_prev,
                    img=z_drag,
                    img_mid=z_mid_drag,
                    velocity=v_drag,
                    target_inputs=target_inputs,
                    return_intermediates=False,
                    guidance=self.conf["sampling_guidance_scale"],
                    device=self.device_0,
                    dtype=self.dtype,
                    need_adapt=self.conf["use_adapter"],
                )

                z_orig, v_orig = self.pipeline.sampling_step_fireflow(
                    i_real=timeidx,
                    t_curr=t_curr,
                    t_prev=t_prev,
                    img=z_orig,
                    img_mid=z_mid_orig,
                    velocity=v_orig,
                    target_inputs=target_inputs,
                    return_intermediates=False,
                    guidance=self.conf["inversion_guidance_scale"],
                    device=self.device_0,
                    dtype=self.dtype,
                    need_adapt=self.conf["use_adapter"],
                )
                z_drag = self._combine_latents(mask=instruction["mask_fit"], z1=z_orig, z2=z_drag)

        del v_drag
        self.hook_hub.unregister_hooks() if self.conf["use_kv_hook"] else None
        reclaim_memory()
        print(f"\t> Dragging Accomplished.")

        x0_drag = decode(
            img=z_drag,
            full_height=self.full_shape[0],
            full_width=self.full_shape[1],
            device=self.device_0,
            dtype=self.dtype
        )
        x0_orig = decode(
            img=z_orig,
            full_height=self.full_shape[0],
            full_width=self.full_shape[1],
            device=self.device_0,
            dtype=self.dtype
        )
        print(f"\t> Image Decoded.")

        step_images = None
        if self.conf["show_step_images"]:
            step_images = self._process_step_images(timesteps, target_inputs, self.full_shape)
            print(f"\t> Step Images Decoded.")

        del z_drag, z_orig, target_inputs
        reclaim_memory()

        return x0_orig, x0_drag, step_images
