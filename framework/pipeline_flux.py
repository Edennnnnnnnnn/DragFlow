import inspect
import os.path
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers.loaders import FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models.controlnet_flux import FluxControlNetModel, FluxMultiControlNetModel
from diffusers.utils import logging
import torch.nn.functional as F
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from dashboard_utils import *
from transformers import SiglipVisionModel, SiglipImageProcessor, AutoModel, AutoImageProcessor, BitsAndBytesConfig
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from adapter.utils import flux_load_lora
from adapter.attn_processor import FluxIPAttnProcessor
from adapter.resampler import CrossLayerCrossScaleProjector



class FluxDragEditPipeline(DiffusionPipeline, FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin):
	model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->transformer->vae"
	_optional_components = ["image_encoder", "feature_extractor"]
	_callback_tensor_inputs = ["latents", "prompt_embeds"]

	def __init__(
		self,
		scheduler=None,
		vae=None,
		text_encoder=None,
		tokenizer=None,
		text_encoder_2=None,
		tokenizer_2=None,
		transformer=None,
		image_encoder=None,
		feature_extractor=None,
		controlnet=None,
	):
		super().__init__()
		self.register_modules(
			vae=vae,
			text_encoder=text_encoder,
			text_encoder_2=text_encoder_2,
			tokenizer=tokenizer,
			tokenizer_2=tokenizer_2,
			transformer=transformer,
			scheduler=scheduler,
			image_encoder=image_encoder,
			feature_extractor=feature_extractor,
			controlnet=controlnet,
		)
		self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels)) \
			if hasattr(self, "vae") and self.vae is not None else 16
		self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
		self.tokenizer_max_length = self.tokenizer.model_max_length \
			if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
		self._joint_attention_kwargs = {}
		self._interrupt = False

		self.default_sample_size = 64
		self.controlnet_keep = []
		self.controlnet_conditioning_scale = 1.0
		self.controlnet_guidance_scale = 1.0
		self.control_mode = None

		self.subject_scale = 0.8
		self.siglip_image_encoder = None
		self.siglip_image_processor = None
		self.dino_image_encoder_2 = None
		self.dino_image_processor_2 = None
		self.subject_image_proj_model = None


	def release_adapter_models(self):
		del self.siglip_image_encoder, self.siglip_image_processor, self.dino_image_encoder_2, self.dino_image_processor_2
		reclaim_memory()
		self.siglip_image_encoder = None
		self.siglip_image_processor = None
		self.dino_image_encoder_2 = None
		self.dino_image_processor_2 = None


	@torch.no_grad()
	def encode_siglip_image_emb(self, siglip_image):
		res = self.siglip_image_encoder(siglip_image, output_hidden_states=True)
		siglip_image_embeds = res.last_hidden_state
		siglip_image_shallow_embeds = torch.cat([res.hidden_states[i] for i in [7, 13, 26]], dim=1)
		return siglip_image_embeds, siglip_image_shallow_embeds


	@torch.no_grad()
	def encode_dinov2_image_emb(self, dinov2_image):
		res = self.dino_image_encoder_2(dinov2_image, output_hidden_states=True)
		dinov2_image_embeds = res.last_hidden_state[:, 1:]
		dinov2_image_shallow_embeds = torch.cat([res.hidden_states[i][:, 1:] for i in [9, 19, 29]], dim=1)
		return dinov2_image_embeds, dinov2_image_shallow_embeds


	@torch.no_grad()
	def encode_image_emb(self, siglip_image, device_0="cuda:0", device_1="cuda:1", dtype=torch.float32):
		object_image_pil = siglip_image
		object_image_pil_low_res = [object_image_pil.resize((384, 384))]
		object_image_pil_high_res = object_image_pil.resize((768, 768))
		object_image_pil_high_res = [
			object_image_pil_high_res.crop((0, 0, 384, 384)),
			object_image_pil_high_res.crop((384, 0, 768, 384)),
			object_image_pil_high_res.crop((0, 384, 384, 768)),
			object_image_pil_high_res.crop((384, 384, 768, 768)),
		]
		nb_split_image = len(object_image_pil_high_res)

		dinov2_image = self.siglip_image_processor(images=object_image_pil_low_res, return_tensors="pt").pixel_values
		siglip_image = self.dino_image_processor_2(images=object_image_pil_low_res, return_tensors="pt").pixel_values
		dinov2_image = dinov2_image.to(device=device_1, dtype=dtype)
		siglip_image = siglip_image.to(device=device_1, dtype=dtype)
		siglip_image_embeds = self.encode_siglip_image_emb(dinov2_image)
		dinov2_image_embeds = self.encode_dinov2_image_emb(siglip_image)

		image_embeds_low_res_deep = torch.cat([siglip_image_embeds[0], dinov2_image_embeds[0]], dim=2)
		image_embeds_low_res_shallow = torch.cat([siglip_image_embeds[1], dinov2_image_embeds[1]], dim=2)

		siglip_image_high_res = self.siglip_image_processor(images=object_image_pil_high_res, return_tensors="pt").pixel_values
		siglip_image_high_res = siglip_image_high_res[None]
		siglip_image_high_res = rearrange(siglip_image_high_res, 'b n c h w -> (b n) c h w')
		siglip_image_high_res = siglip_image_high_res.to(device=device_1, dtype=dtype)
		siglip_image_high_res_embeds = self.encode_siglip_image_emb(siglip_image_high_res)
		siglip_image_high_res_deep = rearrange(siglip_image_high_res_embeds[0], '(b n) l c -> b (n l) c', n=nb_split_image)

		dinov2_image_high_res = self.dino_image_processor_2(images=object_image_pil_high_res, return_tensors="pt").pixel_values
		dinov2_image_high_res = dinov2_image_high_res[None]
		dinov2_image_high_res = rearrange(dinov2_image_high_res, 'b n c h w -> (b n) c h w')
		dinov2_image_high_res = dinov2_image_high_res.to(device=device_1, dtype=dtype)
		dinov2_image_high_res_embeds = self.encode_dinov2_image_emb(dinov2_image_high_res)
		dinov2_image_high_res_deep = rearrange(dinov2_image_high_res_embeds[0], '(b n) l c -> b (n l) c', n=nb_split_image)
		image_embeds_high_res_deep = torch.cat([siglip_image_high_res_deep, dinov2_image_high_res_deep], dim=2)

		image_embeds_dict = dict(
			image_embeds_low_res_shallow=image_embeds_low_res_shallow.to(device_0),
			image_embeds_low_res_deep=image_embeds_low_res_deep.to(device_0),
			image_embeds_high_res_deep=image_embeds_high_res_deep.to(device_0),
		)
		return image_embeds_dict


	@torch.no_grad()
	def init_ccp_and_attn_processor(self, device_0="cuda:0", device_1="cuda:1", dtype=torch.float32, *args, **kwargs):
		subject_ip_adapter_path = kwargs['subject_ip_adapter_path']
		nb_token = kwargs['nb_token']
		state_dict = torch.load(subject_ip_adapter_path, map_location="cpu")

		print(f"=> init attn processor")
		attn_procs = {}
		for idx_attn, (name, v) in enumerate(self.transformer.attn_processors.items()):
			text_encoder_2_d_model = 4096
			layer = FluxIPAttnProcessor(
				hidden_size=self.transformer.config.attention_head_dim * self.transformer.config.num_attention_heads,
				ip_hidden_states_dim=text_encoder_2_d_model,
			).to(device_1, dtype=dtype)
			layer.requires_grad_(False)
			attn_procs[name] = layer
		self.transformer.set_attn_processor(attn_procs)
		tmp_ip_layers = torch.nn.ModuleList(self.transformer.attn_processors.values())
		key_name = tmp_ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)
		tmp_ip_layers.requires_grad_(False)
		print(f"=> load attn processor: {key_name}")

		print(f"=> init project")
		image_proj_model = CrossLayerCrossScaleProjector(
			inner_dim=1152 + 1536,
			num_attention_heads=42,
			attention_head_dim=64,
			cross_attention_dim=1152 + 1536,
			num_layers=4,
			dim=1280,
			depth=4,
			dim_head=64,
			heads=20,
			num_queries=nb_token,
			embedding_dim=1152 + 1536,
			output_dim=4096,
			ff_mult=4,
			timestep_in_dim=320,
			timestep_flip_sin_to_cos=True,
			timestep_freq_shift=0,
		)
		image_proj_model.eval()
		image_proj_model.to(device_1, dtype=dtype)
		key_name = image_proj_model.load_state_dict(state_dict["image_proj"], strict=False)
		image_proj_model.requires_grad_(False)
		print(f"=> load project: {key_name}")
		self.subject_image_proj_model = image_proj_model


	@torch.no_grad()
	def init_adapter(
		self,
		image_encoder_path=None,
		image_encoder_2_path=None,
		subject_ipadapter_cfg=None,
		subject_scale=0.8,
		device_0='cuda:0',
		device_1='cuda:1',
		dtype=torch.float32,
	):
		print(f"=> loading image_encoder_1: {image_encoder_path}")
		image_encoder = SiglipVisionModel.from_pretrained(image_encoder_path)
		image_processor = SiglipImageProcessor.from_pretrained(image_encoder_path)
		image_encoder.eval()
		image_encoder.to(device_1, dtype=dtype)
		self.siglip_image_encoder = image_encoder
		self.siglip_image_processor = image_processor

		print(f"=> loading image_encoder_2: {image_encoder_2_path}")
		image_encoder_2 = AutoModel.from_pretrained(image_encoder_2_path)
		image_processor_2 = AutoImageProcessor.from_pretrained(image_encoder_2_path)
		image_encoder_2.eval()
		image_encoder_2.to(device_1, dtype=dtype)
		image_processor_2.crop_size = dict(height=384, width=384)
		image_processor_2.size = dict(shortest_edge=384)
		self.dino_image_encoder_2 = image_encoder_2
		self.dino_image_processor_2 = image_processor_2

		self.subject_scale = subject_scale
		self.init_ccp_and_attn_processor(device_0=device_0, device_1=device_1, **subject_ipadapter_cfg)


	def _set_controlnet_keep(self, timesteps, control_guidance_start, control_guidance_end):
		timesteps = timesteps[:-1]
		if (not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list)):
			mult = len(self.controlnet.nets) if isinstance(self.controlnet, FluxMultiControlNetModel) else 1
			control_guidance_start, control_guidance_end = (
				mult * [control_guidance_start],
				mult * [control_guidance_end],
			)

		for i in range(len(timesteps)):
			keeps = [1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
					 for s, e in zip(control_guidance_start, control_guidance_end)]
			self.controlnet_keep.append(keeps[0])


	@torch.no_grad()
	def sampling_step_controlnet(
		self,
		i,
		t,
		img,
		target_inputs,
		device="cuda",
		dtype=torch.float32,
	):
		self.controlnet.to(device)
		timestep = torch.tensor(t).expand(img.shape[0]).to(device)
		use_guidance = self.controlnet.config.guidance_embeds
		control_guidance = torch.tensor([self.controlnet_guidance_scale], device=device) if use_guidance else None
		control_guidance = control_guidance.expand(img.shape[0]) if control_guidance is not None else None

		if isinstance(self.controlnet_keep[i], list):
			cond_scale = [c * s for c, s in zip(self.controlnet_conditioning_scale, self.controlnet_keep[i])]
		else:
			controlnet_cond_scale = self.controlnet_conditioning_scale
			if isinstance(controlnet_cond_scale, list):
				controlnet_cond_scale = controlnet_cond_scale[0]
			cond_scale = controlnet_cond_scale * self.controlnet_keep[i]

		controlnet_block_samples, controlnet_single_block_samples = self.controlnet(
			hidden_states=img,
			controlnet_cond=target_inputs["c_img"],
			controlnet_mode=self.control_mode,
			conditioning_scale=cond_scale,
			timestep=timestep,
			guidance=control_guidance,
			pooled_projections=target_inputs["vec"],
			encoder_hidden_states=target_inputs["c_txt"],
			txt_ids=target_inputs["c_txt_ids"],
			img_ids=target_inputs["img_ids"],
			joint_attention_kwargs=None,
			return_dict=False,
		)
		return controlnet_block_samples, controlnet_single_block_samples

	# Abandoned
	@torch.no_grad()
	def inverse_fireflow(
		self,
		source_inputs,
		timesteps,
		inverse,
		skip_step_num=None,
		return_intermediates=True,
		guidance=4.0,
		device="cuda",
		need_adapt=False,
		dtype=torch.float32,
	):
		if inverse:
			timesteps = timesteps[::-1]
			return_intermediates = False
			timesteps = timesteps[:-skip_step_num] if skip_step_num else timesteps
		else:
			timesteps = timesteps[skip_step_num:] if skip_step_num else timesteps

		img = source_inputs["img"].clone().detach().to(device)
		img_ids = source_inputs["img_ids"].to(device)
		txt = source_inputs["txt"].to(device)
		txt_ids = source_inputs["txt_ids"].to(device)
		vec = source_inputs["vec"].to(device)
		guidance_vec = torch.full([1], guidance, device=device, dtype=img.dtype)
		guidance_vec = guidance_vec.expand(img.shape[0])

		velocity = None
		for i, (t_curr, t_prev) in tqdm(enumerate(zip(timesteps[:-1], timesteps[1:])), "Inversion Process" if inverse else "Sampling Process"):
			t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=device)
			img_ids = img_ids.view(-1, img_ids.size(-1))
			txt_ids = txt_ids.view(-1, txt_ids.size(-1))

			if velocity is None:
				joint_attention_kwargs = self.preprocess_adapter(info_inputs=source_inputs, t_vec=t_vec) \
					if need_adapt else None

				outputs = self.transformer(
					hidden_states=img,
					encoder_hidden_states=txt,
					pooled_projections=vec,
					timestep=t_vec,
					img_ids=img_ids,
					txt_ids=txt_ids,
					guidance=guidance_vec,
					return_intermediates=return_intermediates,
					joint_attention_kwargs=joint_attention_kwargs,
					controlnet_block_samples=None,
					controlnet_single_block_samples=None,
					return_dict=False,
					controlnet_blocks_repeat=False,
				)
				pred, intermediates, _ = outputs[0][0], outputs[1], outputs[2]
				del outputs
				reclaim_memory()
			else:
				pred = velocity
			img_mid = img + (t_prev - t_curr) / 2 * pred
			t_vec_mid = torch.full((img.shape[0],), t_curr + (t_prev - t_curr) / 2, dtype=img.dtype, device=device)

			joint_attention_kwargs = self.preprocess_adapter(info_inputs=source_inputs, t_vec=t_vec_mid) \
				if need_adapt else None

			outputs_mid = self.transformer(
				hidden_states=img_mid.to(device),
				encoder_hidden_states=txt.to(device),
				pooled_projections=vec.to(device),
				timestep=t_vec_mid.to(device),
				img_ids=img_ids.to(device),
				txt_ids=txt_ids.to(device),
				guidance=guidance_vec.to(device),
				return_intermediates=return_intermediates,
				joint_attention_kwargs=joint_attention_kwargs,
				controlnet_block_samples=None,
				controlnet_single_block_samples=None,
				return_dict=True,
				controlnet_blocks_repeat=False,
			)
			pred_mid, intermediates_mid, _ = outputs_mid[0][0], outputs_mid[1], outputs_mid[2]
			velocity = pred_mid
			img = img + (t_prev - t_curr) * pred_mid
		return img


	def preprocess_adapter(self, info_inputs, t_vec, device_0="cuda:0", dtype=torch.float32):
		assert info_inputs["s_img_ids"] is not None, "SubjectImageEmbdedsNotFound"
		txt = info_inputs["txt"].to(device_0)
		low_res_shallow = info_inputs["s_img_ids"]['image_embeds_low_res_shallow']
		low_res_deep = info_inputs["s_img_ids"]['image_embeds_low_res_deep']
		high_res_deep = info_inputs["s_img_ids"]['image_embeds_high_res_deep']

		low_res_shallow_temp = low_res_shallow.clone().requires_grad_(False)
		low_res_deep_temp = low_res_deep.clone().requires_grad_(False)
		high_res_deep_temp = high_res_deep.clone().requires_grad_(False)

		subject_image_prompt_embeds, _ = self.subject_image_proj_model(
			low_res_shallow=low_res_shallow_temp,
			low_res_deep=low_res_deep_temp,
			high_res_deep=high_res_deep_temp,
			timesteps=t_vec.to(dtype=dtype),
			need_temb=True
		)

		"""
		if 'subject_emb_dict' in self._joint_attention_kwargs:
			del self._joint_attention_kwargs['subject_emb_dict']
		if 'emb_dict' in self._joint_attention_kwargs:
			del self._joint_attention_kwargs['emb_dict']
		del low_res_shallow_temp, low_res_deep_temp, high_res_deep_temp, _
		reclaim_memory()

		self._joint_attention_kwargs['emb_dict'] = dict(length_encoder_hidden_states=txt.shape[1])
		self._joint_attention_kwargs['subject_emb_dict'] = dict(
			ip_hidden_states=subject_image_prompt_embeds,
			scale=self.subject_scale,
		)
		"""
		joint_attention_kwargs = dict(
			emb_dict=dict(length_encoder_hidden_states=txt.shape[1]),
			subject_emb_dict=dict(
				ip_hidden_states=subject_image_prompt_embeds,
				scale=self.subject_scale,
			)
		)
		return joint_attention_kwargs


	@torch.no_grad()
	def sampling_velocity_fireflow(
		self,
		i_real,
		t_curr,
		t_prev,
		img,
		target_inputs,
		guidance=3.5,
		device="cuda",
		dtype=torch.float32,
		need_control=False,
		need_adapt=False,
		cyclic_ennoising=False
	):
		img_ids = target_inputs["img_ids"].to(device)
		txt = target_inputs["txt"].to(device)
		txt_ids = target_inputs["txt_ids"].to(device)
		vec = target_inputs["vec"].to(device)

		img = img.clone().detach()
		guidance_vec = torch.full([1], guidance, device=device, dtype=img.dtype)
		guidance_vec = guidance_vec.expand(img.shape[0]).to(device)

		t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=device)
		img_ids = img_ids.view(-1, img_ids.size(-1)).to(device)
		txt_ids = txt_ids.view(-1, txt_ids.size(-1)).to(device)

		c_double_samples, c_single_samples = None, None
		if need_control:
			c_double_samples, c_single_samples = self.sampling_step_controlnet(
				i=i_real,
				t=t_curr,
				img=img,
				target_inputs=target_inputs,
			)
		joint_attention_kwargs = self.preprocess_adapter(info_inputs=target_inputs, t_vec=t_vec) \
			if need_adapt else None

		outputs = self.transformer(
			hidden_states=img,
			encoder_hidden_states=txt,
			pooled_projections=vec,
			timestep=t_vec,
			img_ids=img_ids,
			txt_ids=txt_ids,
			guidance=guidance_vec,
			return_intermediates=False,
			joint_attention_kwargs=joint_attention_kwargs,
			controlnet_block_samples=c_double_samples,
			controlnet_single_block_samples=c_single_samples,
			return_dict=False,
			controlnet_blocks_repeat=False,
		)
		velocity, _, _ = outputs[0][0], outputs[1], outputs[2]
		img_mid = img + (t_prev - t_curr) / 2 * velocity
		del velocity
		reclaim_memory()

		if cyclic_ennoising:
			img_mid = self.sampling_velocity_fireflow(
				i_real=None,
				t_curr=t_prev,
				t_prev=t_curr,
				img=img_mid,
				target_inputs=target_inputs,
				guidance=guidance,
				device=device,
				dtype=dtype,
				need_control=False,
				need_adapt=need_adapt,
				cyclic_ennoising=False
			)
		return img_mid


	@torch.no_grad()
	def sampling_step_fireflow(
		self,
		i_real,
		t_curr,
		t_prev,
		img,
		img_mid,
		velocity,
		target_inputs,
		return_intermediates=False,
		guidance=3.0,
		device="cuda",
		dtype=torch.float32,
		need_control=False,
		need_adapt=False,
	):
		assert (velocity is not None) or (img_mid is not None), "StepInfoNotFoundError"
		img_ids = target_inputs["img_ids"].to(device)
		txt = target_inputs["txt"].to(device)
		txt_ids = target_inputs["txt_ids"].to(device)
		vec = target_inputs["vec"].to(device)

		img = img.clone().detach().to(device)
		guidance_vec = torch.full([1], guidance, device=device, dtype=img.dtype)
		guidance_vec = guidance_vec.expand(img.shape[0]).to(device)

		#t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=device)
		img_ids = img_ids.view(-1, img_ids.size(-1))
		txt_ids = txt_ids.view(-1, txt_ids.size(-1))

		if img_mid is None:
			img_mid = img + (t_prev - t_curr) / 2 * velocity
			del velocity
			reclaim_memory()

		t_mid = t_curr + (t_prev - t_curr) / 2
		t_vec_mid = torch.full((img.shape[0],), t_mid, dtype=img.dtype, device=device)

		c_double_samples, c_single_samples = None, None
		if need_control:
			c_double_samples, c_single_samples = self.sampling_step_controlnet(
				i=i_real,
				t=t_mid,
				img=img,
				target_inputs=target_inputs,
			)
		joint_attention_kwargs = self.preprocess_adapter(info_inputs=target_inputs, t_vec=t_vec_mid) \
			if need_adapt else None

		outputs_mid = self.transformer(
			hidden_states=img_mid,
			encoder_hidden_states=txt,
			pooled_projections=vec,
			timestep=t_vec_mid,
			img_ids=img_ids,
			txt_ids=txt_ids,
			guidance=guidance_vec,
			return_intermediates=return_intermediates,
			joint_attention_kwargs=joint_attention_kwargs,
			controlnet_block_samples=c_double_samples,
			controlnet_single_block_samples=c_single_samples,
			return_dict=False,
			controlnet_blocks_repeat=False,
		)
		velocity, intermediates_mid, _ = outputs_mid[0][0], outputs_mid[1], outputs_mid[2]
		img = img + (t_prev - t_curr) * velocity

		return img, velocity


