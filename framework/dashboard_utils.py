import copy
import csv
import json
import math
import os
import pickle
import cv2
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import numpy as np
import torch
import gc
from torch import nn
from PIL import Image, ImageDraw
from optimum.quanto import quantize, freeze, qfloat8, qint8, qint4, qint2
from einops import rearrange, repeat
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from autoencoder import AutoEncoder
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from masker_utils import AdaptiveMaskEstimator


class HFEmbedder(nn.Module):
	def __init__(self, version: str, max_length: int, is_clip, quant_dtype, **hf_kwargs):
		super().__init__()
		self.is_clip = is_clip
		self.max_length = max_length
		self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

		if self.is_clip:
			self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length)
			self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **hf_kwargs)
		else:
			self.tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained(version, subfolder="tokenizer_2")
			t5_encoder = T5EncoderModel.from_pretrained(version, subfolder="text_encoder_2")
			quantize(t5_encoder, weights=quant_dtype)
			freeze(t5_encoder)
			self.hf_module = t5_encoder
		self.hf_module = self.hf_module.eval().requires_grad_(False)

	def forward(self, text: list[str]) -> torch.Tensor:
		batch_encoding = self.tokenizer(
			text,
			truncation=True,
			max_length=self.max_length,
			return_length=False,
			return_overflowing_tokens=False,
			padding="max_length",
			return_tensors="pt",
		)

		outputs = self.hf_module(
			input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
			attention_mask=None,
			output_hidden_states=False,
		)
		return outputs[self.output_key]


def reclaim_memory():
	import gc
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


def _get_independent_regions(instruction, operation_region_path, device='cuda', dtype=torch.float32):
	operation_region = cv2.imread(operation_region_path, cv2.IMREAD_GRAYSCALE)
	_, binary_mask = cv2.threshold(operation_region, 1, 255, cv2.THRESH_BINARY)
	contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	region_info = []
	for i, contour in enumerate(contours):
		region_mask = np.zeros_like(operation_region)
		cv2.drawContours(region_mask, [contour], -1, 255, thickness=cv2.FILLED)
		if len(region_mask.shape) == 3:
			region_mask = region_mask[:, :, 0]
		region_tensor = torch.tensor(region_mask, device=device, dtype=dtype)
		region_info.append({
			"contour": contour,
			"tensor": region_tensor
		})

	for key in instruction["region_operations"].keys():
		pt = instruction["region_operations"][key]["centroids"][0]
		for region in region_info:
			if cv2.pointPolygonTest(region["contour"], pt, False) >= 0:
				region_tensor = region["tensor"]
				region_tensor = region_tensor.float() / 255.
				region_tensor[region_tensor > 0.0] = 1.0
				region_tensor = rearrange(region_tensor, "h w -> 1 1 h w")
				instruction["region_operations"][key]["region"] = region_tensor
				break
	return instruction


def _get_region_weights(instruction):
	total = 0.
	count = len(instruction["region_operations"])
	if count == 1:
		instruction["region_operations"]["0"]["raw_weight"] = torch.tensor(1.)
		instruction["region_operations"]["0"]["norm_weight"] = torch.tensor(1.)
		return instruction

	for op_id in instruction["region_operations"].keys():
		region = instruction["region_operations"][op_id]["region"]
		unmasked_pixels = torch.sum(region).item()
		total_pixels = region.numel()
		relative_size = unmasked_pixels / total_pixels if total_pixels != 0 else 0.0
		base_weight = 1.0 + 0.5 / (relative_size + 0.1)
		temp = torch.clamp(torch.tensor(base_weight, dtype=torch.float32), 1.0, 5.0)
		instruction["region_operations"][op_id]["raw_weight"] = temp
		total += temp

	for op_id in instruction["region_operations"].keys():
		instruction["region_operations"][op_id]["norm_weight"] = torch.tensor(1.0 / count) if total == 0 \
			else instruction["region_operations"][op_id]["raw_weight"] / total
	return instruction


def load_data(folder_path, output_path=None, device="cuda", dtype=torch.float32, debug_mode=False):
	print(f"\n>> Loading Sample Data...")
	raw_image_path = os.path.join(folder_path, 'original_image.png')
	operation_region_path = os.path.join(folder_path, 'operation.png')
	instruction_path = os.path.join(folder_path, 'instruction.json')

	raw_image = Image.open(raw_image_path).convert('RGB')
	print(f"\t> Original Image Loaded.")

	with open(instruction_path, 'r') as f:
		instruction = json.load(f)
	print(f"\t> Instruction Loaded.")

	instruction = _get_independent_regions(instruction=instruction, operation_region_path=operation_region_path)
	instruction = _get_region_weights(instruction)
	print(f"\t> Regions Loaded and Weighted.")

	instruction = AdaptiveMaskEstimator.create_adaptive_mask(
		operation_region_filepath=operation_region_path,
		instruction=instruction,
		output_path=output_path,
		device=device,
		dtype=dtype,
		debug_mode=debug_mode
	)
	print(f"\t> Adaptive Mask Created.")
	return raw_image, instruction


def solve_outcomes(x0_orig, x0_drag, x0_for_steps, raw_image, output_dir, save_inversion_img=False):
	raw_shape = (raw_image.size[1], raw_image.size[0])
	if x0_orig and save_inversion_img:
		img = x0_orig
		img = img.resize((raw_shape[1], raw_shape[0]), Image.LANCZOS)
		img.save(os.path.join(output_dir, f"original_image.png"))
		print(f"\t> Original Image Saved ({output_dir}).")
		del img, x0_orig

	if x0_drag:
		img = x0_drag
		img = img.resize((raw_shape[1], raw_shape[0]), Image.LANCZOS)
		img.save(os.path.join(output_dir, f"dragged_image.png"))
		print(f"\t> Dragged Image Saved ({output_dir}).")
		del img, x0_drag

	if x0_for_steps:
		os.makedirs(step_dir := os.path.join(output_dir, f"steps"))
		for step in x0_for_steps.keys():
			img = x0_for_steps.get(step)
			img = img.resize((raw_shape[1], raw_shape[0]), Image.LANCZOS)
			img.save(os.path.join(step_dir, f"{step}.png"))
		print(f"\t> Step Images Saved ({step_dir}).")
		del img, x0_for_steps
	del raw_image
	reclaim_memory()



def scale_coordinates(instruction, raw_shape, cut_shape, update_forward=True):
	for key in instruction["region_operations"].keys():
		begin_pt = instruction["region_operations"][key]["centroids"][0]
		target_pt = instruction["region_operations"][key]["centroids"][1]
		if update_forward:
			print("\nSCALE_COORDS:")
			print(f"[0]: {raw_shape[1] / cut_shape[1]}")
			print(f"[1]: {raw_shape[0] / cut_shape[0]}")

			begin_pt = torch.round(torch.tensor([
				begin_pt[0] / raw_shape[1] * cut_shape[1],
				begin_pt[1] / raw_shape[0] * cut_shape[0]
			]))
			target_pt = torch.round(torch.tensor([
				target_pt[0] / raw_shape[1] * cut_shape[1],
				target_pt[1] / raw_shape[0] * cut_shape[0]
			]))
			instruction["region_operations"][key]["points_fit"] = [
				copy.deepcopy(begin_pt),
				copy.deepcopy(begin_pt),
				copy.deepcopy(target_pt)
			]

			if anchor_pt := instruction["region_operations"][key].get("anchors"):
				anchor_pt = torch.round(torch.tensor([
					anchor_pt[0] / raw_shape[1] * cut_shape[1],
					anchor_pt[1] / raw_shape[0] * cut_shape[0]
				]))
				instruction["region_operations"][key]["anchors_fit"] = [copy.deepcopy(anchor_pt)]
				print(f'\t> Set {key} | Begin Centroid: {begin_pt.tolist()} | Target Centroid: {target_pt.tolist()} | Anchor: {anchor_pt.tolist()};')
			else:
				del instruction["region_operations"][key]["anchors"]
				print(f'\t> Set {key} | Begin Centroid: {begin_pt.tolist()} | Target Centroid: {target_pt.tolist()};')
		else:
			"""
			for point in points:
				point = point.tolist() if type(point) is not list else point
				original_point = torch.round(torch.tensor([
					point[1] * raw_shape[1] / cut_shape[1],
					point[0] * raw_shape[0] / cut_shape[0]
				]))
				handle_points.append(original_point)
			print(f'\t> Handle Points: {handle_points};')
			return handle_points
			"""
			pass
	return instruction


@torch.no_grad()
def _prepare_inputs_forFlux(
	T5,
	CLIP,
	img,
	prompt,
	c_image=None,
	c_txt=None,
	c_txt_ids=None,
	device="cuda:0",
	dtype=torch.float32,
	needs_img=True,
	needs_controls=False,
):
	inp = {}
	bs, c, h, w = img.shape
	if needs_img:
		if bs == 1 and not isinstance(prompt, str):
			bs = len(prompt)
		img = rearrange(
			img,
			pattern="b c (h ph) (w pw) -> b (h w) (c ph pw)",
			ph=2,
			pw=2
		)
		if img.shape[0] == 1 and bs > 1:
			img = repeat(img, "1 ... -> bs ...", bs=bs)
		inp["img"] = img.to(device, dtype)

		img_ids = torch.zeros(h // 2, w // 2, 3)
		img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
		img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
		img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
		inp["img_ids"] = img_ids.to(device, dtype)

	if isinstance(prompt, str):
		prompt = [prompt]
	txt = T5(prompt)
	if txt.shape[0] == 1 and bs > 1:
		txt = repeat(txt, "1 ... -> bs ...", bs=bs)
	txt_ids = torch.zeros(bs, txt.shape[1], 3)
	inp["txt"] = txt.to(device, dtype)
	inp["txt_ids"] = txt_ids.to(device, dtype)

	vec = CLIP(prompt)
	if vec.shape[0] == 1 and bs > 1:
		vec = repeat(vec, "1 ... -> bs ...", bs=bs)
	inp["vec"] = vec.to(device, dtype)

	if needs_controls:
		inp["c_img"] = c_image.to(device, dtype) if c_image is not None else None
		inp["c_txt"] = c_txt.to(device, dtype) if c_txt is not None else None
		inp["c_txt_ids"] = c_txt_ids.to(device, dtype) if c_txt_ids is not None else None
	return inp


@torch.no_grad()
def prepare_forFlux(
	repo_id,
	source_image,
	source_prompt,
	target_prompt,
	device,
	quant_dtype,
	dtype,
	needs_controls=False,
	control_image=None,
	control_prompt_embeds=None,
	control_text_ids=None,
):
	T5 = HFEmbedder(
		version=repo_id,
		max_length=256,
		is_clip=False,
		quant_dtype=quant_dtype,
		torch_dtype=dtype
	).to(device)
	CLIP = HFEmbedder(
		version="openai/clip-vit-large-patch14",
		max_length=77,
		is_clip=True,
		quant_dtype=quant_dtype,
		torch_dtype=dtype
	).to(device)

	inp_source = _prepare_inputs_forFlux(
		T5,
		CLIP,
		img=source_image,
		prompt=source_prompt,
		c_image=None,
		c_txt=None,
		c_txt_ids=None,
		device=device,
		dtype=torch.float32,
		needs_img=True,
		needs_controls=False
	)
	inp_target = _prepare_inputs_forFlux(
		T5,
		CLIP,
		img=source_image,
		prompt=target_prompt,
		c_image=control_image,
		c_txt=control_prompt_embeds,
		c_txt_ids=control_text_ids,
		device=device,
		dtype=torch.float32,
		needs_img=False,
		needs_controls=needs_controls
	)
	del T5, CLIP
	reclaim_memory()
	return inp_source, inp_target


def schedule(num_steps: int, image_seq_len: int, base_shift: float = 0.5, max_shift: float = 1.15, shift=True):
	def time_shift(mu: float, sigma: float, t: torch.Tensor):
		return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
	def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
		m = (y2 - y1) / (x2 - x1)
		b = y1 - m * x1
		return lambda x: m * x + b
	# extra step for zero
	timesteps = torch.linspace(1, 0, num_steps + 1)
	# shifting the schedule to favor high timesteps for higher signal images
	if shift:
		# estimate mu based on linear estimation between two points
		mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
		timesteps = time_shift(mu, 1.0, timesteps)
	return timesteps.tolist()


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
	if len(missing) > 0 and len(unexpected) > 0:
		print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
		print("\n" + "-" * 79 + "\n")
		print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
	elif len(missing) > 0:
		print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
	elif len(unexpected) > 0:
		print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


# Copied from diffusers.pipelines.controlnet_sd3.pipeline_stable_diffusion_3_controlnet.StableDiffusion3ControlNetPipeline.prepare_image
def _prepare_image(
	image,
	width,
	height,
	batch_size=1,
	num_images_per_prompt=1,
	vae_scale_factor=16,
	device="cuda",
	dtype=torch.float32,
	do_classifier_free_guidance=False,
):
	if isinstance(image, torch.Tensor):
		pass
	else:
		image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
		image = image_processor.preprocess(image, height=height, width=width)
	image_batch_size = image.shape[0]

	if image_batch_size == 1:
		repeat_by = batch_size
	else:
		# image batch size is the same as prompt batch size
		repeat_by = num_images_per_prompt

	image = image.repeat_interleave(repeat_by, dim=0)
	image = image.to(device=device, dtype=dtype)
	if do_classifier_free_guidance:
		image = torch.cat([image] * 2)
	return image


@torch.no_grad()
def encode(img, device="cuda", dtype=torch.bfloat16, do_preprocess=True):
	with torch.autocast(device_type=str(device), dtype=dtype):
		if do_preprocess and type(img) is Image:
			img = np.array(img)
			img = torch.from_numpy(img).float() / 127.5 - 1
			img = img.permute(2, 0, 1).unsqueeze(0).to(device)
		ckpt_path = hf_hub_download('black-forest-labs/FLUX.1-dev', "ae.safetensors")
		vae = AutoEncoder().to(device="meta")
		ckpt = load_file(ckpt_path, device=str(device))
		missing, unexpected = vae.load_state_dict(ckpt, strict=False, assign=True)
		print_load_warning(missing, unexpected)
		output = vae.encode(img)
		del vae
		reclaim_memory()
		return output


@torch.no_grad()
def decode(img, full_width, full_height, device="cuda", dtype=torch.bfloat16):
	with torch.autocast(device_type=str(device), dtype=dtype):
		ckpt_path = hf_hub_download('black-forest-labs/FLUX.1-dev', "ae.safetensors")
		vae = AutoEncoder().to(device="meta")
		ckpt = load_file(ckpt_path, device=str(device))
		missing, unexpected = vae.load_state_dict(ckpt, strict=False, assign=True)
		print_load_warning(missing, unexpected)

		batch_x = rearrange(
			img.float(),
			pattern="b (h w) (c ph pw) -> b c (h ph) (w pw)",
			h=math.ceil(full_height / 16),
			w=math.ceil(full_width / 16),
			ph=2,
			pw=2
		)
		x = batch_x[0].unsqueeze(0)
		x = vae.decode(x)
		x = x.clamp(-1, 1)
		x = rearrange(x[0], "c h w -> h w c")
		output = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

		del vae
		reclaim_memory()
		return output


def encode_for_calculation(zs, full_shape, device="cuda", dtype=torch.bfloat16):
	outputs = []
	for z in zs:
		bs, c, h, w = z.shape
		assert h % 2 == 0 and w % 2 == 0, "SizeError: height or width input cannot be divided by 2;"

		h = math.ceil(full_shape[0] / 16)
		w = math.ceil(full_shape[1] / 16)
		x = rearrange(
			z,
			pattern="b c (h ph) (w pw) -> b (h w) (c ph pw)",
			h=h,
			w=w,
			ph=2,
			pw=2
		)
		outputs.append(x)
	return outputs


def decode_for_calculation(zs, full_shape):
	outputs = []
	for z in zs:
		h = math.ceil(full_shape[0] / 16)
		w = math.ceil(full_shape[1] / 16)
		# full_shape=(384, 512), h=24, w=32, ph=2, pw=2, c=16
		# b (h w) (c ph pw) => 1 (24 32) (16 2 2) = (1, 768, 64)
		# b c (h ph) (w pw) => 1 16 (24 2) (32 2) = (1, 16, 48, 64)

		# b (h w) (c ph pw) => 1 (24 32) (16 ph pw) ->
		# b c (h ph) (w pw) => 1 16 (24 ph=8) (32 pw=8) -> (1, 768, 192, 256)
		x = rearrange(
			z.float(),
			pattern="b (h w) (c ph pw) -> b c (h ph) (w pw)",
			h=h,
			w=w,
			ph=2,
			pw=2
		)
		outputs.append(x)
		del z
		reclaim_memory()
	return outputs


def read_and_split_prompt(folder):
	file_path = os.path.join(folder, "prompt.txt")
	with open(file_path, "r", encoding="utf-8") as f:
		content = f.read().strip()
	paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
	if len(paragraphs) != 2:
		return ("", "", f"PrompyMissingError: {len(paragraphs)} paragraph of texts found, needs 2;")
	str1, str2 = paragraphs[0], paragraphs[1]
	str3 = paragraphs[0] + '' + paragraphs[1]
	return str1, str2, str3


def visualize_pca_feature(feature, save_path, log_path, comment=None, n_components=10):
	if isinstance(feature, torch.Tensor):
		feature = feature.detach().cpu().numpy()
	C, H, W = feature.shape[1], feature.shape[2], feature.shape[3]
	features_2d = feature.reshape(-1, C, H * W).squeeze(0).T  # (H*W, C)

	scaler = StandardScaler()
	features_scaled = scaler.fit_transform(features_2d)
	pca = PCA(n_components=min(n_components, 10))
	pca_result = pca.fit_transform(features_scaled)

	explained_variance_ratio = pca.explained_variance_ratio_
	cumulative_variance = np.sum(explained_variance_ratio)
	pca_normalized = np.zeros_like(pca_result)
	for i in range(pca_result.shape[1]):
		pca_normalized[:, i] = (pca_result[:, i] - pca_result[:, i].min()) / (pca_result[:, i].max() - pca_result[:, i].min() + 1e-8)

	fig_width = 5 * 3
	plt.figure(figsize=(fig_width, 5))
	for i in range(min(5, pca_result.shape[1])):
		plt.subplot(1, 7, i + 1)
		component = pca_normalized[:, i].reshape(H, W)
		plt.imshow(component, cmap='viridis')
		plt.title(f'PC{i + 1} ({explained_variance_ratio[i]:.1%})')
		plt.axis('off')

	if pca_result.shape[1] >= 3:
		plt.subplot(1, 7, 6)
		rgb_image = pca_normalized[:, :3].reshape(H, W, 3)
		plt.imshow(rgb_image)
		plt.title('RGB Composition')
		plt.axis('off')

	plt.subplot(1, 7, 7)
	weights = explained_variance_ratio / np.sum(explained_variance_ratio)
	aggregated = np.zeros((H * W,))
	for i in range(pca_result.shape[1]):
		aggregated += pca_normalized[:, i] * weights[i]
	aggregated = aggregated.reshape(H, W)

	plt.imshow(aggregated, cmap='magma')
	plt.title(f'Aggregated (Top 10\n{cumulative_variance:.1%})')
	plt.axis('off')
	plt.tight_layout()

	image_dir = os.path.dirname(save_path)
	os.makedirs(image_dir, exist_ok=True)
	plt.savefig(save_path, dpi=300, bbox_inches='tight')
	print(f"\t * Saved PCA-feature image to {save_path}.")

	def save_pca_log(csv_path, explained_variance_ratio, cumulative_variance, comment):
		ratios = list(explained_variance_ratio) + [0.0] * (10 - len(explained_variance_ratio))
		fieldnames = ['Comment'] + [f'PC{i + 1}' for i in range(10)] + ['Total']
		file_exists = os.path.exists(csv_path)
		with open(csv_path, 'a', newline='') as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
			if not file_exists:
				writer.writeheader()
			row_data = {
				'Comment': comment,
				**{f'PC{i + 1}': f'{ratios[i] * 100:.2f}%' for i in range(10)},
				'Total': f'{cumulative_variance * 100:.2f}%'
			}
			writer.writerow(row_data)

	save_pca_log(log_path, explained_variance_ratio, cumulative_variance, comment)
	print(f"\t * Saved PCA statistics to {log_path}.")
	plt.close()
	del feature, pca, pca_result, scaler, features_2d, features_scaled, pca_normalized
	reclaim_memory()
