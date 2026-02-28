import torch
import yaml
from diffusers.utils import is_torch_version
from dashboard_utils import reclaim_memory
from diffusers.models.transformers.transformer_flux import *
with open(config_filepath := "./framework/config.yaml", 'r') as file:
    config = yaml.safe_load(file)


@maybe_allow_in_graph
class Override_FluxSingleTransformerBlock(FluxSingleTransformerBlock):
    r"""
        An extended version of FluxSingleTransformerBlock with additional features like KV operations
    and intermediate feature capture.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        mlp_ratio (`float`): The ratio for the MLP hidden dimension.
    """
    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0):
        super().__init__(dim, num_attention_heads, attention_head_dim, mlp_ratio)
        processor = FluxAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
        intermediates_carrier=None,
        block_id="SINGLE_?",
        return_intermediates=False,
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
        if return_intermediates and (f"{block_id}-0" in config["target_block_feature_ids_flux"]):
            intermediates_carrier[f"{block_id}-0"] = norm_hidden_states

        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)

        if return_intermediates and (f"{block_id}-1" in config["target_block_feature_ids_flux"]):
            intermediates_carrier[f"{block_id}-1"] = norm_hidden_states

        hidden_states = residual + hidden_states

        if return_intermediates and (f"{block_id}-2" in config["target_block_feature_ids_flux"]):
            intermediates_carrier[f"{block_id}-2"] = norm_hidden_states

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        del residual, norm_hidden_states, gate, mlp_hidden_states, joint_attention_kwargs, attn_output
        reclaim_memory()
        return hidden_states


@maybe_allow_in_graph
class Override_FluxTransformerBlock(FluxTransformerBlock):
    r"""
    An extended version of FluxTransformerBlock with KV operations and intermediate feature capture.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6):
        super().__init__(dim, num_attention_heads, attention_head_dim, qk_norm, eps)
        processor = FluxAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )


    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
        intermediates_carrier=None,
        block_id="DOUBLE_?",
        return_intermediates=False,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        if return_intermediates and (f"{block_id}-0" in config["target_block_feature_ids_flux"]):
            intermediates_carrier[f"{block_id}-0"] = norm_hidden_states

        joint_attention_kwargs = joint_attention_kwargs or {}
        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if return_intermediates and (f"{block_id}-1" in config["target_block_feature_ids_flux"]):
            intermediates_carrier[f"{block_id}-1"] = norm_hidden_states

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        # Process attention outputs for the `encoder_hidden_states`.
        if return_intermediates and (f"{block_id}-2" in config["target_block_feature_ids_flux"]):
            intermediates_carrier[f"{block_id}-2"] = hidden_states

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class Override_FluxTransformer2DModel(FluxTransformer2DModel):
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int] = (16, 56, 56),
    ):
        self.override_doubles = True
        self.override_singles = False

        super().__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            #out_channels=out_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            pooled_projection_dim=pooled_projection_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
        )

        if self.override_doubles:
            self.transformer_blocks = nn.ModuleList(
                [
                    Override_FluxTransformerBlock(
                        dim=self.inner_dim,
                        num_attention_heads=self.config.num_attention_heads,
                        attention_head_dim=self.config.attention_head_dim,
                    )
                    for i in range(self.config.num_layers)
                ]
            )

        if self.override_singles:
            self.single_transformer_blocks = nn.ModuleList(
                [
                    Override_FluxSingleTransformerBlock(
                        dim=self.inner_dim,
                        num_attention_heads=self.config.num_attention_heads,
                        attention_head_dim=self.config.attention_head_dim,
                    )
                    for i in range(self.config.num_single_layers)
                ]
            )

    def set_overrider(self, override_doubles, override_singles):
        self.override_doubles = override_doubles
        self.override_singles = override_singles


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
        return_intermediates: bool = False,
    ):
        """
        The [`Override_FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings from input conditions (e.g., prompts).
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`):
                Projected embeddings from input conditions.
            timestep (`torch.LongTensor`):
                Denoising step indicator.
            img_ids (`torch.Tensor`):
                Image positional IDs.
            txt_ids (`torch.Tensor`):
                Text positional IDs.
            guidance (`torch.Tensor`, *optional*):
                Guidance signal for conditional generation.
            joint_attention_kwargs (`dict`, *optional*):
                Kwargs for the `AttentionProcessor`.
            controlnet_block_samples (`list` of `torch.Tensor`, *optional*):
                Residuals to add to transformer blocks.
            controlnet_single_block_samples (`list` of `torch.Tensor`, *optional*):
                Residuals to add to single transformer blocks.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a `Transformer2DModelOutput`.
            controlnet_blocks_repeat (`bool`, *optional*, defaults to `False`):
                Whether to repeat controlnet residuals cyclically.
            return_intermediates (`bool`, *optional*, defaults to `False`):
                Whether to return intermediate features.

        Returns:
            Tuple containing:
                - `Transformer2DModelOutput`: Model output.
                - `intermediates_carrier` (`dict` or `None`): Intermediate features if `return_intermediates` is True.
                - None
        """
        intermediates_carrier = {}
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)
        if return_intermediates and "INIT" in config["target_block_feature_ids_flux"]:
            intermediates_carrier["INIT"] = hidden_states.clone()

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                if self.override_doubles:
                    encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        joint_attention_kwargs,
                        intermediates_carrier,
                        f"DOUBLE-{index_block}",
                        return_intermediates,
                        **ckpt_kwargs,
                    )
                else:
                    encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        joint_attention_kwargs,
                        **ckpt_kwargs,
                    )

            else:
                if self.override_doubles:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                        intermediates_carrier=intermediates_carrier,
                        block_id=f"DOUBLE-{index_block}",
                        return_intermediates=return_intermediates
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        if return_intermediates and "INTERMEDIATE" in config["target_block_feature_ids_flux"]:
            intermediates_carrier["INTERMEDIATE"] = hidden_states.clone()
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                if self.override_singles:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        temb,
                        image_rotary_emb,
                        joint_attention_kwargs,
                        intermediates_carrier,
                        f"SINGLE-{index_block}",
                        return_intermediates,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        temb,
                        image_rotary_emb,
                        joint_attention_kwargs,
                        **ckpt_kwargs,
                    )

            else:
                if self.override_singles:
                    hidden_states = block(
                        hidden_states=hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                        intermediates_carrier=intermediates_carrier,
                        block_id=f"SINGLE-{index_block}",
                        return_intermediates=return_intermediates,
                    )
                else:
                    hidden_states = block(
                        hidden_states=hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if return_intermediates and "FINAL" in config["target_block_feature_ids_flux"]:
            intermediates_carrier["FINAL"] = hidden_states.clone()

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_intermediates:
            intermediates_carrier = None
        return Transformer2DModelOutput(sample=output), intermediates_carrier, None