

class KVHookHub:
    def __init__(
        self,
        mode="flux",
        for_keys=True,
        for_values=True,
        for_doubles=None,
        for_singles=None,
        for_ups=None,
        for_mid=None,
        for_downs=None,
        eta=0.0
    ):
        self.mode = mode
        self.eta = eta
        self.key_handlers = []
        self.value_handlers = []
        self.keys = {}
        self.values = {}
        self.timeidx = 0

        self.for_keys = for_keys
        self.for_values = for_values

        self.for_doubles = for_doubles
        self.for_singles = for_singles
        self.for_downs = for_downs
        self.for_mid = for_mid
        self.for_ups = for_ups

        self.do_operation__timesteps = None
        self.do_operation__blocks = None


    def _capture_key_hook(self, module, input, output, debug=False):
        module_id = id(module)
        if self.keys.get(module_id) is None:
            self.keys[module_id] = []
        self.keys[module_id].append(output.detach().clone().cpu())
        print(f"Key-in => ({module_id}-{len(self.keys[module_id])-1}) with {output.shape}") if debug else None


    def _inject_key_hook(self, module, input, output, debug=False):
        module_id = id(module)
        if self.do_operation__timesteps[self.timeidx] and self.keys.get(module_id):
            source_key = self.keys[module_id][self.timeidx].to(output.device)
            target_key = output.detach().clone()
            if source_key.shape != target_key.shape:
                raise ValueError(
                    f"KeyShapeMismatch: module {module_id} got {source_key.shape},"
                    f" where {output.shape} is expected."
                )
            # higher eta -> less source injection
            fused_key = self.eta * target_key + (1.0 - self.eta) * source_key
            output[:] = fused_key
            print(f"Key-out => ({module_id}-{self.timeidx}) with {output.shape}") if debug else None


    def _capture_value_hook(self, module, input, output, debug=False):
        module_id = id(module)
        if self.values.get(module_id) is None:
            self.values[module_id] = []
        self.values[module_id].append(output.detach().clone().cpu())
        print(f"Value-in => ({module_id}-{len(self.values[module_id])-1}) with {output.shape}") if debug else None


    def _inject_value_hook(self, module, input, output, debug=False):
        module_id = id(module)
        if self.do_operation__timesteps[self.timeidx] and self.values.get(module_id):
            source_value = self.values[module_id][self.timeidx].to(output.device)
            target_value = output.detach().clone()
            if source_value.shape != target_value.shape:
                raise ValueError(
                    f"KeyShapeMismatch: module {module_id} got {source_value.shape},"
                    f" where {output.shape} is expected."
                )
            # higher eta -> less source injection
            fused_value = self.eta * target_value + (1.0 - self.eta) * source_value
            output[:] = fused_value
            print(f"Value-out => ({module_id}-{self.timeidx}) with {output.shape}") if debug else None


    def register_dit_hooks(self, dit, do='CAPTURE'):
        if self.for_keys:
            key_hook = self._capture_key_hook if do == 'CAPTURE' else self._inject_key_hook
        if self.for_values:
            value_hook = self._capture_value_hook if do == 'CAPTURE' else self._inject_value_hook

        if self.for_doubles:
            do_operation__double = self.do_operation__blocks.get("double")
            assert len(do_operation__double) == len(dit.transformer_blocks), "\t> StepUnmatchError: do_operation__double length does not match the block number;"
            for d_block_idx, d_block in enumerate(dit.transformer_blocks):
                if do_operation__double[d_block_idx] and hasattr(d_block, 'attn'):
                    if self.for_keys:
                        self.key_handlers.append(d_block.attn.to_k.register_forward_hook(key_hook))
                    if self.for_values:
                        self.value_handlers.append(d_block.attn.to_v.register_forward_hook(value_hook))
        if self.for_singles:
            do_operation__single = self.do_operation__blocks.get("single")
            assert len(do_operation__single) == len(dit.single_transformer_blocks), "\t> StepUnmatchError: do_operation__single length does not match the block number;"
            for s_block_idx, s_block in enumerate(dit.single_transformer_blocks):
                if do_operation__single[s_block_idx] and hasattr(s_block, 'attn'):
                    if self.for_keys:
                        self.key_handlers.append(s_block.attn.to_k.register_forward_hook(key_hook))
                    if self.for_values:
                        self.value_handlers.append(s_block.attn.to_v.register_forward_hook(value_hook))


    def register_unet_hooks(self, unet, do='CAPTURE'):
        if self.for_keys:
            key_hook = self._capture_key_hook if do == 'CAPTURE' else self._inject_key_hook
        if self.for_values:
            value_hook = self._capture_value_hook if do == 'CAPTURE' else self._inject_value_hook

        if self.for_downs:
            do_operation__down = self.do_operation__blocks.get("down")
            assert len(do_operation__down) == len(unet.down_blocks), "StepUnmatchError: do_operation__down length does not match the block number;"
            for down_block_idx, down_block in enumerate(unet.down_blocks):
                if do_operation__down[down_block_idx] and hasattr(down_block, 'attentions'):
                    for attention in down_block.attentions:
                        for transformer_block in attention.transformer_blocks:
                            if self.for_keys:
                                self.key_handlers.append(transformer_block.attn1.to_k.register_forward_hook(key_hook))
                            if self.for_values:
                                self.value_handlers.append(transformer_block.attn1.to_v.register_forward_hook(value_hook))
        if self.for_mid:
            if self.do_operation__blocks.get("mid")[0] and hasattr(unet.mid_block, 'attentions'):
                for attention in unet.mid_block.attentions:
                    for transformer_block in attention.transformer_blocks:
                        if self.for_keys:
                            self.key_handlers.append(transformer_block.attn1.to_k.register_forward_hook(key_hook))
                        if self.for_values:
                            self.value_handlers.append(transformer_block.attn1.to_v.register_forward_hook(value_hook))
        if self.for_ups:
            do_operation__up = self.do_operation__blocks.get("up")
            assert len(do_operation__up) == len(unet.up_blocks), "StepUnmatchError: do_operation__up length does not match the block number;"
            for up_block_idx, up_block in enumerate(unet.up_blocks):
                if do_operation__up[up_block_idx] and hasattr(up_block, 'attentions'):
                    for attention in up_block.attentions:
                        for transformer_block in attention.transformer_blocks:
                            if self.for_keys:
                                self.key_handlers.append(transformer_block.attn1.to_k.register_forward_hook(key_hook))
                            if self.for_values:
                                self.value_handlers.append(transformer_block.attn1.to_v.register_forward_hook(value_hook))


    def unregister_hooks(self):
        if self.for_keys and self.key_handlers:
            for h in self.key_handlers:
                h.remove()
            self.key_handlers.clear()
        if self.for_values and self.value_handlers:
            for h in self.value_handlers:
                h.remove()
            self.value_handlers.clear()


    def countdown_hooks(self):
        if self.timeidx > 0:
            self.timeidx = self.timeidx - 1


    def countup_hooks(self):
        if self.timeidx < len(self.do_operation__timesteps-1):
            self.timeidx = self.timeidx + 1


    def set_operation_blocks(self, combo):

        def _get_bool_sequence(idx_list, max_value):
            bool_list = [False] * max_value
            for index in idx_list:
                if 0 <= index < max_value:
                    bool_list[index] = True
            return bool_list

        if self.mode == "flux":
            double_max = 19
            single_max = 38
            begin_from = None    # {double=0~18, single=0~37}  (ONLY FOR `hook_block_combo => 'begin-from'`)
            if combo == "non-ridge":
                do_operation = {
                    "double": [0, 7, 8, 9, 10, 18],
                    "single": [6, 9, 18, 23, 26, 31, 37]
                }
            elif combo == "position":
                do_operation = {
                    "double": [1, 2, 4],
                    "single": [7, 11, 35, 36]
                }
            elif combo == "begin-from":
                assert (begin_from is not None) and (len(begin_from) == 2), \
                    "KVHookBeginFromError: begin-from list must be provided in form [<DOUBLE-BLK-ID>, <SINGLE-BLK-ID>];"
                do_operation = {
                    "double": list(range(begin_from[0], double_max)),
                    "single": list(range(begin_from[1], single_max))
                }
            else:
                do_operation = {
                    "double": list(range(0, double_max)),
                    "single": list(range(0, single_max))
                }
            self.do_operation__blocks = {
                "double": _get_bool_sequence(idx_list=do_operation.get("double"), max_value=double_max),
                "single": _get_bool_sequence(idx_list=do_operation.get("single"), max_value=single_max)
            }
        # Not Used
        elif self.mode == "sd":
            down_max = 0
            mid_max = 1
            up_max = 0
            if (combo == "begin-from") and (begin_from is not None):
                do_operation = {
                    "down": list(range(begin_from[0], down_max)),
                    "mid": list(range(begin_from[1], mid_max)),
                    "up": list(range(begin_from[2], up_max)),
                }
            else:
                do_operation = {
                    "down": list(range(0, down_max)),
                    "mid": list(range(0, mid_max)),
                    "up": list(range(0, up_max)),
                }
            self.do_operation__blocks = {
                "down": _get_bool_sequence(idx_list=do_operation.get("down"), max_value=down_max),
                "mid": _get_bool_sequence(idx_list=do_operation.get("mid"), max_value=mid_max),
                "up": _get_bool_sequence(idx_list=do_operation.get("up"), max_value=up_max)
            }


    def set_operation_timesteps(self, combo, do_operation):
        assert len(do_operation) - 1 == self.timeidx, "StepUnmatchError: found len(do_operation) != self.timeidx;"
        do_operation = list(reversed(do_operation))
        if combo == "all":
            self.do_operation__timesteps = [True] * len(do_operation)
        elif combo.split(':')[0] == "except-last":
            self.do_operation__timesteps = [True] * len(do_operation)
            num_timestep_skipped = int(combo.split(':')[-1])
            self.do_operation__timesteps[0:num_timestep_skipped] = [False] * num_timestep_skipped
        elif combo == "after-drag":
            self.do_operation__timesteps = [True if (step is False) else False for step in do_operation]
        elif combo == "on-drag":
            self.do_operation__timesteps = do_operation


    def switch_hooks(self, unet_or_dit, do='INJECT'):
        self.unregister_hooks()
        self.timeidx = len(self.values[list(self.values.keys())[0]]) - 1 \
            if self.for_values else len(self.keys[list(self.keys.keys())[0]]) - 1
        if self.mode == "flux":
            self.register_dit_hooks(unet_or_dit, do)
        else:
            self.register_unet_hooks(unet_or_dit, do)


    def clear_hooks(self):
        self.unregister_hooks()
        self.key_handlers = []
        self.value_handlers = []
        self.keys = {}
        self.values = {}
        self.timeidx = 0
