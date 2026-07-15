"""
CoT compressor
Render CoT to image
"""

from operator import ne
from socket import AI_PASSIVE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import math
from typing import Optional, Tuple, Dict, Any, List, Union
from PIL import Image
import numpy as np
from transformers import DynamicCache

from .text_to_image import TextToImageRenderer
from .ocr_wrapper import OCRVisionEncoder
from .loss import StableLossFunctions


class SwiGLU(nn.Module):
    """
    SwiGLU
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)  # gate projection
        self.w2 = nn.Linear(input_dim, hidden_dim, bias=False)  # value projection
        self.w3 = nn.Linear(hidden_dim, input_dim, bias=False)  # output projection

    def forward(self, x):
        # x: (..., input_dim)
        gate = F.silu(self.w1(x))      # (..., hidden_dim)
        value = self.w2(x)              # (..., hidden_dim)
        hidden = gate * value           # element-wise product
        return self.w3(hidden)          # (..., input_dim)


class CoTCompressor(nn.Module):
    """
    CoTCompressor
    """

    def __init__(
        self,
        ocr_model_path: str = "path/to/xxx",
        llm_model_path: str = "path/to/xxx",
        image_size: int = 512,
        font_size: int = 16,
        device: str = "cuda",
        freeze_vision: bool = True,
        use_projection_head: bool = True,
        projection_hidden_dim: int = 2048,
        enable_lora: bool = True,
        use_lora: Optional[bool] = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        use_uncertainty_weighting: bool = True,
        vision_loss_weight: float = 1.0,
        lm_loss_weight: float = 1.0,
        use_custom_llm: bool = True,
        loss_type: str = "mse_only",
        # Stage II
        stage2_mode: bool = False,
        train_lm_head_only: bool = False,
        freeze_projection_head: bool = False,
        include_img_end_loss: bool = False,
        include_vision_loss: bool = True,
    ):
        super().__init__()

        self.device = device
        self.image_size = image_size
        self.use_projection_head = use_projection_head
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.vision_loss_weight = vision_loss_weight
        self.lm_loss_weight = lm_loss_weight
        
        # Stage II config
        self.stage2_mode = stage2_mode
        self.train_lm_head_only = train_lm_head_only
        self.freeze_projection_head = freeze_projection_head
        self.include_img_end_loss = include_img_end_loss
        self.include_vision_loss = include_vision_loss
        
        # Loss type
        self.loss_type = getattr(self, 'loss_type', 'mse_only')
        self.loss_functions = StableLossFunctions(self.loss_type)

        # Text Renderer
        self.text_renderer = TextToImageRenderer(
            image_size=image_size,
            font_size=font_size,
        )

        # Vision Encoder
        self.vision_encoder = OCRVisionEncoder(
            ocr_model_path=ocr_model_path,
            llm_model_path=llm_model_path,
            device=device,
            freeze_vision=freeze_vision,
            use_custom_llm=use_custom_llm,
        )

        self.llm_loss_function = self.vision_encoder.mllm_model.loss_function
        self.llm_lm_head = self.vision_encoder.mllm_model.lm_head

        # tokenizer & language_model
        self.tokenizer = self.vision_encoder.get_tokenizer()
        self.language_model = self.vision_encoder.get_language_model()

        # hidden dimension
        self.hidden_dim = self.language_model.config.get_text_config().hidden_size

        # projection head
        if use_projection_head:
            self.projection_head = nn.Sequential(
                nn.LayerNorm(self.hidden_dim, eps=1e-6),
                nn.Linear(self.hidden_dim, projection_hidden_dim),
                SwiGLU(input_dim=projection_hidden_dim, hidden_dim=projection_hidden_dim),
                nn.Linear(projection_hidden_dim, self.hidden_dim),
            )
            self.projection_head = self.projection_head.to(dtype=self.language_model.dtype)
        else:
            self.projection_head = None
        
        if use_uncertainty_weighting:
            self.log_var_vision = nn.Parameter(torch.zeros(1), requires_grad=True)  # log(σ_vision²)
            self.log_var_lm = nn.Parameter(torch.zeros(1), requires_grad=True)      # log(σ_lm²)
        
        # <img_begin> <img_end> token
        new_tokens = []
        if "<img_begin>" not in self.tokenizer.vocab:
            new_tokens.append("<img_begin>")
        if "<img>" not in self.tokenizer.vocab:
            new_tokens.append("<img>")
        if "<img_end>" not in self.tokenizer.vocab:
            new_tokens.append("<img_end>")
        
        if new_tokens:
            self.tokenizer.add_tokens(new_tokens)
            if self.language_model:
                self.language_model.resize_token_embeddings(len(self.tokenizer))
                self.llm_lm_head = self.vision_encoder.mllm_model.lm_head
        
        self.img_begin_token_id = self.tokenizer.vocab["<img_begin>"]
        self.img_token_id = self.tokenizer.vocab["<img>"] 
        self.img_end_token_id = self.tokenizer.vocab["<img_end>"]
        
        self._init_special_token_embeddings()
    
    def _init_special_token_embeddings(self):
        """
        """
        target_norm_scale = 1.0

        model_dtype = self.language_model.dtype
        try:
            model_device = next(self.language_model.parameters()).device
        except:
            model_device = torch.device(self.device) if isinstance(self.device, str) else self.device
        
        embed_table = self.language_model.get_input_embeddings()
        
        # 1. <img_begin>
        with torch.no_grad():
            v_begin = torch.randn(self.hidden_dim, dtype=model_dtype, device=model_device)
            v_begin = v_begin / (v_begin.norm() + 1e-6)
            v_begin = v_begin * (target_norm_scale * math.sqrt(self.hidden_dim))
            embed_table.weight[self.img_begin_token_id] = v_begin
        
        # 2. <img>
        with torch.no_grad():
            v_img = torch.randn(self.hidden_dim, dtype=model_dtype, device=model_device)
            v_img = v_img / (v_img.norm() + 1e-6)
            v_img = v_img * (target_norm_scale * math.sqrt(self.hidden_dim))
            embed_table.weight[self.img_token_id] = v_img
        
        # 3. <img_end>
        with torch.no_grad():
            v_end = torch.randn(self.hidden_dim, dtype=model_dtype, device=model_device)
            v_end = v_end / (v_end.norm() + 1e-6)
            v_end = v_end * (target_norm_scale * math.sqrt(self.hidden_dim))
            embed_table.weight[self.img_end_token_id] = v_end
    
    def compress_cot(self, cot_text: str, return_image: bool = False) -> Tuple[torch.Tensor, Optional[Image.Image]]:
        """
        compress cot
        """
        image = self.text_renderer.render(cot_text)

        compressed = self.vision_encoder.encode_image(image, self.image_size)
        if return_image:
            return compressed, image
        return compressed, None

    def forward(
        self,
        question_texts,  # Union[str, List[str]]
        cot_texts,  # Union[str, List[str]]
        answer_texts=None,  # Optional[Union[str, List[str]]]
        return_loss: bool = True,
    ) -> Dict[str, Any]:
        """
        Args:
            question_texts
            cot_texts
            answer_texts
            return_loss

        Returns:
            outputs: loss, vision_loss, lm_loss etc.
        """
        if isinstance(question_texts, str):
            question_texts = [question_texts]
            cot_texts = [cot_texts]
            if answer_texts is not None:
                answer_texts = [answer_texts]

        batch_size = len(question_texts)
        
        try:
            model_device = next(self.language_model.parameters()).device
        except:
            model_device = torch.device(self.device) if isinstance(self.device, str) else self.device

        target_vision_embeds_list = []
        num_vision_tokens_list = []
        for cot_text in cot_texts:
            with torch.no_grad():  
                target_vision_emb, _ = self.compress_cot(cot_text)
                if target_vision_emb.dim() == 2:
                    target_vision_emb = target_vision_emb.unsqueeze(0)
                target_vision_embeds_list.append(target_vision_emb.detach().clone())
                num_vision_tokens_list.append(target_vision_emb.shape[1])

        embed_tokens = self.language_model.get_input_embeddings()
        
        all_predicted_vision_embeds = [] 
        all_vision_losses = []  
        all_question_embeds = []  
        
        for sample_idx in range(batch_size):
            question_text = question_texts[sample_idx]
            target_vision_embeds = target_vision_embeds_list[sample_idx]  # [1, num_tokens, hidden_dim]
            num_vision_tokens = num_vision_tokens_list[sample_idx]
            
            question_message = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."},],},
                {"role": "user", "content": [{"type": "text", "text": question_text},],},
            ] 
            question_text = self.tokenizer.apply_chat_template(question_message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            question_encoding = self.tokenizer([question_text], return_tensors="pt").to(model_device)
            question_ids = question_encoding["input_ids"]  # [1, q_len]
            question_embeds = embed_tokens(question_ids)  # [1, q_len, hidden_dim]
            all_question_embeds.append(question_embeds)
            
            img_begin_ids = torch.full((1, 1), self.img_begin_token_id, dtype=torch.long, device=model_device)
            img_begin_embeds = embed_tokens(img_begin_ids)  # [1, 1, hidden_dim]
            
            predicted_vision_embeds_list = []
            sample_vision_losses = []
            current_embeds = torch.cat([question_embeds, img_begin_embeds], dim=1)  # [1, q_len+1, hidden_dim]
            
            kv_cache = None
            
            for i in range(num_vision_tokens):
                if kv_cache is None:
                    lm_outputs = self.language_model(
                        inputs_embeds=current_embeds,
                        output_hidden_states=True,
                        use_cache=True,
                    )
                else:
                    cache_obj = DynamicCache()
                    for layer_idx, layer in enumerate(kv_cache.layers):
                        k = layer.keys
                        v = layer.values
                        cache_obj.update(k, v, layer_idx)
                    
                    lm_outputs = self.language_model(
                        inputs_embeds=current_embeds[:, -1:, :], 
                        past_key_values=cache_obj,
                        output_hidden_states=True,
                        use_cache=True,
                    )

                kv_cache = lm_outputs.past_key_values

                last_hidden = lm_outputs.hidden_states[-1][:, -1:, :]  # [1, 1, hidden_dim]

                if self.projection_head is not None:
                    predicted_vision_emb = self.projection_head(last_hidden)  # [1, 1, hidden_dim]
                else:
                    predicted_vision_emb = last_hidden

                target_vision_emb = target_vision_embeds[:, i:i+1, :]  # [1, 1, hidden_dim]
                step_loss = self.loss_functions.compute_vision_loss(predicted_vision_emb, target_vision_emb)
                sample_vision_losses.append(step_loss)

                predicted_vision_embeds_list.append(predicted_vision_emb)

                current_embeds = torch.cat([current_embeds, predicted_vision_emb.detach()], dim=1)

                del lm_outputs, last_hidden, target_vision_emb

            sample_predicted_vision_embeds = torch.cat(predicted_vision_embeds_list, dim=1)  # [1, num_tokens, hidden_dim]
            all_predicted_vision_embeds.append(sample_predicted_vision_embeds)

            if self.include_vision_loss and len(sample_vision_losses) > 0:
                sample_vision_loss = torch.stack(sample_vision_losses).mean()
                all_vision_losses.append(sample_vision_loss)

            del predicted_vision_embeds_list, sample_vision_losses, current_embeds, target_vision_embeds, kv_cache

        if not self.include_vision_loss or len(all_vision_losses) == 0:
            vision_loss = None
        else:
            vision_loss = torch.stack(all_vision_losses).mean()

        del target_vision_embeds_list, all_vision_losses

        lm_loss = None
        
        if answer_texts is not None and return_loss:
            all_lm_losses = []
            
            for sample_idx in range(batch_size):
                question_embeds = all_question_embeds[sample_idx]  # [1, q_len, hidden_dim]
                predicted_vision_embeds = all_predicted_vision_embeds[sample_idx]  # [1, num_tokens, hidden_dim]
                answer_text = answer_texts[sample_idx]
                num_vision_tokens = num_vision_tokens_list[sample_idx]
                
                img_begin_ids = torch.full((1, 1), self.img_begin_token_id, dtype=torch.long, device=model_device)
                img_begin_embeds = embed_tokens(img_begin_ids)  # [1, 1, hidden_dim]
                
                img_end_ids = torch.full((1, 1), self.img_end_token_id, dtype=torch.long, device=model_device)
                img_end_embeds = embed_tokens(img_end_ids)  # [1, 1, hidden_dim]
                
                # answer embedding
                answer_message = [
                    {"role": "assistant", "content": [{"type": "text", "text": answer_text},],},
                ] 
                answer_text = self.tokenizer.apply_chat_template(answer_message, tokenize=False, add_generation_prompt=False, enable_thinking=False)
                answer_encoding = self.tokenizer([answer_text], return_tensors="pt").to(model_device)
                answer_ids = answer_encoding["input_ids"]  # [1, a_len]
                answer_embeds = embed_tokens(answer_ids)  # [1, a_len, hidden_dim]
                
                # [question] + [<img_begin>] + [vision] + [<img_end>] + [answer]
                full_embeds = torch.cat([
                    question_embeds, 
                    img_begin_embeds, 
                    predicted_vision_embeds, 
                    img_end_embeds, 
                    answer_embeds
                ], dim=1)
                
                q_len = question_embeds.shape[1]
                question_labels = torch.full((1, q_len), -100, dtype=torch.long, device=model_device)
                
                img_begin_labels = torch.full((1, 1), -100, dtype=torch.long, device=model_device)
                
                vision_labels = torch.full((1, num_vision_tokens), -100, dtype=torch.long, device=model_device)
                
                img_end_labels = img_end_ids  
                
                labels = torch.cat([
                    question_labels, 
                    img_begin_labels, 
                    vision_labels, 
                    img_end_labels, 
                    answer_ids
                ], dim=1)

                lm_outputs = self.language_model(
                    inputs_embeds=full_embeds,
                    labels=labels,
                    output_hidden_states=True,
                )
                
                # For Qwen3
                hidden_states = lm_outputs.last_hidden_state
                logits_to_keep = 0
                slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
                logits = self.llm_lm_head(hidden_states[:, slice_indices, :])
                sample_lm_loss = self.llm_loss_function(
                    logits=logits, 
                    labels=labels, 
                    vocab_size=self.language_model.config.get_text_config().vocab_size
                )
                all_lm_losses.append(sample_lm_loss)
            
            lm_loss = torch.stack(all_lm_losses).mean()
            del all_lm_losses
        
        total_loss = None
        if return_loss:
            if self.use_uncertainty_weighting:
                total_loss, loss_info = self.loss_functions.compute_weighted_loss(
                    vision_loss, 
                    lm_loss, 
                    self.vision_loss_weight, 
                    self.lm_loss_weight, 
                    self.use_uncertainty_weighting, 
                    self.log_var_vision, 
                    self.log_var_lm
                )
            else:
                if vision_loss is not None and lm_loss is not None:
                    total_loss = self.vision_loss_weight * vision_loss + self.lm_loss_weight * lm_loss
                elif vision_loss is not None:
                    total_loss = self.vision_loss_weight * vision_loss
                elif lm_loss is not None:
                    total_loss = self.lm_loss_weight * lm_loss
        
        result = {
            "loss": total_loss,
            "vision_loss": vision_loss.detach() if vision_loss is not None else None,
            "lm_loss": lm_loss.detach() if lm_loss is not None else None,
            "predicted_vision_embeds": None,
            "target_vision_embeds": None,
            "num_vision_tokens": num_vision_tokens_list,  
        }
        del all_question_embeds, all_predicted_vision_embeds
        del embed_tokens, num_vision_tokens_list

        return result

    def generate(
        self,
        question_text: str,
        cot_text: Optional[str] = None,  # 仅用于 debug/对比（可选）
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        verbose: bool = False,
        # 自适应停止参数
        max_vision_tokens: int = 1024,  # 防止无限生成的上限
        stop_threshold: float = 0.02,   # 与 img_end_emb 的距离阈值
        return_latents: bool = False,  # 是否返回latent vectors
        replace_latent_vectors: Optional[Dict[int, np.ndarray]] = None,
        question_idx: Optional[int] = None,
        **generation_kwargs,
    ) -> Union[str, Tuple[str, List[torch.Tensor]]]:
        """
        Args:
            question_text
            cot_text
            max_new_tokens
            temperature
            verbose
            max_vision_tokens
            stop_threshold
            return_latents: 是否返回latent vectors

        Returns:
            answer_text or (answer_text, latent_vectors) if return_latents=True
        """
        cot_text = None
        with torch.no_grad():
            expected_vision_tokens = None
            if cot_text is not None and verbose:
                try:
                    target_embeds, _ = self.compress_cot(cot_text)
                    expected_vision_tokens = target_embeds.shape[0]
                except:
                    pass
            
            try:
                model_device = next(self.language_model.parameters()).device
            except:
                model_device = torch.device(self.device) if isinstance(self.device, str) else self.device
            
            embed_tokens = self.language_model.get_input_embeddings()
            
            question_message = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question_text},
            ] 
            question_text = self.tokenizer.apply_chat_template(question_message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            question_encoding = self.tokenizer([question_text], return_tensors="pt").to(model_device)
            question_ids = question_encoding["input_ids"]  # [1, q_len]
            question_embeds = embed_tokens(question_ids)  # [1, q_len, hidden_dim]
            
            img_begin_ids = torch.full((1, 1), self.img_begin_token_id, dtype=torch.long, device=model_device)
            img_begin_embeds = embed_tokens(img_begin_ids)  # [1, 1, hidden_dim]
            
            if cot_text is not None:
                try:
                    target_vision_embeds, _ = self.compress_cot(cot_text)
                    num_vision_tokens = target_vision_embeds.shape[0]
                except:
                    num_vision_tokens = max_vision_tokens
            else:
                num_vision_tokens = max_vision_tokens
            
            predicted_vision_embeds_list = []
            last_hidden_list = []  # 保存每次迭代的 last_hidden
            current_embeds = torch.cat([question_embeds, img_begin_embeds], dim=1)  # [1, q_len+1, hidden_dim]

            kv_cache = None

            for i in range(num_vision_tokens):
                if kv_cache is None:
                    lm_outputs = self.language_model(
                        inputs_embeds=current_embeds,
                        output_hidden_states=True,
                        use_cache=True,
                    )
                else:
                    cache_obj = DynamicCache()
                    for layer_idx, layer in enumerate(kv_cache.layers):
                        k = layer.keys
                        v = layer.values
                        cache_obj.update(k, v, layer_idx)

                    lm_outputs = self.language_model(
                        inputs_embeds=current_embeds[:, -1:, :],
                        past_key_values=cache_obj,
                        output_hidden_states=True,
                        use_cache=True,
                    )

                kv_cache = lm_outputs.past_key_values

                last_hidden = lm_outputs.hidden_states[-1][:, -1:, :]  # [1, 1, hidden_dim]
                last_hidden_list.append(last_hidden.clone())  # 保存原始 last_hidden

                last_hidden_real = last_hidden
                # 替换推理向量 (如果启用)
                if replace_latent_vectors is not None and question_idx is not None:
                    if question_idx in replace_latent_vectors:
                        vec_idx = i
                        if vec_idx < replace_latent_vectors[question_idx].shape[0]:
                            last_hidden_real = torch.from_numpy(replace_latent_vectors[question_idx][vec_idx]).to(last_hidden_real.device).to(last_hidden_real.dtype).unsqueeze(0)

                if self.projection_head is not None:
                    predicted_vision_emb = self.projection_head(last_hidden_real)  # [1, 1, hidden_dim]
                else:
                    predicted_vision_emb = last_hidden_real

                predicted_vision_embeds_list.append(predicted_vision_emb)

                current_embeds = torch.cat([current_embeds, predicted_vision_emb.detach()], dim=1)

                del lm_outputs, last_hidden

            actual_vision_tokens = len(predicted_vision_embeds_list)

            if return_latents and actual_vision_tokens > 0:
                # 返回每次迭代的 last_hidden 作为 latent vectors
                latent_vectors = [h.squeeze(0) for h in last_hidden_list]

            if actual_vision_tokens > 0:
                predicted_vision_embeds = torch.cat(predicted_vision_embeds_list, dim=1)  # [1, num_tokens, hidden_dim]
            else:
                predicted_vision_embeds = torch.empty((1, 0, self.hidden_dim), device=model_device)

            del predicted_vision_embeds_list, last_hidden_list, kv_cache

            img_end_ids = torch.full((1, 1), self.img_end_token_id, dtype=torch.long, device=model_device)
            img_end_embeds = embed_tokens(img_end_ids)  # [1, 1, hidden_dim]
            
            full_embeds = torch.cat([
                question_embeds, 
                img_begin_embeds, 
                predicted_vision_embeds, 
                img_end_embeds
            ], dim=1)
                     
            answer_ids = []
            eos_id = self.tokenizer.eos_token_id
            current_seq = full_embeds
            
            for step in range(max_new_tokens):
                lm_out = self.language_model(
                    inputs_embeds=current_seq,
                    output_hidden_states=True,
                )
                
                last_hidden = lm_out.last_hidden_state[:, -1:, :]  # [1, 1, hidden_dim]
                logits = self.llm_lm_head(last_hidden)  # [1, 1, vocab_size]
                logits = logits.squeeze(1)  # [1, vocab_size]
                
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [1, 1]
                
                next_token_id = next_token.item()
                
                if next_token_id == eos_id:
                    break
                
                answer_ids.append(next_token_id)
                
                next_emb = embed_tokens(next_token)  # [1, 1, hidden_dim]
                current_seq = torch.cat([current_seq, next_emb], dim=1)
                
                if verbose and (step + 1) % 20 == 0:
                    print(f"  已生成 {step + 1} tokens...")
            
            if len(answer_ids) > 0:
                answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True)
            else:
                answer = ""

            if return_latents and actual_vision_tokens > 0:
                return answer, latent_vectors
            else:
                return answer


    def compute_compression_stats(self, cot_text: str) -> Dict[str, Any]:
        original_tokens = len(self.tokenizer.encode(cot_text))

        compressed, _ = self.compress_cot(cot_text)
        compressed_tokens = compressed.shape[0]

        compression_ratio = original_tokens / compressed_tokens

        return {
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": compression_ratio,
            "saved_tokens": original_tokens - compressed_tokens,
            "saved_percentage": (1 - compressed_tokens / original_tokens) * 100,
        }


    def generate_with_visualization(
        self,
        question_text: str,
        cot_text: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        verbose: bool = False,
        max_vision_tokens: int = 1024,
        stop_threshold: float = 0.02,
        save_visualizations: bool = True,
        output_dir: str = "./visualizations",
        sample_id: str = "sample",
        **generation_kwargs,
    ) -> Dict[str, Any]:
        """
        推理生成函数 - 带完整可视化功能
        
        与原始generate函数的区别：
        1. 保存CoT渲染后的图像（带尺寸标注）
        2. 保存推理过程中生成的所有vision embeddings
        3. 对embeddings进行多维度可视化
        4. 返回完整的生成信息和可视化路径
        
        Args:
            question_text: 问题文本
            cot_text: CoT 文本（可选，用于对比）
            max_new_tokens: 答案最大 token 数
            temperature: 生成温度
            verbose: 是否显示详细信息
            max_vision_tokens: vision tokens 数量上限
            stop_threshold: 停止阈值
            save_visualizations: 是否保存可视化结果
            output_dir: 可视化输出目录
            sample_id: 样本ID（用于文件命名）
            
        Returns:
            包含以下内容的字典：
            - answer: 生成的答案文本
            - vision_embeddings: 生成的vision embeddings [num_tokens, hidden_dim]
            - num_vision_tokens: vision token数量
            - cot_image_path: CoT渲染图像路径（如果保存）
            - visualization_paths: 各种可视化结果的路径字典
        """
        import os
        from pathlib import Path
        
        if save_visualizations:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        with torch.no_grad():
            # 1. 如果提供了cot_text，渲染并保存图像（不标注尺寸）
            cot_image_path = None
            cot_image_width = None
            cot_image_height = None
            
            if cot_text is not None and save_visualizations:
                # 渲染CoT文本为图像
                cot_image = self.text_renderer.render(cot_text)
                cot_image_width, cot_image_height = cot_image.size
                
                # 保存图像（不添加标注）
                cot_image_path = output_path / f"{sample_id}_cot_rendered.png"
                cot_image.save(cot_image_path)
                
                if verbose:
                    print(f"✓ Saved CoT rendered image: {cot_image_path}")
                    print(f"  Image size: {cot_image_width}x{cot_image_height} pixels")
            
            # 2. 生成答案并记录vision embeddings
            if verbose:
                print(f"\n{'='*60}")
                print(f"🎯 生成 Vision Embeddings（带完整记录）")
                print(f"{'='*60}")
            
            # 获取模型设备
            try:
                model_device = next(self.language_model.parameters()).device
            except:
                model_device = torch.device(self.device) if isinstance(self.device, str) else self.device
            
            embed_tokens = self.language_model.get_input_embeddings()
            
            # 编码问题
            question_message = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question_text},
            ]
            question_text_formatted = self.tokenizer.apply_chat_template(
                question_message, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            question_encoding = self.tokenizer([question_text_formatted], return_tensors="pt").to(model_device)
            question_ids = question_encoding["input_ids"]
            question_embeds = embed_tokens(question_ids)
            
            # 添加 <img_begin> token
            img_begin_ids = torch.full((1, 1), self.img_begin_token_id, dtype=torch.long, device=model_device)
            img_begin_embeds = embed_tokens(img_begin_ids)
            
            # 确定 vision token 数量（使用固定数量，不再通过compress_cot计算）
            num_vision_tokens = max_vision_tokens
            if verbose:
                print(f"  使用固定 vision tokens 数量: {num_vision_tokens}")
            
            # 自回归生成 vision embeddings（记录所有中间结果）
            predicted_vision_embeds_list = []
            current_embeds = torch.cat([question_embeds, img_begin_embeds], dim=1)
            
            kv_cache = None
            
            for i in range(num_vision_tokens):
                # 通过 LM 获取 hidden states
                if kv_cache is None:
                    lm_outputs = self.language_model(
                        inputs_embeds=current_embeds,
                        output_hidden_states=True,
                        use_cache=True,
                    )
                else:
                    # 使用 KV cache
                    cache_obj = DynamicCache()
                    for layer_idx, layer in enumerate(kv_cache.layers):
                        k = layer.keys
                        v = layer.values
                        cache_obj.update(k, v, layer_idx)
                    
                    lm_outputs = self.language_model(
                        inputs_embeds=current_embeds[:, -1:, :],
                        past_key_values=cache_obj,
                        output_hidden_states=True,
                        use_cache=True,
                    )
                
                kv_cache = lm_outputs.past_key_values
                last_hidden = lm_outputs.hidden_states[-1][:, -1:, :]
                
                # 通过 projection_head 预测
                if self.projection_head is not None:
                    predicted_vision_emb = self.projection_head(last_hidden)
                else:
                    predicted_vision_emb = last_hidden
                
                predicted_vision_embeds_list.append(predicted_vision_emb)
                current_embeds = torch.cat([current_embeds, predicted_vision_emb.detach()], dim=1)
                
                del lm_outputs, last_hidden
                
                if verbose and (i + 1) % 50 == 0:
                    print(f"  已生成 {i + 1}/{num_vision_tokens} vision tokens")
            
            actual_vision_tokens = len(predicted_vision_embeds_list)
            
            # 合并所有 vision embeddings
            if actual_vision_tokens > 0:
                predicted_vision_embeds = torch.cat(predicted_vision_embeds_list, dim=1)  # [1, num_tokens, hidden_dim]
            else:
                predicted_vision_embeds = torch.empty((1, 0, self.hidden_dim), device=model_device)
            
            del predicted_vision_embeds_list, kv_cache
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"✓ Vision Tokens 生成完成")
                print(f"{'='*60}")
                print(f"生成数量: {actual_vision_tokens}")
            
            # 保存vision embeddings到numpy数组（用于后续可视化）
            # 修复BFloat16转换问题：先转换为float32再转numpy
            vision_embeddings_numpy = predicted_vision_embeds.squeeze(0).float().cpu().numpy()  # [num_tokens, hidden_dim]
            
            # 3. 拼接完整输入序列并生成答案
            img_end_ids = torch.full((1, 1), self.img_end_token_id, dtype=torch.long, device=model_device)
            img_end_embeds = embed_tokens(img_end_ids)
            
            full_embeds = torch.cat([
                question_embeds,
                img_begin_embeds,
                predicted_vision_embeds,
                img_end_embeds
            ], dim=1)
            
            # 自回归生成答案
            if verbose:
                print(f"\n{'='*60}")
                print(f"📝 生成答案 (max_new_tokens={max_new_tokens})")
                print(f"{'='*60}")
            
            answer_ids = []
            eos_id = self.tokenizer.eos_token_id
            current_seq = full_embeds
            
            for step in range(max_new_tokens):
                lm_out = self.language_model(
                    inputs_embeds=current_seq,
                    output_hidden_states=True,
                )
                
                last_hidden = lm_out.last_hidden_state[:, -1:, :]
                logits = self.llm_lm_head(last_hidden)
                logits = logits.squeeze(1)
                
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                next_token_id = next_token.item()
                
                if next_token_id == eos_id:
                    break
                
                answer_ids.append(next_token_id)
                next_emb = embed_tokens(next_token)
                current_seq = torch.cat([current_seq, next_emb], dim=1)
                
                if verbose and (step + 1) % 20 == 0:
                    print(f"  已生成 {step + 1} tokens...")
            
            # 解码答案
            if len(answer_ids) > 0:
                answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True)
            else:
                answer = ""
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"✓ 生成完成")
                print(f"{'='*60}")
                preview = answer[:100] + ('...' if len(answer) > 100 else '')
                print(f"答案 ({len(answer_ids)} tokens): {preview}")
                print(f"{'='*60}\n")
            
            # 4. 生成embedding可视化（如果需要）
            visualization_paths = {}
            
            if save_visualizations and actual_vision_tokens > 0:
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"📊 生成 Embedding 可视化")
                    print(f"{'='*60}")
                
                # 将可视化功能委托给外部函数（避免在模型代码中添加过多依赖）
                try:
                    visualization_paths = self._create_embedding_visualizations(
                        vision_embeddings_numpy,
                        output_path,
                        sample_id,
                        verbose
                    )
                except Exception as e:
                    # 打印异常信息（即使verbose=False也打印，因为这是重要错误）
                    print(f"⚠️  Warning: Failed to create visualizations: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 5. 返回完整结果
            result = {
                "answer": answer,
                "vision_embeddings": vision_embeddings_numpy,
                "num_vision_tokens": actual_vision_tokens,
                "cot_image_path": str(cot_image_path) if cot_image_path else None,
                "cot_image_width": cot_image_width,
                "cot_image_height": cot_image_height,
                "visualization_paths": visualization_paths,
            }
            
            return result
    
    def _create_embedding_visualizations(
        self,
        embeddings,  # np.ndarray: [num_tokens, hidden_dim]
        output_dir,  # Path
        sample_id: str,
        verbose: bool = False
    ) -> Dict[str, str]:
        """
        创建embedding的多种可视化
        
        Args:
            embeddings: vision embeddings数组 [num_tokens, hidden_dim]
            output_dir: 输出目录
            sample_id: 样本ID
            verbose: 是否显示详细信息
            
        Returns:
            各种可视化结果的路径字典
        """
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import seaborn as sns
        from pathlib import Path
        
        # 设置全局字体大小
        plt.rcParams.update({'font.size': 16})
        
        paths = {}
        num_tokens, hidden_dim = embeddings.shape
        
        if verbose:
            print(f"  Embedding shape: {embeddings.shape}")
        
        # 1. 热力图可视化
        if verbose:
            print(f"  [1/5] 生成热力图...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 对embedding进行标准化以便更好地可视化
        embeddings_normalized = (embeddings - embeddings.mean()) / (embeddings.std() + 1e-8)
        
        # 如果维度太大，进行采样
        if hidden_dim > 500:
            step = hidden_dim // 500
            embeddings_plot = embeddings_normalized[:, ::step]
        else:
            embeddings_plot = embeddings_normalized
        
        sns.heatmap(embeddings_plot, cmap='viridis', ax=ax, 
                   cbar_kws={'label': 'Normalized Value'})
        # 设置colorbar的字体大小
        # seaborn heatmap会在figure中创建colorbar，找到它并设置字体
        for cax in ax.figure.axes:
            if cax != ax and hasattr(cax, 'yaxis'):
                # 这可能是colorbar axes
                try:
                    cax.yaxis.label.set_fontsize(16)
                    cax.yaxis.label.set_fontweight('bold')
                    cax.tick_params(labelsize=14)
                except:
                    pass
        ax.set_xlabel('Feature Dimension', fontsize=18, fontweight='bold')
        ax.set_ylabel('Token Position', fontsize=18, fontweight='bold')
        ax.set_title(f'Vision Embeddings Heatmap\n{num_tokens} tokens × {hidden_dim} dimensions', fontsize=20, fontweight='bold')
        ax.tick_params(labelsize=14)
        
        heatmap_path = output_dir / f"{sample_id}_embedding_heatmap.png"
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        paths['heatmap'] = str(heatmap_path)
        
        # 2. PCA降维可视化（2D）
        if verbose:
            print(f"  [2/5] 生成PCA降维可视化...")
        
        if num_tokens > 1:
            pca = PCA(n_components=min(2, num_tokens))
            embeddings_pca = pca.fit_transform(embeddings)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 绘制轨迹
            ax.plot(embeddings_pca[:, 0], embeddings_pca[:, 1], 'o-', alpha=0.6, linewidth=1, markersize=4)
            
            # 标注起点和终点
            ax.scatter(embeddings_pca[0, 0], embeddings_pca[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
            ax.scatter(embeddings_pca[-1, 0], embeddings_pca[-1, 1], c='red', s=100, marker='s', label='End', zorder=5)
            
            # 添加颜色渐变（表示时间顺序）
            colors = np.arange(num_tokens)
            scatter = ax.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=colors, cmap='viridis', s=30, alpha=0.7)
            cbar = plt.colorbar(scatter, ax=ax, label='Token Position')
            cbar.set_label('Token Position', fontsize=16, fontweight='bold')
            cbar.ax.tick_params(labelsize=14)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=18, fontweight='bold')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)' if len(pca.explained_variance_ratio_) > 1 else 'PC2', fontsize=18, fontweight='bold')
            ax.set_title(f'PCA Visualization of Vision Embeddings\n{num_tokens} tokens', fontsize=20, fontweight='bold')
            ax.legend(fontsize=16, prop={'weight': 'bold'})
            ax.tick_params(labelsize=14)
            ax.grid(True, alpha=0.3)
            
            pca_path = output_dir / f"{sample_id}_embedding_pca.png"
            plt.tight_layout()
            plt.savefig(pca_path, dpi=150, bbox_inches='tight')
            plt.close()
            paths['pca'] = str(pca_path)
        
        # 3. t-SNE降维可视化（如果token数量足够多）
        if num_tokens >= 5:
            if verbose:
                print(f"  [3/5] 生成t-SNE降维可视化...")
            
            try:
                # t-SNE需要至少perplexity+1个样本
                perplexity = min(30, num_tokens - 1)
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                embeddings_tsne = tsne.fit_transform(embeddings)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # 绘制轨迹
                ax.plot(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 'o-', alpha=0.6, linewidth=1, markersize=4)
                
                # 标注起点和终点
                ax.scatter(embeddings_tsne[0, 0], embeddings_tsne[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
                ax.scatter(embeddings_tsne[-1, 0], embeddings_tsne[-1, 1], c='red', s=100, marker='s', label='End', zorder=5)
                
                # 添加颜色渐变
                colors = np.arange(num_tokens)
                scatter = ax.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=colors, cmap='viridis', s=30, alpha=0.7)
                cbar = plt.colorbar(scatter, ax=ax, label='Token Position')
                cbar.set_label('Token Position', fontsize=16, fontweight='bold')
                cbar.ax.tick_params(labelsize=14)
                
                ax.set_xlabel('t-SNE Dimension 1', fontsize=18, fontweight='bold')
                ax.set_ylabel('t-SNE Dimension 2', fontsize=18, fontweight='bold')
                ax.set_title(f't-SNE Visualization of Vision Embeddings\n{num_tokens} tokens', fontsize=20, fontweight='bold')
                ax.legend(fontsize=16, prop={'weight': 'bold'})
                ax.tick_params(labelsize=14)
                ax.grid(True, alpha=0.3)
                
                tsne_path = output_dir / f"{sample_id}_embedding_tsne.png"
                plt.tight_layout()
                plt.savefig(tsne_path, dpi=150, bbox_inches='tight')
                plt.close()
                paths['tsne'] = str(tsne_path)
            except Exception as e:
                if verbose:
                    print(f"    ⚠️  t-SNE失败: {e}")
        
        # 4. 统计特征可视化
        if verbose:
            print(f"  [4/5] 生成统计特征可视化...")
        
        # 计算统计特征
        norms = np.linalg.norm(embeddings, axis=1)
        means = np.mean(embeddings, axis=1)
        stds = np.std(embeddings, axis=1)
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 范数
        axes[0].plot(norms, marker='o', linewidth=2, markersize=4)
        axes[0].set_xlabel('Token Position', fontsize=18, fontweight='bold')
        axes[0].set_ylabel('L2 Norm', fontsize=18, fontweight='bold')
        axes[0].set_title('Embedding Norm Across Tokens', fontsize=20, fontweight='bold')
        axes[0].tick_params(labelsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # 均值
        axes[1].plot(means, marker='o', linewidth=2, markersize=4, color='orange')
        axes[1].set_xlabel('Token Position', fontsize=18, fontweight='bold')
        axes[1].set_ylabel('Mean Value', fontsize=18, fontweight='bold')
        axes[1].set_title('Embedding Mean Across Tokens', fontsize=20, fontweight='bold')
        axes[1].tick_params(labelsize=14)
        axes[1].grid(True, alpha=0.3)
        
        # 标准差
        axes[2].plot(stds, marker='o', linewidth=2, markersize=4, color='green')
        axes[2].set_xlabel('Token Position', fontsize=18, fontweight='bold')
        axes[2].set_ylabel('Standard Deviation', fontsize=18, fontweight='bold')
        axes[2].set_title('Embedding Std Across Tokens', fontsize=20, fontweight='bold')
        axes[2].tick_params(labelsize=14)
        axes[2].grid(True, alpha=0.3)
        
        stats_path = output_dir / f"{sample_id}_embedding_stats.png"
        plt.tight_layout()
        plt.savefig(stats_path, dpi=150, bbox_inches='tight')
        plt.close()
        paths['stats'] = str(stats_path)
        
        # 5. 相似度矩阵
        if verbose:
            print(f"  [5/5] 生成相似度矩阵...")
        
        # 计算余弦相似度
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(similarity_matrix, cmap='coolwarm', center=0, ax=ax, 
                   cbar_kws={'label': 'Cosine Similarity'}, vmin=-1, vmax=1)
        # 设置colorbar的字体大小
        # seaborn heatmap会在figure中创建colorbar，找到它并设置字体
        for cax in ax.figure.axes:
            if cax != ax and hasattr(cax, 'yaxis'):
                # 这可能是colorbar axes
                try:
                    cax.yaxis.label.set_fontsize(16)
                    cax.yaxis.label.set_fontweight('bold')
                    cax.tick_params(labelsize=14)
                except:
                    pass
        ax.set_xlabel('Token Position', fontsize=18, fontweight='bold')
        ax.set_ylabel('Token Position', fontsize=18, fontweight='bold')
        ax.set_title(f'Token Similarity Matrix\n{num_tokens} tokens', fontsize=20, fontweight='bold')
        ax.tick_params(labelsize=14)
        
        similarity_path = output_dir / f"{sample_id}_embedding_similarity.png"
        plt.tight_layout()
        plt.savefig(similarity_path, dpi=150, bbox_inches='tight')
        plt.close()
        paths['similarity'] = str(similarity_path)
        
        if verbose:
            print(f"  ✓ 所有可视化已生成")
        
        return paths


if __name__ == "__main__":
    print("Testing CoT Compressor...")

    compressor = CoTCompressor()

    test_cot = """
                Let me think step by step:

                Step 1: Identify the given information
                - Ducks lay 16 eggs per day
                - Janet eats 3 eggs for breakfast
                - Janet uses 4 eggs for baking

                Step 2: Calculate eggs used
                Total used = 3 + 4 = 7 eggs

                Step 3: Calculate eggs remaining
                Remaining = 16 - 7 = 9 eggs

                Step 4: Calculate earnings
                Earnings = 9 × $2 = $18

                Therefore, Janet makes $18 per day.
                """

    stats = compressor.compute_compression_stats(test_cot)
    print(f"\n✓ Compression stats:")
    print(f"  Original tokens: {stats['original_tokens']}")
    print(f"  Compressed tokens: {stats['compressed_tokens']}")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}:1")
    print(f"  Saved: {stats['saved_percentage']:.1f}%")

    question = "How much does Janet make per day?"
    answer = "$18"

    outputs = compressor.forward(question, test_cot, answer)
    print(f"\n✓ Forward pass completed")
    print(f"  Loss: {outputs['loss'].item() if outputs['loss'] is not None else 'N/A'}")
    print(f"  Vision tokens: {outputs['num_vision_tokens']}")
