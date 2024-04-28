"""
Score level fusion model using CLIP
Code adapted from OpenAI's CLIP codebase
"""

import torch
import math
from torch import nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPTextModel, CLIPVisionModelWithProjection, AutoProcessor
import torch.distributed.nn


class CLIPInstructFusion(nn.Module):
    def __init__(self, model_name="ViT-B/32", device="cuda", jit=False, download_root=None, config=None):
        super().__init__()
        self.clip_txt_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_img_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        # Load pre-trained CLIP model
        # self.clip_model, self.img_preprocess_fn = clip.load(model_name, device, jit, download_root=download_root)
        self.tokenizer = processor.tokenizer
        self.img_preprocess_fn = processor.image_processor
        self.loss_function = nn.CrossEntropyLoss()
        HIDDEN_SIZE = 768
        num_mm_heads = 8
        num_mm_layers = 6
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=HIDDEN_SIZE, nhead=num_mm_heads, batch_first=True, activation="gelu")
        self.fusion_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_mm_layers)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, HIDDEN_SIZE))
        self.embed_token = torch.nn.Parameter(torch.zeros(1, 1, HIDDEN_SIZE))

        if config is not None:
            self.gather_embeddings = config.model.gather_embeddings
            self.in_batch_neg_num = config.data_config.in_batch_neg_num

    def get_img_preprocess_fn(self):
        return self.img_preprocess_fn

    def get_tokenizer(self):
        def tokenizer_wrapper(txt):
            tokenizer = self.tokenizer
            txt_tensor = tokenizer(txt, max_length=77, truncation=True, padding=True,return_tensors='pt')
            return txt_tensor

        return tokenizer_wrapper


    def fuse_embeddings(self, img_emb, txt_emb):
        fused_emb = img_emb + txt_emb
        return fused_emb

    def encode_multimodal_input(self, txt_tensor, img_tensor, txt_mask, img_mask):
        """
        :param txt_tensor:
        :param img_tensor:
        :param txt_mask:  expected shape: [batch_size, 1]
        :param img_mask:  expected shape: [batch_size, 1]
        :return:
        """
        output = self.clip_txt_model(**txt_tensor, output_hidden_states=True)
        txt_hidden_states = output.last_hidden_state
        
        img = {}
        img["pixel_values"] = img_tensor
        output = self.clip_img_model(**img, output_hidden_states=True)
        img_embed = output.image_embeds 

        
        cls_tokens = self.cls_token.expand(txt_hidden_states.shape[0], -1, -1)
        embed_tokens = self.embed_token.expand(txt_hidden_states.shape[0], -1, -1)

        fusion_transformer_inputs = torch.cat((cls_tokens, img_embed.unsqueeze(1), txt_hidden_states, embed_tokens), dim=1)
        fusion_output = self.fusion_encoder(fusion_transformer_inputs)


        return fusion_output[:,0,:].squeeze(1)  # shape: [batch_size, embed_dim]

    def encode_multimodal_input_with_prompt(self, txt_tensor, img_tensor, txt_mask, img_mask, prompt_tensor, prompt_mask):
        """
        :param txt_tensor:
        :param img_tensor:
        :param txt_mask:  expected shape: [batch_size, 1]
        :param img_mask:  expected shape: [batch_size, 1]
        :return:
        """
        
        output = self.clip_txt_model(**txt_tensor, output_hidden_states=True)
        txt_hidden_states = output.last_hidden_state
        output = self.clip_txt_model(**prompt_tensor, output_hidden_states=True)
        prompt_hidden_states = output.last_hidden_state
        
        img = {}
        img["pixel_values"] = img_tensor
        output = self.clip_img_model(**img, output_hidden_states=True)
        img_embed = output.image_embeds 
        
        cls_tokens = self.cls_token.expand(txt_hidden_states.shape[0], -1, -1)
        
        
        fusion_transformer_inputs = torch.cat((cls_tokens, img_embed.unsqueeze(1), txt_hidden_states, prompt_hidden_states), dim=1)
        fusion_output = self.fusion_encoder(fusion_transformer_inputs)


        return fusion_output[:,0,:].squeeze(1)  # shape: [batch_size, embed_dim]

    def get_logit_scale(self):
        return self.clip_img_model.logit_scale.exp()

    def compute_inbatch_contrastive_loss(self, batch):
        """
         adapted from the CLIP codebase and UniVL-DR codebase

        :param model:
        :param batch:
        :param loss_function:
        :return:
        """
        txt_batched = batch["txt_batched"]
        image_batched = batch["image_batched"]
        prompt_batched = batch["prompt_batched"]
        prompt_mask_batched = batch["prompt_mask_batched"]
        txt_mask_batched = batch["txt_mask_batched"]
        image_mask_batched = batch["image_mask_batched"]
        index_mapping = batch["index_mapping"]
        enable_hard_neg = "neg_cand_list" in index_mapping

        # Compute embeddings
        q_embeds, p_embeds, n_embeds = None, None, None 
        if len(prompt_batched) == 0:
            embeddings = self.encode_multimodal_input(txt_batched, image_batched, txt_mask_batched, image_mask_batched)
            # Extract embeddings
            q_embeds = embeddings[torch.tensor(index_mapping["query"]).flatten()]  # shape: [bs, embed_dim]
            p_embeds = embeddings[torch.tensor(index_mapping["pos_cand"]).flatten()]  # shape: [bs, embed_dim]
            n_embeds = None
            if enable_hard_neg:
                n_embeds = embeddings[torch.tensor(index_mapping["neg_cand_list"])]  # [bs, neg_num, embed_dim]
        else:
            
            #for key in txt_batched:
            query_txt_batched = {}
            for key in txt_batched:
                index_mapping["query"] = index_mapping["query"].int()
                query_txt_batched[key] = txt_batched[key][index_mapping["query"].flatten()]
            query_img_batched = image_batched[torch.tensor(index_mapping["query"]).flatten()]
            # query_prompt_batched = prompt_batched[torch.tensor(index_mapping["query"]).flatten()]
            # query_prompt_mask_batched = prompt_mask_batched[torch.tensor(index_mapping["query"]).flatten()]
            query_txt_mask_batched = txt_mask_batched[torch.tensor(index_mapping["query"]).flatten()]
            query_img_mask_batched = image_mask_batched[torch.tensor(index_mapping["query"]).flatten()]
            

            pos_txt_batched = {}
            index_mapping["pos_cand"] = index_mapping["pos_cand"].int()
            for key in txt_batched:
                pos_txt_batched[key] = txt_batched[key][torch.tensor(index_mapping["pos_cand"]).flatten()]
            pos_img_batched = image_batched[torch.tensor(index_mapping["pos_cand"]).flatten()]
            pos_txt_mask_batched = txt_mask_batched[torch.tensor(index_mapping["pos_cand"]).flatten()]
            pos_img_mask_batched = image_mask_batched[torch.tensor(index_mapping["pos_cand"]).flatten()]
            q_embeds = self.encode_multimodal_input_with_prompt(query_txt_batched, query_img_batched, query_txt_mask_batched, query_img_mask_batched, prompt_batched, prompt_mask_batched)
            p_embeds = self.encode_multimodal_input(pos_txt_batched,pos_img_batched, pos_txt_mask_batched, pos_img_mask_batched)

            
        
        bs = q_embeds.size(0)

        # Normalized features
        q_embeds = F.normalize(q_embeds, dim=-1)
        p_embeds = F.normalize(p_embeds, dim=-1)

        logit_scale = math.exp(2.6592)

        # We gather tensors from all gpus
        if self.gather_embeddings:
            all_p_embeds = torch.cat(torch.distributed.nn.all_gather(p_embeds), dim=0)  # [bs * num_gpus, embed_dim]

        if enable_hard_neg:
            # Normalize the negative embeddings
            n_embeds = F.normalize(n_embeds, dim=-1)

            # Number of in-batch positives to add as negatives
            in_batch_neg_num = min(bs - 1, self.in_batch_neg_num)

            # Augment neg_cand_embeddings with a subset of in-batch positive candidates from other queries
            mask = torch.eye(bs).to(n_embeds.device) == 0
            in_batch_negs = p_embeds.unsqueeze(1).expand(-1, bs, -1)[mask].reshape(bs, bs - 1, -1)
            in_batch_negs = in_batch_negs[:, :in_batch_neg_num, :]
            aug_n_embeds = torch.cat([n_embeds, in_batch_negs], dim=1)  # [bs, neg_num + in_batch_neg_num, embed_dim]

            # Compute similarity scores for positives and negatives
            pos_scores = (q_embeds * p_embeds).sum(-1) * logit_scale  # [bs]
            neg_scores = (q_embeds.unsqueeze(1) * aug_n_embeds).sum(-1) * logit_scale  # [bs, neg_num +in_batch_neg_num]
            logit_matrix = torch.cat([pos_scores.unsqueeze(-1), neg_scores], 1)  # [bs, neg_num + in_batch_neg_num + 1]

            # Compute log softmax over the matrix
            lsm = F.log_softmax(logit_matrix, dim=1)

            # The NNL loss for the positive candidate
            loss = torch.mean(-1.0 * lsm[:, 0])

            # Compute accuracy by checking which instances have the positive candidate as the most similar one
            _max_score, max_idxs = torch.max(logit_matrix, 1)
            accuracy = (max_idxs == 0).sum() / bs
        else:
            if self.gather_embeddings:
                score = torch.matmul(q_embeds, all_p_embeds.t()) * logit_scale  # [bs, bs * num_gpus]
                gpu_id = torch.distributed.get_rank()
                sim_targets = (gpu_id * bs + torch.arange(bs)).to(score.device)  # [bs]
            else:
                score = torch.matmul(q_embeds, p_embeds.t()) * logit_scale  # [bs, bs]
                sim_targets = torch.arange(bs).to(score.device)  # [bs]

            # compute loss
            loss = self.loss_function(score, sim_targets)
            _max_score, max_idxs = torch.max(score, 1)
            accuracy = (max_idxs == sim_targets).sum() / bs

        outputs = {"loss": loss, "accuracy": accuracy}
        return outputs

    def forward(self, batch, encode_mbeir_batch=False):
        if encode_mbeir_batch:
            return self.encode_mbeir_batch(batch)
        return self.compute_inbatch_contrastive_loss(batch)

    def encode_mbeir_batch(self, batch):
        # Get hashed id_list
        id_list = batch.get("did_list") or batch.get("qid_list")
        assert id_list is not None, "id_list must be provided."
        assert isinstance(id_list[0], int), "id_list must be hashed to int."

        # Compute embeddings
        if "prompt_batched" not in batch:
            embeddings = self.encode_multimodal_input(
                batch["txt_batched"], batch["image_batched"], batch["txt_mask_batched"], batch["image_mask_batched"]
            )
            assert embeddings.size(0) == len(id_list), "embeddings and id_batched must have the same batch size."
            return embeddings, id_list
        else:
            
            index_mapping = batch["index_mapping"]
            enable_hard_neg = "neg_cand_list" in index_mapping
            txt_batched = batch["txt_batched"]
            image_batched = batch["image_batched"]
            prompt_batched = batch["prompt_batched"]
            prompt_mask_batched = batch["prompt_mask_batched"]
            txt_mask_batched = batch["txt_mask_batched"]
            image_mask_batched = batch["image_mask_batched"]
            query_txt_batched = {}
            for key in txt_batched:
                query_txt_batched[key] = txt_batched[key][torch.tensor(index_mapping["query"]).flatten()]
            query_img_batched = image_batched[torch.tensor(index_mapping["query"]).flatten()]
            query_txt_mask_batched = txt_mask_batched[torch.tensor(index_mapping["query"]).flatten()]
            query_img_mask_batched = image_mask_batched[torch.tensor(index_mapping["query"]).flatten()]
            
            if "pos_cand" in index_mapping:
                pos_txt_batched = {}
                for key in txt_batched:
                    pos_txt_batched[key] = txt_batched[key][torch.tensor(index_mapping["pos_cand"]).flatten()]
                pos_img_batched = image_batched[torch.tensor(index_mapping["pos_cand"]).flatten()]
                pos_txt_mask_batched = txt_mask_batched[torch.tensor(index_mapping["pos_cand"]).flatten()]
                pos_img_mask_batched = image_mask_batched[torch.tensor(index_mapping["pos_cand"]).flatten()]
                p_embeds = self.encode_multimodal_input(pos_txt_batched, pos_txt_mask_batched, pos_img_batched, pos_img_mask_batched)

            q_embeds = self.encode_multimodal_input_with_prompt(query_txt_batched, query_img_batched, query_txt_mask_batched, query_img_mask_batched, prompt_batched, prompt_mask_batched)

            # if enable_hard_neg:
            #     ## There is no hard neg anyways in evaluate
            #     neg_txt_batched = txt_batched[torch.tensor(index_mapping["neg_cand_list"]).flatten()]
            #     neg_img_batched = image_batched[torch.tensor(index_mapping["neg_cand_list"]).flatten()]
            #     neg_txt_mask_batched = txt_mask_batched[torch.tensor(index_mapping["neg_cand_list"]).flatten()]
            #     neg_img_mask_batched = image_mask_batched[torch.tensor(index_mapping["neg_cand_list"]).flatten()]
            #     n_embeds = self.encode_multimodal_input(neg_txt_batched, neg_txt_mask_batched, neg_img_batched, neg_img_mask_batched)
            embeddings = torch.zeros(len(id_list),q_embeds.size(-1) ,device=q_embeds.device)
            embeddings[torch.tensor(index_mapping["query"]).flatten(),:] = q_embeds
            if "pos_cand" in index_mapping:
                embeddings[torch.tensor(index_mapping["pos_cand"]),:] = p_embeds
            assert embeddings.size(0) == len(id_list), "embeddings and id_batched must have the same batch size."
            return embeddings, id_list


