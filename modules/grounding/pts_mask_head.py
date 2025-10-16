import torch
import torch.nn as nn
import pdb, time
from torch_scatter import scatter_mean, scatter_add
from modules.build import GROUNDING_REGISTRY
from modules.grounding.query_encoder import layer_repeat,QueryEncoderLayer
from modules.weights import _init_weights_bert
from modules.utils import get_activation_fn
class CrossAttentionLayer(nn.Module):
    """Cross attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout, fix=False):
        super().__init__()
        self.fix = fix
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        """Init weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sources, queries, attn_masks=None):
        """Forward pass.

        Args:
            sources (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).
            queries (List[Tensor]): of len batch_size,
                each of shape(n_queries_i, d_model).
            attn_masks (List[Tensor] or None): of len batch_size,
                each of shape (n_queries, n_points).
        
        Return:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        outputs = []
        for i in range(len(sources)):
            k = v = sources[i]
            attn_mask = attn_masks[i] if attn_masks is not None else None
            output, _ = self.attn(queries[i], k, v, attn_mask=attn_mask)
            if self.fix:
                output = self.dropout(output)
            output = output + queries[i]
            if self.fix:
                output = self.norm(output)
            outputs.append(output)
        return outputs


class SelfAttentionLayer(nn.Module):
    """Self attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z, _ = self.attn(y, y, y)
            z = self.dropout(z) + y
            z = self.norm(z)
            out.append(z)
        return out


class FFN(nn.Module):
    """Feed forward network.

    Args:
        d_model (int): Model dimension.
        hidden_dim (int): Hidden dimension.
        dropout (float): Dropout rate.
        activation_fn (str): 'relu' or 'gelu'.
    """

    def __init__(self, d_model, hidden_dim, dropout, activation_fn):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU() if activation_fn == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z = self.net(y)
            z = z + y
            z = self.norm(z)
            out.append(z)
        return out


class QueryDecoder(nn.Module):
    """Query decoder for SPFormer.

    Args:
        num_layers (int): Number of transformer layers.
        num_instance_queries (int): Number of instance queries.
        num_semantic_queries (int): Number of semantic queries.
        num_classes (int): Number of classes.
        in_channels (int): Number of input channels.
        d_model (int): Number of channels for model layers.
        num_heads (int): Number of head in attention layer.
        hidden_dim (int): Dimension of attention layer.
        dropout (float): Dropout rate for transformer layer.
        activation_fn (str): 'relu' of 'gelu'.
        iter_pred (bool): Whether to predict iteratively.
        attn_mask (bool): Whether to use mask attention.
        pos_enc_flag (bool): Whether to use positional enconding.
    """

    def __init__(self, num_layers, num_instance_queries, num_semantic_queries,
                 num_classes, in_channels, d_model, num_heads, hidden_dim,
                 dropout, activation_fn, iter_pred, attn_mask, fix_attention,
                 objectness_flag, **kwargs):
        super().__init__()
        self.objectness_flag = objectness_flag
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.LayerNorm(d_model), nn.ReLU())
        if num_instance_queries + num_semantic_queries > 0:
            self.query = nn.Embedding(num_instance_queries + num_semantic_queries, d_model)
        if num_instance_queries == 0:
            self.query_proj = nn.Sequential(
                nn.Linear(in_channels, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model))
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for i in range(num_layers):
            self.cross_attn_layers.append(
                CrossAttentionLayer(
                    d_model, num_heads, dropout, fix_attention))#fix_attention决定要不要使用dropout和layernorm
            self.self_attn_layers.append(
                SelfAttentionLayer(d_model, num_heads, dropout))
            self.ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, num_classes + 1))
        if objectness_flag:#False
            self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
    
    def _get_queries(self, queries=None, batch_size=None):
        """Get query tensor.

        Args:
            queries (List[Tensor], optional): of len batch_size,
                each of shape (n_queries_i, in_channels).
            batch_size (int, optional): batch size.
        
        Returns:
            List[Tensor]: of len batch_size, each of shape
                (n_queries_i, d_model).
        """
        if batch_size is None:
            batch_size = len(queries)
        
        result_queries = []
        for i in range(batch_size):
            result_query = []
            if hasattr(self, 'query'):
                result_query.append(self.query.weight)
            if queries is not None:
                result_query.append(self.query_proj(queries[i]))
            result_queries.append(torch.cat(result_query))
        return result_queries

    def _forward_head(self, queries, mask_feats):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, pred_scores, pred_masks, attn_masks = [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            cls_preds.append(self.out_cls(norm_query))
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        return cls_preds, pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, queries):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats)
        return dict(
            cls_preds=cls_preds,
            masks=pred_masks,
            scores=pred_scores)

    def forward_iter_pred(self, x, queries):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and aux_outputs.
        """
        cls_preds, pred_scores, pred_masks = [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
            queries, mask_feats)
        cls_preds.append(cls_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
                queries, mask_feats)
            cls_preds.append(cls_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            {'cls_preds': cls_pred, 'masks': masks, 'scores': scores}
            for cls_pred, scores, masks in zip(
                cls_preds[:-1], pred_scores[:-1], pred_masks[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)

    def forward(self, x, queries=None):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        if self.iter_pred:
            return self.forward_iter_pred(x, queries)
        else:
            return self.forward_simple(x, queries)


class ScanNetQueryDecoder(QueryDecoder):
    """We simply add semantic prediction for each instance query.
    """
    def __init__(self, num_instance_classes, num_semantic_classes,
                 d_model, num_semantic_linears, **kwargs):
        super().__init__(
            num_classes=num_instance_classes, d_model=d_model, **kwargs)
        assert num_semantic_linears in [1, 2]
        if num_semantic_linears == 2:
            self.out_sem = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, num_semantic_classes + 1))
        else:
            self.out_sem = nn.Linear(d_model, num_semantic_classes + 1)

    def _forward_head(self, queries, mask_feats, last_flag):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_instance_classes + 1).
                List[Tensor] or None: Semantic predictions of len batch_size,
                    each of shape (n_queries_i, n_semantic_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, sem_preds, pred_scores, pred_masks, attn_masks = [], [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            cls_preds.append(self.out_cls(norm_query))
            if last_flag:
                sem_preds.append(self.out_sem(norm_query))
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        sem_preds = sem_preds if last_flag else None
        return cls_preds, sem_preds, pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, queries):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, sem_preds, pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats, last_flag=True)
        return dict(
            cls_preds=cls_preds,
            sem_preds=sem_preds,
            masks=pred_masks,
            scores=pred_scores)

    def forward_iter_pred(self, x, queries):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, scores,
                and aux_outputs.
        """
        cls_preds, sem_preds, pred_scores, pred_masks = [], [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        cls_pred, sem_pred, pred_score, pred_mask, attn_mask = self._forward_head(
            queries, mask_feats, last_flag=False)
        cls_preds.append(cls_pred)
        sem_preds.append(sem_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            last_flag = i == len(self.cross_attn_layers) - 1
            cls_pred, sem_pred, pred_score, pred_mask, attn_mask = self._forward_head(
                queries, mask_feats, last_flag)
            cls_preds.append(cls_pred)
            sem_preds.append(sem_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            dict(
                cls_preds=cls_pred, sem_preds=sem_pred, masks=masks, scores=scores)
            for cls_pred, sem_pred, scores, masks in zip(
                cls_preds[:-1], sem_preds[:-1], pred_scores[:-1], pred_masks[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            sem_preds=sem_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)


class ScanNetMixQueryDecoder(QueryDecoder):
    """We simply add semantic prediction for each instance query.
    """
    def __init__(self, num_instance_classes, num_semantic_classes,
                 d_model, num_semantic_linears, in_channels, share_attn_mlp, share_mask_mlp,
                 cross_attn_mode, mask_pred_mode, temporal_attn=False, bbox_flag=False, **kwargs):
        super().__init__(
            num_classes=num_instance_classes, d_model=d_model, in_channels=in_channels, **kwargs)
        assert num_semantic_linears in [1, 2]
        assert isinstance(cross_attn_mode, list)
        assert isinstance(mask_pred_mode, list)
        assert mask_pred_mode[-1] == "P"

        self.cross_attn_mode = cross_attn_mode
        self.mask_pred_mode = mask_pred_mode
        self.temporal_attn = temporal_attn# False

        self.share_attn_mlp = share_attn_mlp # False
        if not share_attn_mlp:
            if "P" in self.cross_attn_mode:
                self.input_pts_proj = nn.Sequential(
                    nn.Linear(3 + in_channels, d_model), nn.LayerNorm(d_model), nn.ReLU())

        self.share_mask_mlp = share_mask_mlp
        if not share_mask_mlp:
            if "P" in self.mask_pred_mode:
                self.x_pts_mask = nn.Sequential(
                    nn.Linear(3 + in_channels, d_model), nn.ReLU(),
                    nn.Linear(d_model, d_model))

        self.bbox_flag = bbox_flag
        if self.bbox_flag:#True
            self.out_reg = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, 6))

        if num_semantic_linears == 2:#1
            self.out_sem = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, num_semantic_classes + 1))
        else:
            self.out_sem = nn.Linear(d_model, num_semantic_classes + 1)

    def _forward_head(self, queries, mask_feats, mask_pts_feats, last_flag, layer):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_instance_classes + 1).
                List[Tensor] or None: Semantic predictions of len batch_size,
                    each of shape (n_queries_i, n_semantic_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, sem_preds, pred_scores, pred_masks, attn_masks, pred_bboxes = [], [], [], [], [], []
        object_queries = []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i]) #layerNorm
            object_queries.append(norm_query)
            cls_preds.append(self.out_cls(norm_query))
            if last_flag:
                sem_preds.append(self.out_sem(norm_query))
            pred_score = self.out_score(norm_query) if self.objectness_flag else None
            pred_scores.append(pred_score)
            if self.bbox_flag:
                reg_final = self.out_reg(norm_query)
                reg_distance = torch.exp(reg_final[:, 3:6])
                pred_bbox = torch.cat([reg_final[:, :3], reg_distance], dim=1)
            else: pred_bbox = None
            pred_bboxes.append(pred_bbox)
            if self.mask_pred_mode[layer] == "SP":
                pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            elif self.mask_pred_mode[layer] == "P":
                pred_mask = torch.einsum('nd,md->nm', norm_query, mask_pts_feats[i])
            else:
                raise NotImplementedError("Query decoder not implemented!")
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        sem_preds = sem_preds if last_flag else None
        return cls_preds, sem_preds, pred_scores, pred_masks, attn_masks, object_queries, pred_bboxes

    def forward_iter_pred(self, sp_feats, p_feats, queries, super_points, prev_queries=None):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, scores,
                and aux_outputs.
        """
        cls_preds, sem_preds, pred_scores, pred_masks = [], [], [], []
        object_queries, pred_bboxes = [], []
        inst_feats = [self.input_proj(y) for y in sp_feats] if "SP" in self.cross_attn_mode else None
        inst_pts_feats = [self.input_proj(y) if self.share_attn_mlp else self.input_pts_proj(y)
             for y in p_feats] if "P" in self.cross_attn_mode else None
        mask_feats = [self.x_mask(y) for y in sp_feats] if "SP" in self.mask_pred_mode else None
        mask_pts_feats = [self.x_mask(y) if self.share_mask_mlp else self.x_pts_mask(y)
             for y in p_feats] if "P" in self.mask_pred_mode else None
        queries = self._get_queries(queries, len(sp_feats))
        cls_pred, sem_pred, pred_score, pred_mask, attn_mask, object_query, pred_bbox = \
             self._forward_head(queries, mask_feats, mask_pts_feats, last_flag=False, layer=0) #得到各类预测输出一次
        cls_preds.append(cls_pred)
        sem_preds.append(sem_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        object_queries.append(object_query)
        pred_bboxes.append(pred_bbox)
        for i in range(len(self.cross_attn_layers)):#三层
            if self.cross_attn_mode[i+1] == "SP" and self.mask_pred_mode[i] == "SP":
                queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
                # print("SP SP queries", queries[0].shape)
            elif self.cross_attn_mode[i+1] == "SP" and self.mask_pred_mode[i] == "P":   # current method, change P mask to SP
                
                xyz_weights = torch.chunk(super_points[1], len(super_points[0]), dim=0)
                for att, sp, xyz_w in zip(attn_mask, super_points[0], xyz_weights):
                    # print("att", att.shape)
                    # print("sp", sp.shape)
                    # print("xyz_w", xyz_w.shape)
                    tmp_res=torch.einsum("ij,jk->ij",att.float(),xyz_w) #(sp,points) (points,1)
                    # print("att.float() * xyz_w.view(1, -1)",tmp_res.shape)
                attn_mask_score = [scatter_mean(att.float() * xyz_w.view(1, -1), sp, dim=1)
                     for att, sp, xyz_w in zip(attn_mask, super_points[0], xyz_weights)]
                attn_mask = [(att > 0.5).bool() for att in attn_mask_score] # > 0.5, not <  #注意力遮罩
                # print("attn_mask_score early", attn_mask_score[0].shape)
                # print("attn_mask early", attn_mask[0].shape)
                # If attn_mask has all-True row, the result of CA will be nan
                for j in range(len(attn_mask)):
                    mask = ~(attn_mask_score[j] == attn_mask_score[j].min(dim=1, keepdim=True)[0])
                    attn_mask[j] *= mask
                queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
                # print("xyz_weights", xyz_weights)
                # print("super_points", len(super_points))
                # print("super_points[1]", super_points[1].shape)
                # print("super_points[0]", len(super_points[0]))
                # print("attn_mask_score", attn_mask_score[0].shape)
                # print("attn_mask", attn_mask[0].shape)
                # print("SP P queries", queries[0].shape)
            elif self.cross_attn_mode[i+1] == "P" and self.mask_pred_mode[i] == "SP":#没用到，不会再点层面做交叉注意力
                attn_mask = [att[:, sp] for att, sp in zip(attn_mask, super_points[0])]
                queries = self.cross_attn_layers[i](inst_pts_feats, queries, attn_mask)
            elif self.cross_attn_mode[i+1] == "P" and self.mask_pred_mode[i] == "P":#没用到，不会再点层面做交叉注意力
                queries = self.cross_attn_layers[i](inst_pts_feats, queries, attn_mask)
            else:
                raise NotImplementedError("Not support yet!")
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            last_flag = i == len(self.cross_attn_layers) - 1
            cls_pred, sem_pred, pred_score, pred_mask, attn_mask, object_query, pred_bbox = \
                 self._forward_head(queries, mask_feats, mask_pts_feats, last_flag, layer=i+1)
            cls_preds.append(cls_pred)
            sem_preds.append(sem_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)
            object_queries.append(object_query)
            pred_bboxes.append(pred_bbox)

        aux_outputs = [
            dict(
                cls_preds=cls_pred, sem_preds=sem_pred, masks=masks, scores=scores, bboxes=bboxes)
            for cls_pred, sem_pred, scores, masks, bboxes in zip(
                cls_preds[:-1], sem_preds[:-1], pred_scores[:-1], pred_masks[:-1], pred_bboxes[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            sem_preds=sem_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            queries=object_queries[-1],
            bboxes=pred_bboxes[-1],
            aux_outputs=aux_outputs)
    
    def forward(self, sp_feats, p_feats, queries, super_points, prev_queries=None):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        if self.iter_pred:
            return self.forward_iter_pred(sp_feats, p_feats, queries, super_points, prev_queries)
        else:
            raise NotImplementedError("No simple forward!!!")

@GROUNDING_REGISTRY.register()
class EmbodiedSAMDecoder(nn.Module):
    def __init__(self, cfg, memories=[], hidden_size=768, num_attention_heads=12, share_layer=False, structure='sequential',num_layers=4,
                 num_blocks=1, spatial_selfattn=False, attn_mask=True,mask_pred_mode=['SP','SP','P','P'], cross_attn_mode=[ "", "SP", "SP","SP"],
                 num_instance_classes=1, num_semantic_classes=200):
        super().__init__()

        self.spatial_selfattn = spatial_selfattn
        query_encoder_layer = QueryEncoderLayer(d_model=hidden_size, nhead=num_attention_heads, memories=memories, spatial_selfattn=spatial_selfattn, structure=structure)
        self.unified_encoder = layer_repeat(query_encoder_layer, num_layers, share_layer)
        self.num_heads = num_attention_heads
        self.num_blocks = num_blocks
        d_model = 256
        num_heads = 8
        in_channels = 96
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_size, d_model), 
            nn.LayerNorm(d_model), nn.ReLU())
        self.cross_attn_mode = cross_attn_mode
        self.mask_pred_mode = mask_pred_mode
        self.attn_mask = attn_mask 
       
        # self.input_pts_proj = nn.Sequential(
        #             nn.Linear(3 + in_channels, d_model), nn.LayerNorm(d_model), nn.ReLU()) #instance feat
        self.x_pts_mask = nn.Sequential(
                            nn.Linear(3 + in_channels, d_model), nn.ReLU(),
                            nn.Linear(d_model, d_model))
        self.x_mask = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, d_model))
        
        self.out_norm = nn.LayerNorm(hidden_size)
        self.out_cls = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, num_instance_classes + 1)) # instance or not
        self.out_reg = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, 6))
        self.out_sem = nn.Linear(hidden_size, num_semantic_classes + 1)
        # self.out_score = nn.Sequential(
        #         nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

        self.apply(_init_weights_bert)
        
    def _forward_head(self, queries, mask_feats, mask_pts_feats, last_flag, layer):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_instance_classes + 1).
                List[Tensor] or None: Semantic predictions of len batch_size,
                    each of shape (n_queries_i, n_semantic_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        # import pudb; pudb.set_trace()
        cls_preds, sem_preds, pred_scores, pred_masks, attn_masks, pred_bboxes = [], [], [], [], [], []
       
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i]) #layerNorm
           
            cls_preds.append(self.out_cls(norm_query))
            if last_flag:
                sem_preds.append(self.out_sem(norm_query))
            # pred_score = self.out_score(norm_query)
            # pred_scores.append(pred_score)
           
            reg_final = self.out_reg(norm_query)#anchor more complicated box head
            reg_distance = torch.exp(reg_final[:, 3:6])
            pred_bbox = torch.cat([reg_final[:, :3], reg_distance], dim=1)
        
            pred_bboxes.append(pred_bbox)
            if self.mask_pred_mode[layer] == "SP":
                proj_query= self.input_proj(norm_query)
                pred_mask = torch.einsum('nd,md->nm', proj_query, mask_feats[i])
            elif self.mask_pred_mode[layer] == "P":
                proj_query= self.input_proj(norm_query)
                pred_mask = torch.einsum('nd,md->nm', proj_query, mask_pts_feats[i])
            else:
                raise NotImplementedError("Query decoder not implemented!")
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False #全true会导致CA结果为nan
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)

       
        return cls_preds, sem_preds, pred_scores, pred_masks, attn_masks, pred_bboxes


    def forward(self, input_dict, pairwise_locs, pcds_w, pts2spidx):
        '''
        数据流程：
        QueryEncoderLayer: 完成memories数据的注意力融合，主要是超点级别的
        mask_head:使用query和点云特征计算输出，mask会输出注意力遮罩
        注意力计算：
        用到geoaware得到的点云权重进行注意力加权（记得配置为可选项）

        '''

        cls_preds, sem_preds, pred_scores, pred_masks ,pred_bboxes ,pred_type = [], [], [], [], [], []

        queries = input_dict['query'][0]
        p_feats = input_dict['pts'][0]
        sp_feats= input_dict['voxel'][0][-1]#last layer of voxel features
        input_dict['voxel'][0] = input_dict['voxel'][0][-1] #select the last layer of voxel features
        pts_w  = pcds_w#list
        point2sp = pts2spidx#list
        # import pudb; pudb.set_trace()
    
        mask_feats = self.x_mask(sp_feats)
        mask_pts_feats = self.x_pts_mask(torch.cat([input_dict['pts'][2], p_feats], dim=-1))
        
        cls_pred, sem_pred, pred_score, pred_mask, attn_mask, pred_bbox = \
             self._forward_head(queries, mask_feats, mask_pts_feats, last_flag=False, layer=0) #得到各类预测输出一次
        pred_type.append("SP")
        cls_preds.append(torch.stack(cls_pred) if cls_pred else cls_pred)
        sem_preds.append(torch.stack(sem_pred) if sem_pred else sem_pred)
        pred_scores.append(torch.stack(pred_score) if pred_score else pred_score)
        pred_masks.append(torch.stack(pred_mask) if pred_mask else pred_mask)
        pred_bboxes.append(torch.stack(pred_bbox) if pred_bbox else pred_bbox)
        
        max_sp_len=0
        for i in range(3):#三层
            if self.cross_attn_mode[i+1] == "SP" and self.mask_pred_mode[i] == "SP":
                # attn_mask: list of (N, M), batch size = B
                # Stack to (B, N, M)
                # import pudb; pudb.set_trace()
                attn_mask = torch.stack(attn_mask, dim=0)  # (B, N, N)
                attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
                max_sp_len = max(max_sp_len, attn_mask.shape[2])
                # print("stage i+1,shape of attn_mask", i+1, attn_mask.shape)
              
            elif self.cross_attn_mode[i+1] == "SP" and self.mask_pred_mode[i] == "P":   # current method, change P mask to SP
                # import pudb; pudb.set_trace()
                xyz_weights = pts_w #list of pts weight
                sp_ids = point2sp
                # 将不同尺寸的掩码填充为相同尺寸，然后堆叠
                # 1. 找到批次中最大的超点数量
                attn_mask_score = [scatter_mean(att.float() * xyz_w.view(1, -1), sp, dim=1)
                     for att, sp, xyz_w in zip(attn_mask, sp_ids, xyz_weights)]
                attn_mask = [(att > 0.5).bool() for att in attn_mask_score] # > 0.5, not <  #注意力遮罩
                # print("attn_mask_score early", attn_mask_score[0].shape)
                # print("attn_mask early", attn_mask[0].shape)
                # If attn_mask has all-True row, the result of CA will be nan
                for j in range(len(attn_mask)):
                    mask = ~(attn_mask_score[j] == attn_mask_score[j].min(dim=1, keepdim=True)[0])
                    attn_mask[j] *= mask
 
                # 2. 创建一个填充后的张量列表
                padded_masks = []
                for m in attn_mask:
                    pad_len = max_sp_len - m.shape[1]
                    # 使用 F.pad 进行填充。我们填充最后一个维度（宽度）
                    # (0, pad_len) 表示在最后一个维度的左边不填充，右边填充 pad_len 个
                    # value=True 表示用 True 来填充，因为 True 在注意力中代表“忽略”
                    padded_mask = torch.nn.functional.pad(m, (0, pad_len), mode='constant', value=True)
                    padded_masks.append(padded_mask)

                # 3. 使用 torch.stack 将填充后的张量列表堆叠成一个批处理张量
                attn_mask_batched = torch.stack(padded_masks, dim=0)  # 形状变为 [B, Q, max_S]
               
                # 4. 为多头注意力机制重复张量
                attn_mask = attn_mask_batched.repeat_interleave(self.num_heads, dim=0)
                # print("stage i+1,shape of attn_mask", i+1, attn_mask.shape)
              

             
            # import pudb; pudb.set_trace()
            queries = self.unified_encoder[i](queries, input_dict, pairwise_locs,attn_mask)

            last_flag = 2 == i
            cls_pred, sem_pred, pred_score, pred_mask, attn_mask, pred_bbox = \
                 self._forward_head(queries, mask_feats, mask_pts_feats, last_flag, layer=i+1)

            pred_type.append(self.mask_pred_mode[i+1])
            cls_preds.append(torch.stack(cls_pred) if cls_pred else cls_pred)
            sem_preds.append(torch.stack(sem_pred) if sem_pred else sem_pred)
            pred_scores.append(torch.stack(pred_score) if pred_score else pred_score)
            pred_masks.append(torch.stack(pred_mask) if pred_mask else pred_mask)
            pred_bboxes.append(torch.stack(pred_bbox) if pred_bbox else pred_bbox)




        return queries, cls_preds, sem_preds, pred_masks, pred_scores, pred_bboxes ,pred_type
