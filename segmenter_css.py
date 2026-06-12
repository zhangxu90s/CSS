import math
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. 因果反事实直达效应投影模块 (Causal Counterfactual Projector)
# =====================================================================
class CausalCounterfactualProjector(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )

        self.gate_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, patch_tokens: torch.Tensor, cls_token: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # 使用大写的 Tuple
        B, N, C = patch_tokens.shape

        # 1. 映射 Query 向量
        cls_q = self.query_proj(cls_token).unsqueeze(1)  # [B, 1, C]

        # 2. 映射 Key / Value 向量
        patch_kv = self.kv_proj(patch_tokens)
        patch_k, patch_v = patch_kv.chunk(2, dim=-1)

        # 3. 事实路径
        factual_response, _ = self.cross_attn(
            query=cls_q, key=patch_k, value=patch_v
        )  # [B, 1, C]

        # 4. 反事实路径
        counterfactual_context = torch.mean(
            patch_v, dim=1, keepdim=True
        ).expand(-1, N, -1)
        counterfactual_response, _ = self.cross_attn(
            query=cls_q, key=patch_k, value=counterfactual_context
        )  # [B, 1, C]
        
        # 5. 因果直达效应估计
        causal_direct_effect = factual_response - counterfactual_response
        
        # ==================== 新增：因果正交 Loss ====================
        # 去掉中间的 1 维度，变成 [B, C]
        c_effect = causal_direct_effect.squeeze(1)
        cf_resp = counterfactual_response.squeeze(1)
        
     
        # 计算其余弦相似度，并取绝对值
        cos_sim = F.cosine_similarity(c_effect, cf_resp, dim=-1)
        orthogonal_loss = torch.mean(torch.abs(cos_sim))
        # ==========================================================
        
     
        
        # 6. 生成纯净的通道门控信号并进行残差融合
        gate = self.gate_proj(causal_direct_effect)  # [B, 1, C]
        scaled_patch_tokens = patch_tokens + gate * patch_tokens

        return self.norm(scaled_patch_tokens), orthogonal_loss
    
    

# =====================================================================
# 2. 多尺度谱域切比雪夫拓扑滤波器 (Spectral Chebyshev Graph Convolution)
# =====================================================================
class SpectralChebyshevGraphConv(nn.Module):
    """多尺度谱域切比雪夫拓扑滤波块 (Spectral Chebyshev Graph Convolution).

    引入2阶切比雪夫多项式展开，使图网络具备自适应的高通/带通边界保留能力，防止图过度平滑。
    """

    def __init__(self, dim: int, num_scales: int, dropout: float = 0.1):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim)
        self.self_proj = nn.Linear(dim, dim)

        # 为每个拓扑尺度分别注入 0 阶、1 阶、2 阶切比雪夫谱算子系数投影器
        self.cheb_coeffs_0 = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(num_scales)]
        )
        self.cheb_coeffs_1 = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(num_scales)]
        )
        self.cheb_coeffs_2 = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(num_scales)]
        )

        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(dim)
    
    def forward(
        self, x: torch.Tensor, adjs: List[torch.Tensor]
    ) -> torch.Tensor:
        h = self.pre_norm(x)
        out = self.self_proj(h)
        
        # 多尺度非线性谱变换
        for idx, adj in enumerate(adjs):
            # 切比雪夫多项式谱展开递归基：
            # T_0(A) = H
            t0 = h

            # T_1(A) = A * H (一阶空间邻居流)
            t1 = torch.matmul(adj, h)

            # T_2(A) = 2 * A * T_1 - T_0 (高阶动量与相位的谱域演化)
            t2 = 2.0 * torch.matmul(adj, t1) - t0

            # 聚合各阶谱特征，网络自适应优化多尺度的频率响应（保持边界锐化）
            scale_filtered = (
                self.cheb_coeffs_0[idx](t0)
                + self.cheb_coeffs_1[idx](t1)
                + self.cheb_coeffs_2[idx](t2)
            )

            out = out + scale_filtered

        return x + self.dropout(out)


    
# =====================================================================
# 3. 多尺度谱域图推理网络 (Multi-Scale Spectral Graph Reasoner)
# =====================================================================
class MultiScaleSpectralGraphReasoner(nn.Module):
    """级联的多尺度谱多项式拓扑图推理层."""

    def __init__(
        self, dim: int, num_scales: int, depth: int = 2, dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SpectralChebyshevGraphConv(
                    dim, num_scales=num_scales, dropout=dropout
                )
                for _ in range(depth)
            ]
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self, x: torch.Tensor, adjs: List[torch.Tensor]
    ) -> torch.Tensor:
        res = x
        for layer in self.layers:
            x = layer(x, adjs)
        return self.out_norm(x) 

    
    

# =====================================================================
# 4. 解耦拓扑分割骨干网络 (Causal Spectral Segmenter)
# =====================================================================
class Segmenter(nn.Module):

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        n_cls: int,
        embed_dim: int = 384,
        gnn_layers: int = 8, ####
        dropout: float = 0.1,
        graph_connectivities: Union[List[int], Tuple[int, ...]] = (#####
            10,12,14
        ),
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.n_cls = n_cls
        self.graph_connectivities = graph_connectivities

        # 动态解析 Patch 尺寸大小
        self.patch_size = encoder.patch_size
        if isinstance(self.patch_size, tuple):
            self.patch_h, self.patch_w = self.patch_size
        else:
            self.patch_h = self.patch_w = int(self.patch_size)

        self.encoder_dim = getattr(encoder, "embed_dim", embed_dim)
        self.graph_dim = embed_dim

        # 核心升级模块实例化
        self.causal_projector = CausalCounterfactualProjector(embed_dim)
        
        
        self.graph_reasoner = MultiScaleSpectralGraphReasoner(
            dim=embed_dim,
            num_scales=len(self.graph_connectivities),
            depth=gnn_layers,
            dropout=dropout,
        )

        # 拓扑缓存空间
        self._adj_cache: Dict[Tuple[int, int, str, int, str, int], torch.Tensor] = {}

    def _infer_grid_size(
        self, h: int, w: int, num_patches: int
    ) -> Tuple[int, int]:
        """依据图像宽高与Token总数逆向推导Patch网格的行列数."""
        patch_h = max(1, math.ceil(h / self.patch_h))
        patch_w = max(1, math.ceil(w / self.patch_w))
        if patch_h * patch_w == num_patches:
            return patch_h, patch_w

        target_ratio = h / max(w, 1)
        best_pair = None
        best_score = float("inf")
        for gh in range(1, int(math.sqrt(num_patches)) + 1):
            if num_patches % gh != 0:
                continue
            gw = num_patches // gh
            for cand_h, cand_w in ((gh, gw), (gw, gh)):
                score = abs((cand_h / max(cand_w, 1)) - target_ratio)
                if score < best_score:
                    best_score = score
                    best_pair = (cand_h, cand_w)
        if best_pair is None:
            raise ValueError(
                f"Cannot infer patch grid from num_patches={num_patches}"
            )
        return best_pair

    def _build_grid_adjacency(
        self,
        grid_h: int,
        grid_w: int,
        connectivity: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """构建空间网格的连通性邻接矩阵 (带行归一化 A_rw)."""
        device_index = -1 if device.index is None else device.index
        cache_key = (
            grid_h,
            grid_w,
            device.type,
            device_index,
            str(dtype),
            connectivity,
        )
        if cache_key in self._adj_cache:
            return self._adj_cache[cache_key]

        num_nodes = grid_h * grid_w
        adj = torch.zeros(num_nodes, num_nodes, device=device, dtype=dtype)

        # 依据不同的拓扑连通度配置空间近邻偏移量 (范围: 4-18)
        if connectivity == 4:
            # 曼哈顿距离为1的十字邻域 (上下左右)
            neighbors = (
                (-1,  0),
                ( 0, -1),          ( 0,  1),
                ( 1,  0),
            )
        elif connectivity == 6:
            # 4-连通 + 主对角线对称延伸
            neighbors = (
                (-1,  0),
                ( 0, -1),          ( 0,  1),
                ( 1,  0),
                (-1, -1),          ( 1,  1),
            )
        elif connectivity == 8:
            # 3x3 邻域（距离1的8个方向，不含中心）
            neighbors = (
                (-1, -1), (-1,  0), (-1,  1),
                ( 0, -1),          ( 0,  1),
                ( 1, -1), ( 1,  0), ( 1,  1),
            )
        elif connectivity == 10:
            # 8-邻域 + 距离2水平正交延伸
            neighbors = (
                (-1, -1), (-1,  0), (-1,  1),
                ( 0, -1),          ( 0,  1),
                ( 1, -1), ( 1,  0), ( 1,  1),
                (-2,  0), ( 2,  0),
            )
        elif connectivity == 12:
            # 10-连通 + 距离2垂直正交延伸（完整距离2十字）
            neighbors = (
                (-1, -1), (-1,  0), (-1,  1),
                ( 0, -1),          ( 0,  1),
                ( 1, -1), ( 1,  0), ( 1,  1),
                (-2,  0), ( 2,  0),
                ( 0, -2), ( 0,  2),
            )
        elif connectivity == 14:
            # 12-连通 + 主对角线距离2延伸（半对角扩展）
            neighbors = (
                (-1, -1), (-1,  0), (-1,  1),
                ( 0, -1),          ( 0,  1),
                ( 1, -1), ( 1,  0), ( 1,  1),
                (-2,  0), ( 2,  0),
                ( 0, -2), ( 0,  2),
                (-2, -2), ( 2,  2),
            )
        elif connectivity == 16:
            # 12-连通 + 完整距离2对角（4个方向，不含距离2十字）
            neighbors = (
                (-1, -1), (-1,  0), (-1,  1),
                ( 0, -1),          ( 0,  1),
                ( 1, -1), ( 1,  0), ( 1,  1),
                (-2,  0), ( 2,  0),
                ( 0, -2), ( 0,  2),
                (-2, -2), (-2,  2),
                ( 2, -2), ( 2,  2),
            )
        elif connectivity == 18:
            # 16-连通 + 马步(knight-move)对角延伸（高频纹理增强）
            neighbors = (
                (-1, -1), (-1,  0), (-1,  1),
                ( 0, -1),          ( 0,  1),
                ( 1, -1), ( 1,  0), ( 1,  1),
                (-2,  0), ( 2,  0),
                ( 0, -2), ( 0,  2),
                (-2, -2), (-2,  2),
                ( 2, -2), ( 2,  2),
                (-2, -1), ( 2,  1),
            )
        else:
            raise ValueError(
                f"Unsupported connectivity: {connectivity}. "
                f"Supported range: 4-18 (even numbers)."
            )

        for r in range(grid_h):
            for c in range(grid_w):
                src = r * grid_w + c
                adj[src, src] = 1.0  # 自环
                for dr, dc in neighbors:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_h and 0 <= nc < grid_w:
                        dst = nr * grid_w + nc
                        adj[src, dst] = 1.0

        # 行归一化 A_rw = D^(-1) * A
        degree = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        adj = adj / degree
        
        self._adj_cache[cache_key] = adj
        return adj

    def forward(self, im: torch.Tensor) -> Dict[str, torch.Tensor]:
        h, w = im.shape[2], im.shape[3]

        # 1. Encoder 特征提取与 Token 拆分
        x = self.encoder(im, return_features=True)
        cls_token = x[:, 0]
        num_extra_tokens = 1 + getattr(self.encoder, "distilled_num", 0)
        patch_tokens = x[:, num_extra_tokens:]
        
        
        # 2. 因果反事实干预投影 (消解全局偏置)
        #patch_tokens, orthogonal_loss = self.causal_projector(patch_tokens, cls_token)        
        
        '''
        B, N, _ = patch_tokens.shape
        grid_h, grid_w = self._infer_grid_size(h, w, N)
        
        # 3. 动态构建多尺度拓扑邻接矩阵组
        adjs = []
        for conn in self.graph_connectivities:
            adj = self._build_grid_adjacency(
                grid_h=grid_h,
                grid_w=grid_w,
                connectivity=conn,
                device=patch_tokens.device,
                dtype=patch_tokens.dtype,
            )
            adj = adj.unsqueeze(0).expand(B, -1, -1)
            adjs.append(adj)
            

        # 4. 多尺度切比雪夫图谱拓扑推理
        patch_tokens = self.graph_reasoner(patch_tokens, adjs) 
        '''
        
        
        # 5. Decoder 恢复分辨率并解码输出
        fin_patch_masks, _, _ = self.decoder(
            patch_tokens,
            (h, w),
            distilled=False,
        )

        fin_pix_masks = F.interpolate(
            fin_patch_masks,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        return {
            "fin_patch_masks": fin_patch_masks,
            "fin_pix_masks": fin_pix_masks,
            #"orthogonal_loss": orthogonal_loss
        }
