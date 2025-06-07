import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv # 使用 PyG 的 GATConv
# 如果使用 DGL，则导入 DGL 的相应 GNN 层
# import dgl.nn as dglnn
# import dgl

# 导入配置
try:
    from config import (
        INPUT_DIM, HIDDEN_DIM, GNN_LAYERS_PER_SOURCE, GNN_HEADS, # 导入 GNN_HEADS
        LSTM_HIDDEN_DIM, OUTPUT_REGRESSION_DIM, OUTPUT_CLASSIFICATION_DIM,
        GRAPH_SOURCES, # 导入图源列表
        DEVICE # 导入设备配置
    )
    # print("layers.py: 成功导入 config.py 配置。") # Keep config import print in dataset.py and train.py
except ImportError:
    print("layers.py: 错误：无法导入 config.py。请确保 config.py 文件存在且位于 Python 路径中。")
    # 如果无法导入 config，您需要在这里手动设置所有必要的配置变量或退出
    raise # 如果 config 无法导入，脚本应该停止


# --- 初始特征转换层 ---
# 将原始节点特征投影到 GNN 输入维度
class InitialFeatureTransform(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(InitialFeatureTransform, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        # 可以选择添加激活函数、Batch Normalization 或 Dropout
        # self.relu = nn.ReLU()
        # self.bn = nn.BatchNorm1d(output_dim) # Batch Norm for [N, output_dim] or [B*N, output_dim]
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: Input features [NumStocks, input_dim] or [BatchSize * NumStocks, input_dim]
        # Depending on how it's used in the main model's forward pass.
        # Assuming input is [B*N, input_dim] when used after flattening batch and stocks
        x = self.linear(x)
        # if hasattr(self, 'bn'): x = self.bn(x)
        # if hasattr(self, 'relu'): x = self.relu(x)
        # if hasattr(self, 'dropout'): x = self.dropout(x)
        return x # Output shape [BatchSize * NumStocks, output_dim]


# --- 单一图源的 GNN 层堆叠 ---
# 封装一个或多个 GATConv 层，处理一个图源的边信息
class SingleSourceGNN(nn.Module):
    # in_channels is the input dimension (from previous layer or initial features)
    # total_output_channels is the desired total output dimension of this block (per node)
    def __init__(self, in_channels, total_output_channels, num_layers, heads=1, dropout=0.0):
        super(SingleSourceGNN, self).__init__()
        self.num_layers = num_layers
        self.heads = heads

        # Calculate output channels per head for GATConv
        # total_output_channels = per_head_output_channels * heads (assuming concat=True)
        # so, per_head_output_channels = total_output_channels // heads
        # Check if total_output_channels is divisible by heads
        if total_output_channels % heads != 0:
            print(f"警告: SingleSourceGNN 初始化时，总输出通道数 ({total_output_channels}) 不能被头数 ({heads}) 整除。这可能导致 GATConv 输出维度问题。")
            # Decide how to handle this - maybe raise error or adjust total_output_channels
            # For now, proceed with integer division, but be aware of potential size issues.
            # A common practice is to ensure hidden_dim is divisible by heads.

        gat_out_channels_per_head = total_output_channels // heads


        self.convs = nn.ModuleList()
        for i in range(num_layers):
            # Input to the first layer is `in_channels`.
            # Input to subsequent layers is the concatenated output of the previous layer: `gat_out_channels_per_head * heads`.
            layer_in_channels = in_channels if i == 0 else gat_out_channels_per_head * self.heads
            layer_out_channels = gat_out_channels_per_head # GATConv takes per-head output size

            self.convs.append(GATConv(layer_in_channels, layer_out_channels, heads=self.heads, dropout=dropout))

        # The final output dimension is gat_out_channels_per_head * self.heads, which should be equal to total_output_channels.


    # forward method remains the same
    def forward(self, x, edge_index, edge_weight=None):
        # x: Node features [TotalNodesInBatch, in_channels]
        # edge_index: Edge indices [2, NumEdges] (for a batched graph)
        # edge_weight: Edge weights [NumEdges,] (Optional, pass to GATConv's edge_attr parameter based on TypeError)

        # Check if x is empty (can happen with empty batches or no valid nodes)
        if x.numel() == 0:
             # Return an empty tensor with the expected output dimension
             # Expected output dim after all layers is gat_out_channels_per_head * self.heads (if num_layers > 0) or in_channels (if num_layers == 0)
             # Need to determine the output dim if convs is empty
             # Reference the output dim based on init calculation
             # Corrected output_dim calculation for num_layers == 0
             output_dim = (self.convs[-1].out_channels * self.heads) if len(self.convs) > 0 else x.size(-1)


             # Return empty tensor with shape [0, output_dim] and same device/dtype as input x
             return torch.empty(0, output_dim, device=x.device, dtype=x.dtype)


        # print(f"      SingleSourceGNN forward - Input x shape: {x.shape}, dtype: {x.dtype}, device: {x.device}") # Debug print

        h = x
        for i, conv in enumerate(self.convs):
            # print(f"        SingleSourceGNN layer {i} - Input h shape (before conv): {h.shape}, dtype: {h.dtype}, device: {h.device}") # Debug print
            # print(f"        SingleSourceGNN layer {i} - edge_index shape: {edge_index.shape}, dtype: {edge_index.dtype}, device: {edge_index.device}") # Debug print
            # if edge_weight is not None:
            #     print(f"        SingleSourceGNN layer {i} - edge_weight shape: {edge_weight.shape}, dtype: {edge_weight.dtype}, device: {edge_weight.device}") # Debug print
            # else:
            # print(f"        SingleSourceGNN layer {i} - edge_weight is None")

            # Pass the received edge_weight parameter value to GATConv's edge_attr parameter
            # Based on the observed TypeError, this is the parameter GATConv expects for edge information in this version.
            h = conv(h, edge_index, edge_attr=edge_weight) # Pass edge_weight as edge_attr

            # print(f"        SingleSourceGNN layer {i} - Output h shape (after conv): {h.shape}, dtype: {h.dtype}, device: {h.device}") # Debug print


            if i < self.num_layers - 1:
                h = F.relu(h) # Using ReLU activation
                # Dropout is already handled by GATConv's internal dropout parameter
                # print(f"        SingleSourceGNN layer {i} - Output h shape (after activation): {h.shape}, dtype: {h.dtype}, device: {h.device}") # Debug print


        # print(f"      SingleSourceGNN forward - Final output h shape: {h.shape}, dtype: {h.dtype}, device: {h.device}") # Debug print
        return h # Output shape [TotalNodesInBatch, out_channels * heads]


# --- 多图 GNN 块 (处理一个时间步内的所有图源) ---
# 这个模块处理一个时间步的数据，包括初始特征转换和并行 GNN
class MultiSourceGNNBlock(nn.Module):
    # 这里的 gnn_heads 参数是从 config 导入并传递进来的 GNN_HEADS
    def __init__(self, input_dim, hidden_dim, gnn_layers_per_source, gnn_heads, graph_sources):
        super(MultiSourceGNNBlock, self).__init__()
        self.graph_sources = graph_sources
        self.input_dim = input_dim # Store input_dim
        self.hidden_dim = hidden_dim # This is the input dim to SingleSourceGNN (output of InitialTransform)
        self.gnn_layers_per_source = gnn_layers_per_source # Store for reference
        self.gnn_heads = gnn_heads # Storage for the number of heads

        # InitialTransform is now handled before this block in the main model

        # 并行单一图源 GNN 层 (使用 ModuleDict 存储)
        self.source_gnns = nn.ModuleDict()
        for source in graph_sources:
            # Each source GNN takes hidden_dim as input and outputs hidden_dim * gnn_heads
            # Pass hidden_dim as in_channels
            # Pass hidden_dim * self.gnn_heads as total_output_channels (desired output)
            # SingleSourceGNN will calculate its internal per-head out_channels based on this
            self.source_gnns[source] = SingleSourceGNN(
                self.hidden_dim,             # in_channels (input from InitialTransform)
                self.hidden_dim * self.gnn_heads, # total_output_channels (desired total output dimension)
                self.gnn_layers_per_source,  # num_layers
                heads=self.gnn_heads,        # heads for GATConv
                dropout=0.0 # GATConv handles dropout internally
            )
        # print(f"MultiSourceGNNBlock initialized with {len(self.graph_sources)} sources, input_dim_to_gnn={self.hidden_dim}, output_dim_per_source={self.hidden_dim * self.gnn_heads}")


    def forward(self, h0_flat, graphs_batch_t, batch_size, num_stocks):
        # h0_flat: Initial node features for time step t flattened across batch [BatchSize * NumStocks, HiddenDim]
        # h0_flat is the output of InitialFeatureTransform for time step t
        # graphs_batch_t: List of dictionaries for time step t across batch [BatchSize, {source: {edge_index, edge_weight} or None}]
        # batch_size: Actual batch size
        # num_stocks: Number of stocks (nodes) per graph

        # --- 调试打印 (MultiSourceGNNBlock forward entry) ---
        # print(f"\nMultiSourceGNNBlock forward - Batch Size: {batch_size}, Num Stocks: {num_stocks}")
        # print(f"  Input h0_flat shape: {h0_flat.shape}, dtype: {h0_flat.dtype}, device: {h0_flat.device}")
        # --- 调试打印结束 ---

        h_sources_flat = {} # {source: Tensor [BatchSize * NumStocks, HiddenDim * Heads]}

        # Process each graph source
        for source in self.graph_sources:
            # Collect graph data for this source across the batch
            # List of {edge_index, edge_weight} or None for this source at time t across the batch
            source_graph_data_list_t = [item.get(source) for item in graphs_batch_t]

            # Filter out None graph data and record which batch indices are valid
            valid_graph_data = []
            valid_batch_indices = []
            for b, data in enumerate(source_graph_data_list_t):
                 # Check if data exists and edge_index is not None and not empty
                 if data is not None and 'edge_index' in data and data['edge_index'] is not None and data['edge_index'].numel() > 0:
                      valid_graph_data.append(data)
                      valid_batch_indices.append(b)

            # --- 调试打印 (Valid Batches per Source) ---
            # print(f"  Source: {source}, Valid Batches Indices: {valid_batch_indices}")
            # --- 调试打印结束 ---


            # If there is valid graph data for this source in the batch
            if valid_graph_data:
                 # --- Manually batch graph data ---
                 batched_edge_index_list = []
                 batched_edge_weight_list = [] # edge_weight is passed as edge_attr

                 # --- Remap edge_index indices to be relative to the valid batches tensor ---
                 # The total number of nodes in the batched graph (from valid samples)
                 total_nodes_in_batched_graph = len(valid_batch_indices) * num_stocks
                 # print(f"    Source: {source}, Total nodes in batched graph (from valid samples): {total_nodes_in_batched_graph}") # Debug print


                 for i, data in enumerate(valid_graph_data):
                     edge_index = data['edge_index'].to(h0_flat.device) # Ensure edge_index on same device as features
                     edge_weight = data.get('edge_weight', None)
                     if edge_weight is not None:
                          edge_weight = edge_weight.to(h0_flat.device) # Ensure edge_weight on same device as features

                     # --- 调试打印 (Original edge_index and offset) ---
                     # print(f"    Source: {source}, Valid Batch List Index: {i}, Original Batch Index: {valid_batch_indices[i]}")
                     if edge_index.numel() > 0:
                          # print(f"      Original edge_index shape: {edge_index.shape}, min: {edge_index.min()}, max: {edge_index.max()}")
                          # Original edge_index indices should be within [0, num_stocks - 1] for a single graph
                          if edge_index.numel() > 0 and (edge_index.min() < 0 or edge_index.max() >= num_stocks):
                              print(f"      !!! 警告: 原始 edge_index 索引超出单个图的节点范围 [0, {num_stocks-1}] (min: {edge_index.min()}, max: {edge_index.max()}) !!!")
                              # If original indices are bad, skip this graph data
                              # This prevents attempting to remap invalid original indices
                              # Decide how to handle this - print warning and skip the sample's graph for this source/timestep
                              print(f"      跳过此样本({valid_batch_indices[i]})的图数据 ({source}) 由于原始 edge_index 索引无效.")
                              continue # Skip the rest of the loop for this valid sample index i

                     else: # edge_index is empty
                          # print(f"      Original edge_index is empty.")
                          # If edge_index is empty, it's a valid case, just no edges for this sample/source/day.
                          # We can still proceed, remapped_edge_index will be empty.
                          pass # No need to print empty edge_index warning unless debugging data loading issues

                     if edge_weight is not None:
                         # print(f"      Original edge_weight shape: {edge_weight.shape}, dtype: {edge_weight.dtype}")
                         pass # No need to print edge_weight shape unless debugging
                     else:
                         # print(f"      Original edge_weight is None.")
                         pass # No need to print None unless debugging
                     # --- 调试打印结束 ---

                     # --- Apply Remapping ---
                     # Remap edge_index indices (0 to num_stocks-1 for each sample)
                     # to be relative to the flattened tensor of features from valid samples
                     # Node index k in sample at valid_list_index i maps to i * num_stocks + k
                     if edge_index.numel() > 0: # Only remap if edge_index is not empty
                          # Create mapping tensor: [i*num_stocks, i*num_stocks+1, ..., (i+1)*num_stocks-1]
                          remap_indices = torch.arange(i * num_stocks, (i + 1) * num_stocks, device=edge_index.device).long()
                          # Apply the mapping: old_index -> remap_indices[old_index]
                          remapped_edge_index = remap_indices[edge_index] # This applies the mapping element-wise

                     else: # edge_index is empty
                          remapped_edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)


                     # --- 调试打印 (Remapped edge_index) ---
                     if remapped_edge_index.numel() > 0:
                          # print(f"      Remapped edge_index shape: {remapped_edge_index.shape}, min: {remapped_edge_index.min()}, max: {remapped_edge_index.max()}")
                          # Check for invalid indices in the remapped edge_index (should be >= 0 and < total_nodes_in_batched_graph)
                          if remapped_edge_index.numel() > 0:
                              min_idx_remapped = remapped_edge_index.min()
                              max_idx_remapped = remapped_edge_index.max()
                              if min_idx_remapped < 0:
                                   print(f"      !!! 警告: Remapped edge_index 的最小索引小于 0 ({min_idx_remapped}) !!!")
                              # The max index must be strictly less than the total number of nodes it can address (total_nodes_in_batched_graph)
                              if max_idx_remapped >= total_nodes_in_batched_graph:
                                   print(f"      !!! 警告: Remapped edge_index 的最大索引 ({max_idx_remapped}) 超出批处理图的总节点数 ({total_nodes_in_batched_graph}) !!!")


                     else:
                           # print(f"      Remapped edge_index is empty.")
                           pass # No need to print empty remapped edge_index warning unless debugging
                     # --- 调试打印结束 ---


                     batched_edge_index_list.append(remapped_edge_index)
                     if edge_weight is not None: # Edge weight doesn't need index remapping, just concatenation
                          batched_edge_weight_list.append(edge_weight)

                 # Concatenate remapped edge indices and weights for the batched graph
                 batched_edge_index = torch.cat(batched_edge_index_list, dim=1) if batched_edge_index_list else torch.empty((2, 0), dtype=torch.long, device=h0_flat.device)
                 batched_edge_weight = torch.cat(batched_edge_weight_list, dim=0) if batched_edge_weight_list else None # batched_edge_weight might be None


                 # --- 调试打印 (Batched edge_index and Total Nodes) ---
                 # print(f"  Source: {source}, Batched edge_index shape: {batched_edge_index.shape}")
                 # print(f"    Total nodes in batched graph (from valid samples): {total_nodes_in_batched_graph}")

                 if batched_edge_index.numel() > 0:
                      # min_idx = batched_edge_index.min()
                      # max_idx = batched_edge_index.max()
                      # print(f"    Batched edge_index min: {min_idx}, max: {max_idx}")
                      pass # No need to print min/max unless debugging


                      # Final check for invalid indices in the batched edge_index
                      # if min_idx < 0:
                      #      print("!!! 警告: batched_edge_index 的最终最小索引小于 0 !!!")
                      # # The max index must be strictly less than the total number of nodes it can address (total_nodes_in_batched_graph)
                      # if max_idx >= total_nodes_in_batched_graph:
                      #      print(f"!!! 警告: batched_edge_index 的最终最大索引 ({max_idx}) 超出批处理图的总节点数 ({total_nodes_in_batched_graph}) !!!")
                 else:
                      # print("    Batched edge_index is empty.")
                      pass # No need to print empty batched edge_index unless debugging

                 if batched_edge_weight is not None:
                      # print(f"  Source: {source}, Batched edge_weight shape: {batched_edge_weight.shape}, dtype: {batched_edge_weight.dtype}")
                      pass # No need to print batched edge_weight shape unless debugging
                 else:
                      # print(f"  Source: {source}, Batched edge_weight is None.")
                      pass # No need to print None unless debugging

                 # --- 调试打印结束 ---


                 # --- Filter initial features for valid batches ---
                 # Get h0_flat for only the batches that had valid graph data for this source
                 h0_t_batch = h0_flat.view(batch_size, num_stocks, -1) # Reshape flattened input to [B, N, H_in]
                 h0_t_valid_batches = h0_t_batch[valid_batch_indices] # Select valid batches [NumValidBatches, N, H_in]
                 h0_t_valid_batches_flat = h0_t_valid_batches.view(-1, self.hidden_dim) # Flatten again [NumValidBatches * N, H_in]

                 # --- 调试打印 (Features for Valid Batches) ---
                 # print(f"  Source: {source}, h0_t_valid_batches_flat shape (input to SingleSourceGNN): {h0_t_valid_batches_flat.shape}, dtype: {h0_t_valid_batches_flat.dtype}, device: {h0_t_valid_batches_flat.device}") # Debug print
                 # --- 调试打印结束 ---

                 # --- Apply GNN layer(s) for this source ---
                 # Pass features for valid nodes (h0_t_valid_batches_flat) and batched graph data (batched_edge_index, batched_edge_weight) to GNN
                 # The GNN layer will operate on h0_t_valid_batches_flat and batched_edge_index/weight.
                 # Its internal scatter operations should use h0_t_valid_batches_flat.size(0) as the target dimension size.
                 # h_source_valid_flat: GNN output for nodes in valid batches [NumValidBatches * N, HiddenDim * Heads]
                 # Ensure h0_t_valid_batches_flat is on the correct device (it should be, as it comes from h0_flat which is on device)
                 # Pass batched_edge_weight to SingleSourceGNN as edge_weight parameter
                 # Based on the observed TypeError, pass batched_edge_weight to SingleSourceGNN's edge_weight parameter,
                 # and inside SingleSourceGNN, pass it as edge_attr to GATConv.

                 # print(f"  Source: {source}, Calling SingleSourceGNN...") # Debug print before calling SingleSourceGNN
                 h_source_valid_flat = self.source_gnns[source](h0_t_valid_batches_flat, batched_edge_index, edge_weight=batched_edge_weight) # Pass batched_edge_weight using edge_weight keyword
                 # print(f"  Source: {source}, SingleSourceGNN returned h_source_valid_flat shape: {h_source_valid_flat.shape}, dtype: {h_source_valid_flat.dtype}, device: {h_source_valid_flat.device}") # Debug print after calling SingleSourceGNN


                 # --- Scatter results back into a full zero tensor [BatchSize * NumStocks, HiddenDim * Heads] ---
                 # Create a zero tensor for the full batch (including samples with no graph data for this source)
                 # Correct the full_output_shape feature dimension to match SingleSourceGNN output
                 # The output dimension of SingleSourceGNN should be self.hidden_dim * self.gnn_heads based on init
                 full_output_feature_dim = self.hidden_dim * self.gnn_heads # This should be the output dim of SingleSourceGNN
                 full_output_shape = (batch_size * num_stocks, full_output_feature_dim)
                 # >>> h_source_full_flat 在这里被定义和初始化为全零 Tensor <<<
                 # Use device and dtype from h_source_valid_flat as a reference
                 h_source_full_flat = torch.zeros(full_output_shape, dtype=h_source_valid_flat.dtype, device=h_source_valid_flat.device) # Use device/dtype from h_source_valid_flat

                 # Need indices to scatter the valid results back to the full batch tensor
                 # These indices map the flattened nodes in h_source_valid_flat (range [0, NumValidBatches*N-1])
                 # back to their original positions in the full batch tensor (range [0, BatchSize*N-1]).
                 # We use the original batch indices to construct these scatter target indices.
                 full_batch_indices = []
                 for b_idx in valid_batch_indices:
                     full_batch_indices.extend(range(b_idx * num_stocks, (b_idx + 1) * num_stocks))
                 # Convert to tensor on the correct device
                 full_batch_indices_tensor = torch.tensor(full_batch_indices, dtype=torch.long, device=h_source_valid_flat.device) # Use device from valid data


                 # --- Debug Print Before Scatter (Expanded) ---
                 # print(f"    Source: {source}, Before scatter - h_source_full_flat shape: {h_source_full_flat.shape}, dtype: {h_source_full_flat.dtype}, device: {h_source_full_flat.device}")
                 # print(f"    Source: {source}, Before scatter - h_source_valid_flat shape: {h_source_valid_flat.shape}, dtype: {h_source_valid_flat.dtype}, device: {h_source_valid_flat.device}")
                 # print(f"    Source: {source}, Before scatter - full_batch_indices_tensor shape: {full_batch_indices_tensor.shape}, dtype: {full_batch_indices_tensor.dtype}, device: {full_batch_indices_tensor.device}")
                 # if full_batch_indices_tensor.numel() > 0:
                     # print(f"    Source: {source}, Before scatter - full_batch_indices_tensor min: {full_batch_indices_tensor.min()}, max: {full_batch_indices_tensor.max()}")
                     # Print first few indices
                     # print(f"    Source: {source}, Before scatter - full_batch_indices_tensor[:10]: {full_batch_indices_tensor[:10]}")
                 # --- End Debug Print Before Scatter (Expanded) ---


                 # Perform the scatter operation using index_copy_
                 # Copies elements from h_source_valid_flat into h_source_full_flat at the indices specified by full_batch_indices_tensor
                 # The '0' dimension is the target dimension (node dimension)
                 if h_source_valid_flat.numel() > 0: # Only scatter if there's data to scatter
                      # Check if indices are valid for scattering target (h_source_full_flat)
                      if full_batch_indices_tensor.numel() > 0:
                           min_idx_full = full_batch_indices_tensor.min()
                           max_idx_full = full_batch_indices_tensor.max()
                           if min_idx_full < 0 or max_idx_full >= full_output_shape[0]:
                               print(f"!!! 警告: Scatter target indices out of bounds for h_source_full_flat shape {full_output_shape} (min: {min_idx_full}, max: {max_idx_full}) !!!")
                               # Decide how to handle - potentially skip scattering for this source/batch/timestep
                               # For now, print warning and proceed, but this indicates a problem in index construction
                           else:
                                # print(f"    Source: {source}, Attempting index_copy_...") # Debug print before scatter
                                h_source_full_flat.index_copy_(0, full_batch_indices_tensor, h_source_valid_flat)
                                # print(f"    Source: {source}, index_copy_ successful.") # Debug print after scatter
                      else: print("警告: full_batch_indices_tensor is empty, skipping scatter.")


                 # else: h_source_full_flat remains all zeros


                 # 将当前图源的全批次输出存储到字典中
                 # 这里使用了 h_source_full_flat，将其赋值给 h_sources_flat[source]
                 h_sources_flat[source] = h_source_full_flat # <<< 注意，这里是 h_source_full_flat

            else:
                # If no valid graph data for this source in the entire batch at this timestep
                # Create a zero tensor for the full batch with the correct shape and device
                # Use the correct full output feature dimension
                full_output_feature_dim = self.hidden_dim * self.gnn_heads # This should be the output dim of SingleSourceGNN
                full_output_shape = (batch_size * num_stocks, full_output_feature_dim)
                # Use the device and dtype from h0_flat as a reference
                h_sources_flat[source] = torch.zeros(full_output_shape, device=h0_flat.device, dtype=h0_flat.dtype)
                # print(f"警告：时间步图源 {source} 没有有效数据在整个批次中，使用零填充 GNN 输出。")


        return h_sources_flat # Output {source: Tensor [BatchSize * NumStocks, HiddenDim * Heads]}


# --- 多图融合层 (注意力机制) ---
# 融合来自不同图源的节点表示
class GraphFusionLayer(nn.Module):
    # fusion_output_dim 应该是 LSTM 的输入维度
    # Input dim per source is HiddenDim * GNN_HEADS
    def __init__(self, input_dim_per_source, num_sources, fusion_output_dim):
        super(GraphFusionLayer, self).__init__()
        self.input_dim_per_source = input_dim_per_source # Output dim of SingleSourceGNN (HiddenDim * Heads)
        self.num_sources = num_sources # Number of graph sources
        self.fusion_output_dim = fusion_output_dim # Desired output dimension after fusion (LSTM_HIDDEN_DIM)


        # Attention mechanism: learn per-node, per-source importance
        # Using a simple additive attention like mechanism or projecting to scores
        # Project each source's representation to a scalar score per node
        self.source_attention_scores = nn.ModuleDict()
        # Use the global list GRAPH_SOURCES to ensure all sources are covered
        # The input_dim_per_source is the same for all sources (HiddenDim * GNN_HEADS)
        for source in GRAPH_SOURCES:
             # Linear layer: Input [B*N, input_dim_per_source] -> Output [B*N, 1] (scalar score per node per source)
             self.source_attention_scores[source] = nn.Linear(input_dim_per_source, 1)

        # Linear layer to potentially transform the fused representation
        # Input dimension is input_dim_per_source (after weighted sum), Output dimension should match LSTM input dimension
        self.fusion_transform = nn.Linear(input_dim_per_source, fusion_output_dim)


    def forward(self, h_sources_flat_dict, batch_size, num_stocks): # Pass batch_size and num_stocks here
        # h_sources_flat_dict: Dictionary {source: Tensor [BatchSize * NumStocks, InputDimPerSource]}
        # InputDimPerSource is the output dimension of SingleSourceGNN (HiddenDim * Heads)
        # batch_size: Actual batch size from DataLoader
        # num_stocks: Number of stocks (nodes) per graph

        # Determine total_nodes_in_batch from batch_size and num_stocks
        total_nodes_in_batch = batch_size * num_stocks

        # Handle case where input dict might be empty (should not happen if MultiSourceGNNBlock works as expected)
        if not h_sources_flat_dict:
            print("警告：融合层输入字典为空。返回全零张量。")
            # Return a zero tensor with the expected output shape and device/dtype from h0_flat (passed to MultiSourceGNNBlock)
            # Need device/dtype... Let's assume calling code ensures device/dtype consistency
            return torch.zeros((total_nodes_in_batch, self.fusion_output_dim), device=DEVICE, dtype=torch.float32) # Use default DEVICE


        # Calculate attention scores for each source representation per node
        # Handle case where a source's tensor might be None or empty - assign a score of -inf or a very small number
        all_source_scores_list = []
        # Need a reference device/dtype for creating tensors when source data is missing
        reference_device = None
        reference_dtype = None

        # Determine reference device/dtype from the first valid tensor found
        for source in GRAPH_SOURCES:
             if source in h_sources_flat_dict and h_sources_flat_dict[source] is not None and h_sources_flat_dict[source].numel() > 0:
                  reference_device = h_sources_flat_dict[source].device
                  reference_dtype = h_sources_flat_dict[source].dtype
                  break # Found the reference, break the loop

        # If no valid tensor found, use default DEVICE and dtype
        if reference_device is None:
             reference_device = DEVICE
             reference_dtype = torch.float32


        for source in GRAPH_SOURCES:
             if source in h_sources_flat_dict and \
                h_sources_flat_dict[source] is not None and \
                h_sources_flat_dict[source].numel() > 0 and \
                h_sources_flat_dict[source].size(0) == total_nodes_in_batch and \
                h_sources_flat_dict[source].size(-1) == self.input_dim_per_source: # Check size and dimension

                  # Ensure linear layer for this source exists
                  if source in self.source_attention_scores:
                       # Pass tensor to linear layer to get scores [B*N, 1]
                       # Ensure input tensor is on the correct device/dtype
                       input_tensor = h_sources_flat_dict[source].to(device=reference_device, dtype=reference_dtype)
                       all_source_scores_list.append(self.source_attention_scores[source](input_tensor))
                  else:
                       print(f"警告：融合层缺少图源 {source} 的注意力评分层。")
                       # Assign a very small score (effectively zero weight after softmax)
                       all_source_scores_list.append(torch.full((total_nodes_in_batch, 1), -float('inf'), device=reference_device, dtype=reference_dtype))

             else:
                  # If source tensor is None, empty, or size/dimension incorrect
                  # Assign a very small score (effectively zero weight after softmax)
                  # Ensure the score tensor has the correct number of nodes (total_nodes_in_batch)
                  all_source_scores_list.append(torch.full((total_nodes_in_batch, 1), -float('inf'), device=reference_device, dtype=reference_dtype))


        # Stack scores along a new dimension (dim=1): [BatchSize * NumStocks, NumSources, 1]
        if not all_source_scores_list:
             # This should ideally not happen if GRAPH_SOURCES is not empty
             print("警告：融合层注意力评分列表为空。返回全零张量。")
             return torch.zeros((total_nodes_in_batch, self.fusion_output_dim), device=reference_device, dtype=reference_dtype)

        # Check if all tensors in all_source_scores_list have the same size(0)
        if not all(score.size(0) == total_nodes_in_batch for score in all_source_scores_list):
             print("错误：融合层注意力评分张量节点数不一致！返回空张量。")
             return torch.empty(0, self.fusion_output_dim, device=reference_device, dtype=reference_dtype)


        stacked_scores = torch.stack(all_source_scores_list, dim=1) # [B*N, NumSources, 1]

        # Apply softmax over the source dimension (dim=1) [BatchSize * NumStocks, NumSources, 1]
        attention_weights = F.softmax(stacked_scores, dim=1)

        # Collect all source representations (even if they were zero tensors from MultiSourceGNNBlock)
        # Ensure the tensors exist, are not None, and have the correct expected input dimension and total_nodes_in_batch size
        all_source_h_list = []
        # Reference device/dtype is already determined above


        for source in GRAPH_SOURCES:
             if source in h_sources_flat_dict and \
                h_sources_flat_dict[source] is not None and \
                h_sources_flat_dict[source].numel() > 0 and \
                h_sources_flat_dict[source].size(0) == total_nodes_in_batch and \
                h_sources_flat_dict[source].size(-1) == self.input_dim_per_source: # Check size and dimension

                  # Ensure on reference device/dtype
                  all_source_h_list.append(h_sources_flat_dict[source].to(device=reference_device, dtype=reference_dtype)) # [B*N, H_in]
             else:
                  # If source tensor is None, empty, or size/dimension incorrect, append a zero tensor with correct shape
                  all_source_h_list.append(torch.zeros((total_nodes_in_batch, self.input_dim_per_source), device=reference_device, dtype=reference_dtype)) # Append zero tensor [B*N, H_in]


        # Stack all source representations along a new dimension (dim=1): [BatchSize * NumStocks, NumSources, InputDimPerSource]
        if not all_source_h_list:
             print("警告：融合层源表示列表为空。返回全零张量。")
             return torch.zeros((total_nodes_in_batch, self.fusion_output_dim), device=reference_device, dtype=reference_dtype)

        # Check if all tensors in all_source_h_list have the same size(0) and size(1)
        if not all(h.size() == (total_nodes_in_batch, self.input_dim_per_source) for h in all_source_h_list):
             print("错误：融合层源表示张量形状不一致！返回空张量。")
             return torch.empty(0, self.fusion_output_dim, device=reference_device, dtype=reference_dtype)


        stacked_h_all = torch.stack(all_source_h_list, dim=1) # [B*N, NumSources, InputDimPerSource]

        # Apply weights: [B*N, NumSources, 1] * [B*N, NumSources, H_in] -> [B*N, NumSources, H_in] (element-wise multiplication with broadcasting)
        # attention_weights is [B*N, NumSources, 1], stacked_h_all is [B*N, NumSources, H_in] - multiplication works due to broadcasting
        weighted_h = attention_weights * stacked_h_all

        # Sum across the source dimension (dim=1): [BatchSize * NumStocks, InputDimPerSource]
        fused_h_before_transform = torch.sum(weighted_h, dim=1)

        # Apply final fusion transform
        # Input is [B*N, InputDimPerSource], Output is [B*N, FusionOutputDim]
        fused_h_after_transform = self.fusion_transform(fused_h_before_transform)

        return fused_h_after_transform # Output shape [BatchSize * NumStocks, FusionOutputDim]


# --- Output Heads ---
class OutputHeads(nn.Module):
    # in_features is the output dimension of the LSTM (LSTM_HIDDEN_DIM)
    def __init__(self, in_features, regression_dim, classification_dim):
        super(OutputHeads, self).__init__()
        # Regression head: [B*N, in_features] -> [B*N, regression_dim]
        self.regression_head = nn.Linear(in_features, regression_dim)
        # Classification head: [B*N, in_features] -> [B*N, classification_dim]
        self.classification_head = nn.Linear(in_features, classification_dim)

    def forward(self, h):
        # h: Input hidden state from LSTM/GRU for the prediction timestep [BatchSize * NumStocks, in_features]
        regression_output = self.regression_head(h) # [BatchSize * NumStocks, regression_dim]
        classification_output = self.classification_head(h) # [BatchSize * NumStocks, classification_dim]
        return regression_output, classification_output