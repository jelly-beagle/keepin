import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入自定义层和配置
from layers import InitialFeatureTransform, SingleSourceGNN, MultiSourceGNNBlock, GraphFusionLayer, OutputHeads
from config import (
    INPUT_DIM, HIDDEN_DIM, GNN_LAYERS_PER_SOURCE, GNN_HEADS,
    LSTM_LAYERS, LSTM_HIDDEN_DIM, OUTPUT_REGRESSION_DIM, OUTPUT_CLASSIFICATION_DIM,
    GRAPH_SOURCES, TIME_WINDOW_SIZE, DROPOUT_RATE,PREDICTION_TIME_STEP, DEVICE
)

# --- 完整 GNN 预测模型 ---
class StockPredictorGNN(nn.Module):
    def __init__(self, num_stocks, dropout_rate=DROPOUT_RATE):
        super(StockPredictorGNN, self).__init__()
        self.num_stocks = num_stocks # 节点数量 (股票数量)
        self.time_window_size = TIME_WINDOW_SIZE
        self.prediction_time_step = PREDICTION_TIME_STEP
        self.graph_sources = GRAPH_SOURCES

        # 初始特征转换层
        # 将原始输入特征 INPUT_DIM 转换为 HIDDEN_DIM
        self.initial_transform = InitialFeatureTransform(INPUT_DIM, HIDDEN_DIM)
        # 添加 Dropout 层 1
        self.dropout1 = nn.Dropout(dropout_rate)

        # 多图 GNN 块 (处理一个时间步内的所有图源)
        # 输入是 InitialFeatureTransform 的输出 (HIDDEN_DIM)
        # 输出是每个图源的 GNN 输出 (HIDDEN_DIM * GNN_HEADS)
        self.multi_source_gnn = MultiSourceGNNBlock(
            INPUT_DIM, # Although InitialTransform is separate, this block conceptually takes raw input dim for clarity or future use
            HIDDEN_DIM, # Input dimension to the GNN layers within the block (output of InitialTransform)
            GNN_LAYERS_PER_SOURCE,
            GNN_HEADS,
            GRAPH_SOURCES
        )

        # 图融合层
        # 输入是每个图源 GNN 的输出维度 (HIDDEN_DIM * GNN_HEADS)
        # 输出是融合后的节点表示维度，设置为 LSTM 的输入维度
        self.fusion_layer = GraphFusionLayer(
            HIDDEN_DIM * GNN_HEADS, # Input dimension per source
            len(GRAPH_SOURCES),     # Number of sources
            LSTM_HIDDEN_DIM         # Output dimension after fusion
        )
        self.dropout2 = nn.Dropout(dropout_rate)


        # 时序建模层 (LSTM)
        # LSTM 的输入是每个时间步融合后的节点表示序列 [BatchSize * NumStocks, TimeWindow, LSTM_HIDDEN_DIM]
        self.lstm = nn.LSTM(input_size=LSTM_HIDDEN_DIM, # LSTM 输入维度是融合后的维度
                            hidden_size=LSTM_HIDDEN_DIM, # LSTM 隐藏层维度
                            num_layers=LSTM_LAYERS, # LSTM 层数
                            batch_first=True) # 批次维度在前 [BatchSize * NumStocks, TimeWindow, Features]
        self.dropout3 = nn.Dropout(dropout_rate)


        # 输出层
        # 输入是 LSTM 在预测时间步的输出 (LSTM_HIDDEN_DIM)
        self.output_heads = OutputHeads(LSTM_HIDDEN_DIM, OUTPUT_REGRESSION_DIM, OUTPUT_CLASSIFICATION_DIM)

    def forward(self, batch_features, batch_labels, batch_graphs_nested_list):
        # batch_features: Tensor [BatchSize, TimeWindow, NumStocks, INPUT_DIM]
        # batch_labels: Tensor [BatchSize, TimeWindow, NumStocks, NumLabels] - Used for loss calculation
        # batch_graphs_nested_list: [BatchSize, TimeWindow, {source: {edge_index, edge_weight} or None}]

        batch_size = batch_features.size(0)
        num_stocks = batch_features.size(2)

        # --- Process each time step in the window ---
        fused_features_time_sequence = [] # List of Tensors [BatchSize * NumStocks, LSTM_HIDDEN_DIM] for each time step

        for t in range(self.time_window_size):
            # Get features for time step t across the batch and flatten [BatchSize * NumStocks, INPUT_DIM]
            features_t_flat = batch_features[:, t, :, :].reshape(-1, INPUT_DIM)

            # Apply initial feature transformation [BatchSize * NumStocks, HIDDEN_DIM]
            h0_t_flat = self.initial_transform(features_t_flat)
            # 应用 Dropout 1
            h0_t_flat = self.dropout1(h0_t_flat)

            # Get graph data for time step t across the batch: list of dicts [BatchSize, {source: {ei,ew}|None}]
            graphs_t_batch = [batch_graphs_nested_list[b][t] for b in range(batch_size)]

            # --- Multi-Source GNN Block and Fusion for time step t ---
            # Process all graphs for current time step across the batch
            # This block handles graph batching internally and applies GNNs
            # Output h_sources_flat_dict: {source: Tensor [BatchSize * NumStocks, HiddenDim * Heads]}
            h_sources_t_flat_dict = self.multi_source_gnn(h0_t_flat, graphs_t_batch, batch_size, num_stocks)


            # Fuse the outputs from different graph sources
            # Input: {source: Tensor [BatchSize * NumStocks, HiddenDim * Heads]}
            # Output: Tensor [BatchSize * NumStocks, LSTM_HIDDEN_DIM]
            fused_h_t_flat = self.fusion_layer(h_sources_t_flat_dict, batch_size=batch_size, num_stocks=num_stocks)
            # 应用dropout2
            fused_h_t_flat = self.dropout2(fused_h_t_flat)

            # Append fused features for this time step
            fused_features_time_sequence.append(fused_h_t_flat)

        # --- Stack fused features over the time window ---
        # List of [BatchSize * NumStocks, LSTM_HIDDEN_DIM] tensors -> Stack to [TimeWindow, BatchSize * NumStocks, LSTM_HIDDEN_DIM]
        # Transpose to [BatchSize * NumStocks, TimeWindow, LSTM_HIDDEN_DIM] for LSTM batch_first=True
        lstm_input = torch.stack(fused_features_time_sequence, dim=0).transpose(0, 1).contiguous()

        # --- LSTM Processing ---
        # lstm_input shape: [BatchSize * NumStocks, TimeWindow, LSTM_HIDDEN_DIM]
        # output shape: [BatchSize * NumStocks, TimeWindow, LSTM_HIDDEN_DIM]
        # hn, cn shape: [NumLayers, BatchSize * NumStocks, LSTM_HIDDEN_DIM]
        output, (hn, cn) = self.lstm(lstm_input)
        # 应用 Dropout 3 (在 LSTM 输出之后)
        output = self.dropout3(output)

        # --- Get LSTM output at the prediction time step ---
        # Select the output corresponding to the prediction time step for each sequence
        # output[:, PREDICTION_TIME_STEP, :] gives the output tensor [BatchSize * NumStocks, LSTM_HIDDEN_DIM]
        lstm_output_prediction_step_flat = output[:, self.prediction_time_step, :]

        # --- Output Heads ---
        # Apply output heads to the LSTM output at the prediction time step
        # Input to output heads: [BatchSize * NumStocks, LSTM_HIDDEN_DIM]
        # Regression output: [BatchSize * NumStocks, OUTPUT_REGRESSION_DIM]
        # Classification output: [BatchSize * NumStocks, OUTPUT_CLASSIFICATION_DIM]
        regression_output_flat, classification_output_flat = self.output_heads(lstm_output_prediction_step_flat)

        # Reshape outputs back to [BatchSize, NumStocks, OutputDim] for loss calculation
        regression_output = regression_output_flat.view(batch_size, num_stocks, OUTPUT_REGRESSION_DIM)
        classification_output = classification_output_flat.view(batch_size, num_stocks, OUTPUT_CLASSIFICATION_DIM)


        # --- Extract Labels for Loss Calculation ---
        # Labels are for the prediction time step
        # batch_labels shape: [BatchSize, TimeWindow, NumStocks, NumLabels]
        labels_prediction_step = batch_labels[:, self.prediction_time_step, :, :] # [BatchSize, NumStocks, NumLabels]

        # Split labels into regression and classification targets
        # Assuming regression labels are the first OUTPUT_REGRESSION_DIM columns in LABELS_COLS
        regression_target = labels_prediction_step[:, :, :OUTPUT_REGRESSION_DIM] # [B, N, RegDim]

        # Assuming classification labels are the remaining columns
        # Ensure classification target is LongTensor for CrossEntropyLoss
        classification_target = labels_prediction_step[:, :, OUTPUT_REGRESSION_DIM:].squeeze(-1).long()
        # [B, N] (assuming 1 classification label)


        # Return predictions and corresponding targets for the prediction timestep
        return regression_output, classification_output, regression_target, classification_target