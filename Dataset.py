import os
import torch
import pandas as pd
import numpy as np
import pickle # Or joblib for saving/loading stats
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from pathlib import Path
from tqdm import tqdm # Keep tqdm import for progress bar
import math # 导入 math 用于季度计算
import sys # Import sys for detailed exception info (kept for warnings)

# 如果使用 PyTorch Geometric (PyG)
# from torch_geometric.data import Data # PyG Data object structure


# 导入配置
# 确保这里导入了所有必要的配置，特别是 LABELS_COLS
try:
    from config import (
        DATA_ROOT_DIR, INDEX_FILE,
        TRAIN_START_DATE_STR, TRAIN_END_DATE_STR, VAL_START_DATE_STR, VAL_END_DATE_STR,
        TEST_START_DATE_STR, TEST_END_DATE_STR, NORMALIZATION_STATS_FILE, IMPUTATION_VALUES_FILE,
        STOCK_CODE_COL, NODE_FEATURES_COLS, LABELS_COLS, GRAPH_SOURCES, GRAPH_FILE_TEMPLATES,
        GRAPH_SOURCES_NORM_EDGE_WEIGHT, # 导入新增配置
        TIME_WINDOW_SIZE, PREDICTION_TIME_STEP,
        REGRESSION_LABELS, CLASSIFICATION_LABELS, DEFAULT_IMPUTE_VALUE,
        BATCH_SIZE # Added BATCH_SIZE import
    )
    print("dataset.py: 成功导入 config.py 配置。")
except ImportError:
    print("dataset.py: 错误：无法导入 config.py。请确保 config.py 文件存在且位于 Python 路径中。")
    raise # 如果 config 无法导入，应该在这里报错退出


# --- 辅助函数：加载股票池 ---
def load_stock_pool(index_file):
    """从Excel文件加载目标股票代码，返回排序后的列表"""
    try:
        # 假定文件无header，第一列是股票代码
        df = pd.read_excel(index_file) # Added header=None based on previous observation
        symbols = df.iloc[:, 0].astype(str).str.zfill(6).tolist()
        # 去重并排序
        sorted_symbols = sorted(list(set(symbols)))
        print(f"已加载 {len(sorted_symbols)} 个目标股票代码。")
        return sorted_symbols
    except FileNotFoundError:
        print(f"错误：指数文件未找到于 {index_file}")
        return []
    except Exception as e:
        print(f"加载指数文件时出错: {e}")
        return []

# --- 辅助函数：加载交易日列表 ---
def load_trading_days_from_dir(data_root_dir, start_date_str, end_date_str):
    """从数据根目录的子文件夹名称中提取交易日，并按日期过滤和排序"""
    if not os.path.isdir(data_root_dir):
        print(f"错误：数据根目录不存在：{data_root_dir}")
        return []

    all_dates = []
    # Using scandir for potential performance improvement with many entries
    try:
        with os.scandir(data_root_dir) as entries:
            for entry in entries:
                if entry.is_dir():
                    try:
                        date_obj = datetime.strptime(entry.name, '%Y%m%d')
                        all_dates.append(date_obj)
                    except ValueError:
                        continue # Skip directories that are not dates
    except Exception as e:
        print(f"读取数据目录 {data_root_dir} 时出错: {e}")
        # Fallback to listdir if scandir fails
        try: # Added try-except for listdir as well
            for folder_name in os.listdir(data_root_dir):
                try:
                    date_obj = datetime.strptime(folder_name, '%Y%m%d')
                    all_dates.append(date_obj)
                except ValueError:
                    continue
        except Exception as listdir_e:
             print(f"Fallback listdir 读取数据目录 {data_root_dir} 时出错: {listdir_e}")
             return []


    start_date = datetime.strptime(start_date_str, '%Y%m%d')
    end_date = datetime.strptime(end_date_str, '%Y%m%d')

    filtered_dates = sorted([date for date in all_dates if start_date <= date <= end_date])
    print(f"在日期范围 ({start_date.date()} to {end_date.date()}) 内加载了 {len(filtered_dates)} 个交易日 (来自目录名)。")
    return filtered_dates

# --- 辅助函数：加载归一化和填充统计量 ---
# 修改：增加 labels_cols 参数
def load_stats(norm_stats_file, impute_values_file, node_features_cols, labels_cols, graph_sources_norm_edge_weight):
    """加载预先计算好的归一化和填充统计量"""
    norm_stats = {
        'feature_mean': None,
        'feature_std': None,
        'feature_cols_order': None,
        'edge_mean': None,
        'edge_std': None
    }
    impute_values = None

    # 加载归一化统计量 (features and edge weights)
    if os.path.exists(norm_stats_file):
        try:
            with open(norm_stats_file, 'rb') as f:
                loaded_stats = pickle.load(f)

                # Load feature stats
                if 'feature_mean' in loaded_stats and 'feature_std' in loaded_stats and 'feature_cols_order' in loaded_stats:
                    # Check if loaded stats match expected feature dimension based on column list
                    if len(loaded_stats['feature_mean']) == len(node_features_cols) and \
                       len(loaded_stats['feature_std']) == len(node_features_cols):
                         norm_stats['feature_mean'] = loaded_stats['feature_mean']
                         norm_stats['feature_std'] = loaded_stats['feature_std']
                         norm_stats['feature_cols_order'] = loaded_stats['feature_cols_order']
                         print(f"加载特征归一化统计量成功：{norm_stats_file}")
                         if norm_stats['feature_cols_order'] != node_features_cols:
                              # This warning is important, means feature order might be wrong
                              print("警告：加载的特征归一化统计量的列顺序与配置不匹配！请检查 config.NODE_FEATURES_COLS 和统计计算脚本。")
                    else:
                         print(f"警告：加载的特征归一化统计量的维度与节点特征维度不匹配 ({len(loaded_stats['feature_mean'])} vs {len(node_features_cols)})。跳过特征归一化。")
                else:
                     print(f"警告：归一化统计文件缺少特征统计量 ('feature_mean', 'feature_std', 'feature_cols_order')。跳过特征归一化。")


                # Load edge weight stats for specified graph sources
                loaded_edge_mean = {}
                loaded_edge_std = {}
                if 'edge_mean' in loaded_stats and 'edge_std' in loaded_stats:
                     for source in graph_sources_norm_edge_weight:
                          # Check if stats exist for the specified source
                          if source in loaded_stats['edge_mean'] and source in loaded_stats['edge_std']:
                               loaded_edge_mean[source] = loaded_stats['edge_mean'][source]
                               loaded_edge_std[source] = loaded_stats['edge_std'][source]
                          else:
                               print(f"警告：归一化统计文件缺少图源 '{source}' 的边权重统计量。将不会对该图源边权重归一化。")

                     if loaded_edge_mean: # Check if any edge stats were loaded for specified sources
                         norm_stats['edge_mean'] = loaded_edge_mean
                         norm_stats['edge_std'] = loaded_edge_std
                         print(f"加载边权重归一化统计量成功（针对指定图源）。")
                     else:
                          print("警告：未加载到任何指定图源的边权重归一化统计量。跳过边权重归一化。")
                else:
                     print(f"警告：归一化统计文件缺少边权重统计量 ('edge_mean', 'edge_std') 键。跳过边权重归一化。")


        except Exception as e:
            print(f"加载归一化统计量文件 {norm_stats_file} 时出错：{e}。跳过所有归一化。")
            # Reset stats to None if loading fails
            norm_stats = {
                'feature_mean': None, 'feature_std': None, 'feature_cols_order': None,
                'edge_mean': None, 'edge_std': None
            }
    else:
        print(f"警告：归一化统计文件未找到：{norm_stats_file}。将跳过所有归一化。")

    # 加载填充值
    if os.path.exists(impute_values_file):
         try:
             with open(impute_values_file, 'rb') as f:
                 impute_values = pickle.load(f)
             print(f"加载填充值成功：{impute_values_file}")
             # Check if imputation values are available for all required columns
             # Use labels_cols parameter here
             all_expected_impute_cols = node_features_cols + labels_cols
             missing_impute_cols = [col for col in all_expected_impute_cols if col not in impute_values]
             if missing_impute_cols:
                  print(f"警告：填充值文件缺少列 {missing_impute_cols} 的填充值。将使用默认值 {DEFAULT_IMPUTE_VALUE}。")
                  # Add missing columns to impute_values dictionary with default value
                  for col in missing_impute_cols:
                       impute_values[col] = DEFAULT_IMPUTE_VALUE

         except Exception as e:
             # Use labels_cols parameter here
             print(f"加载填充值文件 {impute_values_file} 时出错：{e}。将为所有特征和标签使用默认填充值 {DEFAULT_IMPUTE_VALUE}。")
             # If loading fails, create a new impute_values dict with default for all expected columns
             impute_values = {col: DEFAULT_IMPUTE_VALUE for col in node_features_cols + labels_cols}
    else:
         # Use labels_cols parameter here
         print(f"警告：填充值文件未找到：{impute_values_file}。将为所有特征和标签使用默认填充值 {DEFAULT_IMPUTE_VALUE}。")
         # If file not found, create a new impute_values dict with default for all expected columns
         impute_values = {col: DEFAULT_IMPUTE_VALUE for col in node_features_cols + labels_cols}


    # Check if any normalization stats were successfully loaded
    if (norm_stats['feature_mean'] is None or norm_stats['feature_std'] is None) and \
       (norm_stats['edge_mean'] is None or norm_stats['edge_std'] is None):
         print("将跳过所有归一化步骤。")

    # Return loaded stats and impute values
    return norm_stats, impute_values

# --- 辅助函数：获取基金图季度文件名称 ---
def _get_fund_quarter_filename(date):
    """
    根据当前日期，返回应该加载的基金图季度文件名称。
    基金数据假设根据季度更新，例如 2019年1月1日至2019年3月31日的数据
    使用的是 2018年Q4 的文件。2019年4月1日至2019年6月30日使用 2019年Q1 的文件。
    """
    year = date.year
    month = date.month

    # 确定当前日期所在的季度
    current_quarter = math.ceil(month / 3)

    # 确定应该使用的季度文件（通常是上一个季度末的数据）
    if current_quarter == 1:
        # 如果当前是第一季度，使用前一年的第四季度数据
        prev_year = year - 1
        prev_quarter = 4
    else:
        # 否则使用当前年份的前一个季度数据
        prev_year = year
        prev_quarter = current_quarter - 1

    # 返回基金图文件名称的格式，例如 "fund_graph_2018Q4.pt"
    # 注意：这里的文件名格式需要与您实际保存的文件名一致
    return f"fund_graph_{prev_year}Q{prev_quarter}.pt"


# --- 自定义 Dataset ---
class StockDailyDataset(Dataset):
    # Added parameters to __init__ based on train.py's create_dataloaders call
    def __init__(self, data_root_dir, stock_pool, trading_days, node_features_cols, labels_cols, stock_code_col, graph_sources, graph_file_templates, graph_sources_norm_edge_weight, norm_stats=None, impute_values=None, time_window_size=5):
        self.data_root_dir = data_root_dir
        self.stock_pool = stock_pool
        self.trading_days = trading_days
        self.node_features_cols = node_features_cols
        self.labels_cols = labels_cols # Store labels_cols here
        self.stock_code_col = stock_code_col
        self.graph_sources = graph_sources
        self.graph_file_templates = graph_file_templates
        self.graph_sources_norm_edge_weight = graph_sources_norm_edge_weight
        self.norm_stats = norm_stats
        self.impute_values = impute_values
        self.time_window_size = time_window_size

        self.stock_to_node_index = {stock: i for i, stock in enumerate(self.stock_pool)}
        self.num_stocks = len(self.stock_pool)

        # Calculate the number of valid time windows
        self.valid_window_start_indices = range(max(0, len(self.trading_days) - self.time_window_size + 1))

        print(f"Dataset initialized: {len(self.valid_window_start_indices)} valid time windows.")


    def __len__(self):
        return len(self.valid_window_start_indices)


    def __getitem__(self, idx):
        window_start_day_index = self.valid_window_start_indices[idx]
        window_day_indices = range(window_start_day_index, window_start_day_index + self.time_window_size)

        window_data = [] # To store loaded data for each day in the window

        # Collect dates for the current window (used only for the imputation summary print now)
        window_dates = [self.trading_days[i] for i in window_day_indices]


        # --- Initialize imputation counters for this window ---
        total_elements_in_window = 0
        total_nans_before_impute_window = 0
        # --- End of imputation counters initialization ---


        for day_index in window_day_indices:
            current_day = self.trading_days[day_index]
            day_str = current_day.strftime('%Y%m%d')
            current_day_subdir = Path(self.data_root_dir) / day_str

            # --- 加载每日特征和标签数据 ---
            feature_label_file_path = current_day_subdir / f"{day_str}.csv"

            daily_df = None
            # Check if feature/label file exists and is readable
            if not feature_label_file_path.exists():
                 print(f"警告：日期 {day_str} 的特征/标签文件未找到：{feature_label_file_path}。")
                 # If file is missing, create an empty DataFrame to represent missing data
                 all_expected_cols = [self.stock_code_col] + self.node_features_cols + self.labels_cols
                 daily_df = pd.DataFrame(np.nan, index=self.stock_pool, columns=all_expected_cols)
                 daily_df[self.stock_code_col] = self.stock_pool # Add stock codes
                 # Update NaN counters for the whole day being missing
                 total_elements_in_window += self.num_stocks * (len(self.node_features_cols) + len(self.labels_cols))
                 total_nans_before_impute_window += self.num_stocks * (len(self.node_features_cols) + len(self.labels_cols))


            else:
                 try:
                      df_raw = pd.read_csv(feature_label_file_path)
                      # Check if the stock code column exists and if the DataFrame is not empty
                      if self.stock_code_col in df_raw.columns and not df_raw.empty:
                           # Ensure stock code column is string and formatted
                           df_raw[self.stock_code_col] = df_raw[self.stock_code_col].astype(str).str.zfill(6)
                           # Set stock code as index and reindex to the target stock pool
                           daily_df = df_raw.set_index(self.stock_code_col).reindex(self.stock_pool)

                           # --- Check and Count NaNs BEFORE imputation ---
                           cols_to_check = self.node_features_cols + self.labels_cols
                           existing_cols_to_check = [col for col in cols_to_check if col in daily_df.columns]

                           if existing_cols_to_check:
                                # Count NaNs before imputation
                                nans_before_impute_day = daily_df[existing_cols_to_check].isnull().sum().sum()
                                # Sum of all elements in these columns for this day
                                elements_in_day = daily_df[existing_cols_to_check].size

                                total_elements_in_window += elements_in_day
                                total_nans_before_impute_window += nans_before_impute_day

                           else:
                                estimated_elements = self.num_stocks * (len(self.node_features_cols) + len(self.labels_cols))
                                total_elements_in_window += estimated_elements
                                total_nans_before_impute_window += estimated_elements # Assume all missing if columns not found
                                print(f"警告：日期 {day_str} 加载的文件缺少所有预期特征/标签列。")


                      else:
                           print(f"警告：日期 {day_str} 的特征/标签文件缺少股票代码列 '{self.stock_code_col}' 或文件为空。")
                           # Create an empty DataFrame structured like the expected daily file
                           all_expected_cols = [self.stock_code_col] + self.node_features_cols + self.labels_cols
                           daily_df = pd.DataFrame(np.nan, index=self.stock_pool, columns=all_expected_cols)
                           daily_df[self.stock_code_col] = self.stock_pool # Add stock codes
                           # Update NaN counters for the whole day being missing
                           total_elements_in_window += self.num_stocks * (len(self.node_features_cols) + len(self.labels_cols))
                           total_nans_before_impute_window += self.num_stocks * (len(self.node_features_cols) + len(self.labels_cols))


                 except Exception as e:
                      print(f"警告：加载或初步处理特征/标签文件 {feature_label_file_path} 时出错：{e}。此天数据将全部视为缺失。")
                      # If loading fails, create an empty DataFrame to represent missing data
                      all_expected_cols = [self.stock_code_col] + self.node_features_cols + self.labels_cols
                      daily_df = pd.DataFrame(np.nan, index=self.stock_pool, columns=all_expected_cols)
                      daily_df[self.stock_code_col] = self.stock_pool # Add stock codes
                      # Update NaN counters for the whole day being missing
                      total_elements_in_window += self.num_stocks * (len(self.node_features_cols) + len(self.labels_cols))
                      total_nans_before_impute_window += self.num_stocks * (len(self.node_features_cols) + len(self.labels_cols))


            # --- 最终填充 (使用加载的 impute_values) ---
            # Apply imputation *after* reindexing and initial NaN counting
            if daily_df is not None and self.impute_values:
                # Use self.labels_cols here as it's stored in the instance
                all_cols_to_impute = self.node_features_cols + self.labels_cols
                for col in all_cols_to_impute:
                    if col in daily_df.columns: # Only try to fill if column exists
                        fill_value = self.impute_values.get(col, DEFAULT_IMPUTE_VALUE)
                        # Use .loc and assign back to avoid SettingWithCopyWarning
                        daily_df.loc[:, col] = daily_df.loc[:, col].fillna(fill_value)


            # --- Check for remaining NaNs AFTER imputation (should be 0) ---
            if daily_df is not None:
                remaining_nans_after_impute = daily_df[self.node_features_cols + self.labels_cols].isnull().sum().sum()
                if remaining_nans_after_impute > 0:
                    print(f"警告：日期 {day_str} 在填充后仍存在 {remaining_nans_after_impute} 个 NaN 数据点。请检查填充逻辑或数据源。")
                    pass # Continue for now

            # --- 提取特征和标签为 Tensor，并应用特征归一化 ---
            features_tensor = None
            labels_tensor = None

            # Only proceed if daily_df was successfully loaded/processed and is not None/empty after potential issues
            if daily_df is not None and not daily_df.empty:
                 try:
                     # SELECT and ORDER features and labels
                     # Use .copy() after selection to avoid SettingWithCopyWarning during later operations
                     features_df_selected = daily_df.filter(items=self.node_features_cols).copy()
                     labels_df_selected = daily_df.filter(items=self.labels_cols).copy()

                     # Convert feature columns to numeric, coercing errors
                     for col in self.node_features_cols:
                          if col in features_df_selected.columns:
                               features_df_selected.loc[:, col] = pd.to_numeric(features_df_selected.loc[:, col], errors='coerce')

                     # Ensure the order of columns in the tensor matches the config order
                     features_df_ordered = features_df_selected.reindex(columns=self.node_features_cols)
                     # Convert to tensor, ensuring dtype
                     if not features_df_ordered.empty:
                          features_tensor = torch.tensor(features_df_ordered.values, dtype=torch.float32)

                     # Convert label columns to numeric, coercing errors
                     for col in self.labels_cols:
                          if col in labels_df_selected.columns:
                               labels_df_selected.loc[:, col] = pd.to_numeric(labels_df_selected.loc[:, col], errors='coerce')

                     # Ensure the order of columns in the tensor matches the config order
                     labels_df_ordered = labels_df_selected.reindex(columns=self.labels_cols)
                     # If labels_df_ordered is None or empty, accessing .values will fail
                     if labels_df_ordered is not None and not labels_df_ordered.empty:
                         # Convert labels to tensor
                         labels_tensor = torch.tensor(labels_df_ordered.values, dtype=torch.float32) # Ensure float32 for now


                     # --- 应用特征归一化 ---
                     if features_tensor is not None and self.norm_stats and self.norm_stats['feature_mean'] is not None and self.norm_stats['feature_std'] is not None:
                          # Ensure dimensions match between features tensor and stats
                          if features_tensor.shape[-1] == len(self.norm_stats['feature_mean']):
                               # Ensure stats are Tensors on the correct device and dtype (float32) for element-wise operations
                               mean_vals = torch.tensor(self.norm_stats['feature_mean'], dtype=torch.float32) # Keep on CPU, device handling in train.py
                               std_dev = torch.tensor(self.norm_stats['feature_std'], dtype=torch.float32)   # Keep on CPU
                               std_dev[std_dev == 0] = 1e-8 # Prevent division by zero
                               features_tensor = (features_tensor - mean_vals) / std_dev
                          else:
                               print(f"警告：日期 {day_str} 特征维度 {features_tensor.shape[-1]} 与统计量维度 {len(self.norm_stats['feature_mean'])} 不匹配，跳过特征归一化。")


                 except Exception as e:
                     # Print the actual exception type and details here, including line number
                     exc_type, exc_obj, exc_tb = sys.exc_info()
                     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                     print(f"警告：转换日期 {day_str} 的特征或标签到 Tensor 时出错：{type(e).__name__} - {e}。错误位置: {fname}, 行号: {exc_tb.tb_lineno}。此天数据将视为无效。")
                     features_tensor = None
                     labels_tensor = None


            # --- 加载每日图结构数据并应用边权重归一化 (仅对指定图源) ---
            daily_graphs = {}  # {source: {edge_index, edge_weight} or None}

            # Load graphs even if features/labels failed for consistency in the window structure
            # The model handles None graphs.
            for source in self.graph_sources:
                graph_filename = None
                # Determine filename based on source (date or quarter logic for fund)
                if source == 'fund':
                     graph_filename = _get_fund_quarter_filename(current_day)
                elif source in self.graph_file_templates:
                     graph_filename = self.graph_file_templates[source].format(date_str=day_str)
                else:
                     print(f"警告：日期 {day_str}, 图源 {source} 没有定义文件加载模板或特殊加载逻辑，跳过加载图数据。")
                     daily_graphs[source] = None
                     continue


                graph_file_path = current_day_subdir / graph_filename

                graph_data = None  # Initialize as None
                if not graph_file_path.exists():
                     daily_graphs[source] = None
                else:
                    try:
                         loaded_graph_data = torch.load(graph_file_path)

                         if 'nodes' in loaded_graph_data and 'edge_index' in loaded_graph_data:
                             edge_index_tensor = loaded_graph_data['edge_index'].to(torch.long) # Keep on CPU for now
                             edge_weight_tensor = loaded_graph_data.get('edge_weight', None)

                             if edge_weight_tensor is not None:
                                 edge_weight_tensor = edge_weight_tensor.to(torch.float32) # Keep on CPU for now

                             # --- Apply edge weight normalization (only for specified sources) ---
                             normalized_edge_weight = edge_weight_tensor
                             if source in self.graph_sources_norm_edge_weight and \
                                edge_weight_tensor is not None and \
                                self.norm_stats and \
                                self.norm_stats['edge_mean'] is not None and self.norm_stats[
                                    'edge_std'] is not None and \
                                source in self.norm_stats['edge_mean'] and source in self.norm_stats[
                                    'edge_std']:

                                  mean_weight = self.norm_stats['edge_mean'][source]
                                  std_weight = self.norm_stats['edge_std'][source]
                                  if std_weight == 0:
                                      std_weight = 1e-8

                                  mean_weight_tensor = torch.tensor(mean_weight, dtype=torch.float32) # Keep on CPU
                                  std_weight_tensor = torch.tensor(std_weight, dtype=torch.float32) # Keep on CPU

                                  normalized_edge_weight = (edge_weight_tensor - mean_weight_tensor) / std_weight_tensor

                             graph_data = {
                                 'edge_index': edge_index_tensor,
                                 'edge_weight': normalized_edge_weight # Store normalized (or original float32) edge weight
                             }

                         else:
                             print(f"警告：日期 {day_str}, 图源 {source} 的图文件 {graph_file_path} 缺少必需的字段 ('nodes', 'edge_index')。跳过此图。")

                    except Exception as e:
                         print(f"警告：日期 {day_str}, 图源 {source}：加载或处理图文件 {graph_file_path} 时出错：{e}。跳过此图。")

                daily_graphs[source] = graph_data


            # --- Organize data for the current time step ---
            # Only include this time step if BOTH features and labels were successfully created and have correct shape
            # We still store graph data (even if None) for consistency in the window structure
            if features_tensor is not None and labels_tensor is not None:
                 # Check if features and labels have the expected number of stocks
                 if features_tensor.shape[0] == self.num_stocks and labels_tensor.shape[0] == self.num_stocks:
                     window_data.append({
                        'features': features_tensor,  # Shape: [NumStocks, NumFeatures] (float32)
                        'labels': labels_tensor,  # Shape: [NumStocks, NumLabels] (float32)
                        'graphs': daily_graphs,
                        # {source: {edge_index (long), edge_weight (float32)} or None}
                        'date': current_day  # Store date for context
                     })
                 else:
                      print(f"警告：日期 {day_str} 特征/标签张量形状 {features_tensor.shape}/{labels_tensor.shape} 与预期节点数 {self.num_stocks} 不匹配，跳过此时间步。")

            else:
                 pass


        # --- Calculate and Print Imputation Summary for the Window ---
        # This block is executed AFTER the loop over days in the window
        if len(window_data) == self.time_window_size: # Only calculate for complete windows
             # Total number of elements across all days and all feature/label columns in a complete window
             total_possible_elements = self.time_window_size * self.num_stocks * (len(self.node_features_cols) + len(self.labels_cols))
             total_imputed_elements = total_nans_before_impute_window

             if total_possible_elements > 0:
                  imputed_percentage = (total_imputed_elements / total_possible_elements) * 100
             else:
                  imputed_percentage = 0



        # Check if the window has the expected number of complete time steps
        if len(window_data) == self.time_window_size:
            features_window = torch.stack([item['features'] for item in window_data],
                                          dim=0)  # Shape: [TimeWindow, NumStocks, NumFeatures]
            labels_window = torch.stack([item['labels'] for item in window_data],
                                        dim=0)  # Shape: [TimeWindow, NumStocks, NumLabels]

            graphs_window = [item['graphs'] for item in window_data]
            dates_window = [item['date'] for item in window_data]

            return features_window, labels_window, graphs_window, dates_window

        else:
            # If the window is incomplete, return None
            if len(window_data) > 0: # Only print warning if some data was loaded but not enough for a full window
                 print(f"警告：时间窗口结束日 {self.trading_days[window_start_day_index + len(window_data) - 1].strftime('%Y%m%d')} 数据不完整 (成功加载 {len(window_data)}/{self.time_window_size} 天)，跳过此窗口。")
            return None # Return None to indicate an invalid window


# --- 自定义 collate_fn ---
def custom_collate_fn(batch):
    # Filter out None values (invalid windows)
    batch = [item for item in batch if item is not None]

    if not batch: # If batch is empty after filtering
        return None # Return None, will be skipped in train loop

    # batch is a list of (features_window, labels_window, graphs_window, dates_window)
    batch_features = torch.stack([item[0] for item in batch], dim=0)
    batch_labels = torch.stack([item[1] for item in batch], dim=0)

    # Graphs is a nested list: [BatchSize, TimeWindow, {source: {ei,ew}|None}]
    batch_graphs_nested_list = [item[2] for item in batch]

    # Dates is a list of lists: [BatchSize, TimeWindow]
    batch_dates_list = [item[3] for item in batch]

    # Removed DEBUG prints from collate_fn

    return batch_features, batch_labels, batch_graphs_nested_list, batch_dates_list


# --- 创建 DataLoader ---
def create_dataloaders(norm_stats_file, impute_values_file):
    """创建训练、验证和测试集的 DataLoader"""

    stock_pool = load_stock_pool(INDEX_FILE)
    if not stock_pool:
         raise RuntimeError("无法加载股票池，请检查INDEX_FILE配置。")

    # 加载归一化和填充统计量 (包括边权重统计量)
    norm_stats, impute_values = load_stats(NORMALIZATION_STATS_FILE, IMPUTATION_VALUES_FILE, NODE_FEATURES_COLS, LABELS_COLS, GRAPH_SOURCES_NORM_EDGE_WEIGHT)


    # Load all trading days in the total range first
    all_trading_days = load_trading_days_from_dir(DATA_ROOT_DIR, TRAIN_START_DATE_STR, TEST_END_DATE_STR)
    if not all_trading_days:
         print(f"错误：在总日期范围 ({TRAIN_START_DATE_STR} to {TEST_END_DATE_STR}) 内未找到任何交易日。请检查 DATA_ROOT_DIR 和数据文件夹名称。")
         return None, None, None


    # Filter trading days based on start/end dates for each split
    try:
        train_start_date = datetime.strptime(TRAIN_START_DATE_STR, '%Y%m%d')
        train_end_date = datetime.strptime(TRAIN_END_DATE_STR, '%Y%m%d')
        val_start_date = datetime.strptime(VAL_START_DATE_STR, '%Y%m%d')
        val_end_date = datetime.strptime(VAL_END_DATE_STR, '%Y%m%d')
        test_start_date = datetime.strptime(TEST_START_DATE_STR, '%Y%m%d')
        test_end_date = datetime.strptime(TEST_END_DATE_STR, '%Y%m%d')
    except ValueError as e:
         print(f"错误：配置中的日期格式不正确，应为YYYYMMDD: {e}")
         return None, None, None


    train_days = [day for day in all_trading_days if train_start_date <= day <= train_end_date]
    val_days = [day for day in all_trading_days if val_start_date <= day <= val_end_date]
    test_days = [day for day in all_trading_days if test_start_date <= day <= test_end_date]

    # Print counts of trading days for each split
    print(f"\n交易日划分:")
    print(f"  训练集交易日数量: {len(train_days)}")
    print(f"  验证集交易日数量: {len(val_days)}")
    print(f"  测试集交易日数量: {len(test_days)}")


    # Create Dataset instances
    train_dataset = None
    if len(train_days) >= TIME_WINDOW_SIZE:
        train_dataset = StockDailyDataset(
            data_root_dir=DATA_ROOT_DIR,
            stock_pool=stock_pool,
            trading_days=train_days,
            node_features_cols=NODE_FEATURES_COLS,
            labels_cols=LABELS_COLS, # Pass LABELS_COLS
            stock_code_col=STOCK_CODE_COL,
            graph_sources=GRAPH_SOURCES,
            graph_file_templates=GRAPH_FILE_TEMPLATES,
            graph_sources_norm_edge_weight=GRAPH_SOURCES_NORM_EDGE_WEIGHT,
            norm_stats=norm_stats,
            impute_values=impute_values,
            time_window_size=TIME_WINDOW_SIZE
        )
        print(f"训练集 Dataset 创建成功，包含 {len(train_dataset)} 个有效时间窗口。")
    else:
         print(f"警告：训练集交易日数量 ({len(train_days)}) 少于时间窗口大小 ({TIME_WINDOW_SIZE})，无法创建训练集 Dataset。")


    val_dataset = None
    if len(val_days) >= TIME_WINDOW_SIZE:
        val_dataset = StockDailyDataset(
            data_root_dir=DATA_ROOT_DIR,
            stock_pool=stock_pool,
            trading_days=val_days,
            node_features_cols=NODE_FEATURES_COLS,
            labels_cols=LABELS_COLS, # Pass LABELS_COLS
            stock_code_col=STOCK_CODE_COL,
            graph_sources=GRAPH_SOURCES,
            graph_file_templates=GRAPH_FILE_TEMPLATES,
            graph_sources_norm_edge_weight=GRAPH_SOURCES_NORM_EDGE_WEIGHT,
            norm_stats=norm_stats,
            impute_values=impute_values,
            time_window_size=TIME_WINDOW_SIZE
        )
        print(f"验证集 Dataset 创建成功，包含 {len(val_dataset)} 个有效时间窗口。")
    else:
         print(f"警告：验证集交易日数量 ({len(val_days)}) 少于时间窗口大小 ({TIME_WINDOW_SIZE})，无法创建验证集 Dataset。")

    test_dataset = None
    if len(test_days) >= TIME_WINDOW_SIZE:
        test_dataset = StockDailyDataset(
            data_root_dir=DATA_ROOT_DIR,
            stock_pool=stock_pool,
            trading_days=test_days,
            node_features_cols=NODE_FEATURES_COLS,
            labels_cols=LABELS_COLS, # Pass LABELS_COLS
            stock_code_col=STOCK_CODE_COL,
            graph_sources=GRAPH_SOURCES,
            graph_file_templates=GRAPH_FILE_TEMPLATES,
            graph_sources_norm_edge_weight=GRAPH_SOURCES_NORM_EDGE_WEIGHT,
            norm_stats=norm_stats,
            impute_values=impute_values,
            time_window_size=TIME_WINDOW_SIZE
        )
        print(f"测试集 Dataset 创建成功，包含 {len(test_dataset)} 个有效时间窗口。")
    else:
         print(f"警告：测试集交易日数量 ({len(test_days)}) 少于时间窗口大小 ({TIME_WINDOW_SIZE})，无法创建测试集 Dataset。")


    # Create DataLoader instances
    train_loader = None
    if train_dataset and len(train_dataset) > 0:
         train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=0) # num_workers=0 for easier debugging
         print(f"训练集 DataLoader 创建成功，包含 {len(train_loader)} 批次。")
    else:
         print("训练集 Dataset 无有效窗口，跳过创建 DataLoader。")


    val_loader = None
    if val_dataset and len(val_dataset) > 0:
         val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=0) # num_workers=0 for easier debugging
         print(f"验证集 DataLoader 创建成功，包含 {len(val_loader)} 批次。")
    else:
         print("验证集 Dataset 无有效窗口，跳过创建 DataLoader。")

    test_loader = None
    if test_dataset and len(test_dataset) > 0:
         test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=0) # num_workers=0 for easier debugging
         print(f"测试集 DataLoader 创建成功，包含 {len(test_loader)} 批次。")
    else:
         print("测试集 Dataset 无有效窗口，跳过创建 DataLoader。")


    # Return DataLoaders (some might be None if datasets were empty)
    return train_loader, val_loader, test_loader

# --- 在模型训练前，请先运行独立的统计量计算脚本 (calculate_stats.py) ---
# 这个脚本会生成 normalization_stats.pkl 和 imputation_values.pkl
# 然后 create_dataloaders 函数会自动加载它们