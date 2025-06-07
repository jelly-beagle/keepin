import torch
import numpy as np

# ==================== 数据路径配置 ====================
# 包含每日数据子文件夹的根目录 (每个子文件夹以日期命名，包含当天的所有数据文件)
DATA_ROOT_DIR = '/mnt/workspace/timedataset/timedataset' # <<<< 请修改为您的实际根目录路径

# 股票代码列表文件路径
INDEX_FILE = '/mnt/workspace/index.xlsx' # 请修改为您的实际路径

# 训练集、验证集、测试集日期范围 (YYYYMMDD)
TRAIN_START_DATE_STR = "20190110"
TRAIN_END_DATE_STR = "20211231" # 示例：前三年作为训练集
VAL_START_DATE_STR = "20220101"
VAL_END_DATE_STR = "20221231" # 示例：第四年作为验证集
TEST_START_DATE_STR = "20230101"
TEST_END_DATE_STR = "20231229" # 示例：第五年作为测试集

# 归一化和填充统计量的保存路径
NORMALIZATION_STATS_FILE = "normalization_stats.pkl"
IMPUTATION_VALUES_FILE = "imputation_values.pkl"


# ==================== 数据文件命名配置 (在每日子文件夹内) ====================
# 每日节点特征文件名模板 (例如：20190110.csv)
DAILY_FEATURE_FILE_TEMPLATE = "{date_str}.csv"

# 基金持股图文件名模板 (按季度，例如：fund_graph_2018Q4.pt)。在每日子文件夹内。
FUND_GRAPH_FILE_TEMPLATE = "fund_graph_{quarter_str}.pt"

# 新闻共现图文件名模板 (按天，例如：graph_cooccurrence_20190110.pt)。在每日子文件夹内。
NEWS_GRAPH_FILE_TEMPLATE = "graph_cooccurrence_{date_str}.pt"

# 资金流图文件名模板 (按天，例如：flow_graph_20190110.pt)。在每日子文件夹内。
FLOW_GRAPH_FILE_TEMPLATE = "graph_data_hf_{date_str}.pt"


# ==================== 数据特征和图源配置 ====================
# 股票代码列名 (与每日特征文件中的一致)
STOCK_CODE_COL = 'Scode' # 请根据您实际文件中的列名调整

# 节点特征列名列表 (不包含股票代码和标签列)
NODE_FEATURES_COLS = [
    'FundamentalF1', 'FundamentalF2', 'FundamentalF3', 'FundamentalF4',
    'FundamentalF5', 'FundamentalF6', 'FundamentalF7', 'DaysSinceFundDisclosure',
    'Posnews_All_w', 'Neunews_All_w', 'Negnews_All_w',
    *[f'HFTF{i+1}' for i in range(20)],
]

# 标签列名列表 (需要预测的目标)
LABELS_COLS = [
    'Future5DayVolatility',
    'Future10DayVolatility',
    'RiskCategory'
]

# 回归任务的标签列名列表
REGRESSION_LABELS = ['Future5DayVolatility', 'Future10DayVolatility']

# 分类任务的标签列名列表
CLASSIFICATION_LABELS = ['RiskCategory']




# 图源类型列表
GRAPH_SOURCES = ['fund', 'news', 'flow']

# 每个图源对应的文件模板映射
GRAPH_FILE_TEMPLATES = {
    'fund': FUND_GRAPH_FILE_TEMPLATE,
    'news': NEWS_GRAPH_FILE_TEMPLATE,
    'flow': FLOW_GRAPH_FILE_TEMPLATE,
}

# >>> 新增配置：需要对边权重进行归一化的图源列表 <<<
GRAPH_SOURCES_NORM_EDGE_WEIGHT = ['news'] # <<<< 在这里指定需要标准化边权重的图源


# ==================== 模型超参数配置 (示例，稍后讨论) ====================
INPUT_DIM = len(NODE_FEATURES_COLS)
HIDDEN_DIM = 64
GNN_LAYERS_PER_SOURCE = 1
GNN_HEADS = 4
LSTM_LAYERS = 1
LSTM_HIDDEN_DIM = HIDDEN_DIM * len(GRAPH_SOURCES)
OUTPUT_REGRESSION_DIM = len(REGRESSION_LABELS)
OUTPUT_CLASSIFICATION_DIM = 2


# ==================== 训练参数配置 ====================
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
DROPOUT_RATE = 0.3

# 时间窗口配置
TIME_WINDOW_SIZE = 5

# 预测目标的时间步
PREDICTION_TIME_STEP = TIME_WINDOW_SIZE - 1

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 其他配置 ====================
SEED = 42

# 缺失值填充值 (如果文件级别填充后仍有 NaN，或需要基于全局统计量填充)
DEFAULT_IMPUTE_VALUE = 0.0 # 默认填充值，如果加载imputation_values失败或不存在对应列
# 每个图源对应的文件模板映射
GRAPH_FILE_TEMPLATES = {
    'fund': FUND_GRAPH_FILE_TEMPLATE,
    'news': NEWS_GRAPH_FILE_TEMPLATE,
    'flow': FLOW_GRAPH_FILE_TEMPLATE,
}


# ==================== 模型超参数配置 (示例，稍后讨论) ====================
INPUT_DIM = len(NODE_FEATURES_COLS) # 节点特征的输入维度
HIDDEN_DIM = 64 # GNN层和LSTM的隐藏层维度 (可以根据需要调整)
GNN_LAYERS_PER_SOURCE = 1 # 每个图源对应的GNN层数 (例如，使用1层GAT)
GNN_HEADS = 4 # GAT的注意力头数
LSTM_LAYERS = 1 # LSTM层数
LSTM_HIDDEN_DIM = HIDDEN_DIM * len(GRAPH_SOURCES) # 示例：LSTM 输入维度是融合后的维度
OUTPUT_REGRESSION_DIM = len(REGRESSION_LABELS) # 回归任务的输出维度
OUTPUT_CLASSIFICATION_DIM = 2 # 分类任务的输出维度 (风险分类，假定是二分类：0/1)


# ==================== 训练参数配置 ====================
EPOCHS = 50 # 训练轮数
BATCH_SIZE = 32 # 训练批次大小 (这里的批次是时间窗口的批次)
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001 # 权重衰减 (L2正则化)
DROPOUT_RATE = 0.3
# 时间窗口配置 (使用过去W天的数据来预测未来)
TIME_WINDOW_SIZE = 5 # 使用过去5天的数据作为一个序列输入LSTM

# 预测目标的时间步 (通常预测的是时间窗口最后一天的未来标签)
# 如果标签是与特征对齐的（如FutureXDayVolatility），则预测的是时间窗口最后一天的标签
# 预测时间步是窗口内的最后一个时间步，索引为 TIME_WINDOW_SIZE - 1
PREDICTION_TIME_STEP = TIME_WINDOW_SIZE - 1

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 其他配置 ====================
# 随机种子，用于复现结果
SEED = 42

# 缺失值填充值 (如果文件级别填充后仍有 NaN，或需要基于全局统计量填充)
# 这里的数值需要与您计算统计量的脚本输出一致
# 例如：对于 NODE_FEATURES_COLS 中的每一列，一个默认填充值
DEFAULT_IMPUTE_VALUE = 0.0 # 默认填充值，如果加载imputation_values失败或不存在对应列