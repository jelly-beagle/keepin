import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import os
from tqdm import tqdm

# 导入评估指标和可视化库
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


# 导入自定义模块和配置
try:
    from config import (
        SEED, DEVICE, EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
        NORMALIZATION_STATS_FILE, IMPUTATION_VALUES_FILE,
        TIME_WINDOW_SIZE, PREDICTION_TIME_STEP,
        OUTPUT_REGRESSION_DIM, OUTPUT_CLASSIFICATION_DIM,
        REGRESSION_LABELS, CLASSIFICATION_LABELS,
        DROPOUT_RATE
    )
    print("成功导入 config.py 配置。")
except ImportError:
    print("错误：无法导入 config.py。请确保 config.py 文件存在且位于 Python 路径中。")
    raise


from Dataset import create_dataloaders
from model import StockPredictorGNN

# --- 设置随机种子 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"已设置随机种子: {seed}")

# --- 设置设备 ---
def setup_device():
    device = torch.device(DEVICE)
    print(f"使用设备: {device}")
    return device

# --- 训练一个 epoch ---
# 返回训练集分类损失和准确率
def train_one_epoch(model, train_loader, criterion_reg, criterion_cls, optimizer, device):
    model.train()
    total_loss = 0
    total_reg_loss = 0
    total_cls_loss = 0
    all_cls_preds = []
    all_cls_targets = []

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training Batches", leave=False)):
        if batch is None:
            continue

        batch_features, batch_labels, batch_graphs_nested_list, batch_dates_list = batch
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        regression_output, classification_output, regression_target, classification_target = model(
            batch_features, batch_labels, batch_graphs_nested_list
        )

        reg_loss = criterion_reg(regression_output, regression_target)
        classification_output_flat = classification_output.view(-1, OUTPUT_CLASSIFICATION_DIM)
        classification_target_flat = classification_target.view(-1)
        if classification_target_flat.dtype != torch.long:
             classification_target_flat = classification_target_flat.long()

        cls_loss = criterion_cls(classification_output_flat, classification_target_flat)

        total_batch_loss = reg_loss + cls_loss

        total_batch_loss.backward()
        optimizer.step()

        total_loss += total_batch_loss.item()
        total_reg_loss += reg_loss.item()
        total_cls_loss += cls_loss.item()

        # Collect classification predictions and targets
        if OUTPUT_CLASSIFICATION_DIM > 0:
            predicted_classes = torch.argmax(classification_output, dim=-1)
            all_cls_preds.append(predicted_classes.cpu().numpy())
            all_cls_targets.append(classification_target.cpu().numpy())


    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
    avg_reg_loss = total_reg_loss / len(train_loader) if len(train_loader) > 0 else 0
    avg_cls_loss = total_cls_loss / len(train_loader) if len(train_loader) > 0 else 0

    # Calculate training classification accuracy
    train_accuracy = np.nan
    if OUTPUT_CLASSIFICATION_DIM > 0 and all_cls_targets:
         all_cls_preds_flat = np.concatenate(all_cls_preds).flatten()
         all_cls_targets_flat = np.concatenate(all_cls_targets).flatten()
         if all_cls_targets_flat.size > 0:
              train_accuracy = accuracy_score(all_cls_targets_flat, all_cls_preds_flat)


    return avg_loss, avg_reg_loss, avg_cls_loss, train_accuracy


# --- 验证/评估模型 ---
# 返回损失、指标、扁平化预测/真实值、原始标签和日期
def evaluate_model(model, data_loader, criterion_reg, criterion_cls, device, desc="Evaluating"):
    model.eval()
    total_loss = 0
    total_reg_loss = 0
    total_cls_loss = 0
    all_reg_preds = []
    all_reg_targets = []
    all_cls_preds = []
    all_cls_targets = []
    all_original_labels_batch = []
    all_batch_dates_list = [] # List of lists of datetimes

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc=desc, leave=False)):
            if batch is None:
                continue

            batch_features, batch_labels, batch_graphs_nested_list, batch_dates_list = batch
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            regression_output, classification_output, regression_target, classification_target = model(
                batch_features, batch_labels, batch_graphs_nested_list
            )

            reg_loss = criterion_reg(regression_output, regression_target)
            classification_output_flat = classification_output.view(-1, OUTPUT_CLASSIFICATION_DIM)
            classification_target_flat = classification_target.view(-1)
            if classification_target_flat.dtype != torch.long:
                 classification_target_flat = classification_target_flat.long()

            cls_loss = criterion_cls(classification_output_flat, classification_target_flat)


            total_batch_loss = reg_loss + cls_loss


            total_loss += total_batch_loss.item()
            total_reg_loss += reg_loss.item()
            total_cls_loss += cls_loss.item()

            all_reg_preds.append(regression_output.cpu().numpy())
            all_reg_targets.append(regression_target.cpu().numpy())
            if OUTPUT_CLASSIFICATION_DIM > 0:
                predicted_classes = torch.argmax(classification_output, dim=-1)
                all_cls_preds.append(predicted_classes.cpu().numpy())
                all_cls_targets.append(classification_target.cpu().numpy())

            all_original_labels_batch.append(batch_labels.cpu().numpy())
            all_batch_dates_list.extend(batch_dates_list)


    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    avg_reg_loss = total_reg_loss / len(data_loader) if len(data_loader) > 0 else 0
    avg_cls_loss = total_cls_loss / len(data_loader) if len(data_loader) > 0 else 0

    if not all_reg_preds and not all_cls_preds: # Check if any data was processed for either task
        print(f"\n{desc} 完成 - 没有处理任何数据批次。")
        return avg_loss, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), []


    all_reg_preds_np = np.concatenate(all_reg_preds, axis=0) if all_reg_preds else np.array([])
    all_reg_targets_np = np.concatenate(all_reg_targets, axis=0) if all_reg_targets else np.array([])
    all_cls_preds_np = np.concatenate(all_cls_preds, axis=0) if all_cls_preds else np.array([])
    all_cls_targets_np = np.concatenate(all_cls_targets, axis=0) if all_cls_targets else np.array([])

    all_original_labels_np = np.concatenate(all_original_labels_batch, axis=0) if all_original_labels_batch else np.array([])


    # Reshape for metric calculation: [TotalSamples * NumStocks, Dim]
    all_reg_preds_flat = all_reg_preds_np.reshape(-1, OUTPUT_REGRESSION_DIM) if all_reg_preds_np.size > 0 else np.array([])
    all_reg_targets_flat = all_reg_targets_np.reshape(-1, OUTPUT_REGRESSION_DIM) if all_reg_targets_np.size > 0 else np.array([])
    all_cls_preds_flat = all_cls_preds_np.flatten() if all_cls_preds_np.size > 0 else np.array([])
    all_cls_targets_flat = all_cls_targets_np.flatten() if all_cls_targets_np.size > 0 else np.array([])


    # Calculate metrics
    reg_mse = np.nan
    reg_mae = np.nan
    avg_reg_r2 = np.nan
    reg_r2_list = [] # Keep list for per-target R2 printing

    if all_reg_targets_flat.size > 0:
         reg_mse = mean_squared_error(all_reg_targets_flat, all_reg_preds_flat)
         reg_mae = mean_absolute_error(all_reg_targets_flat, all_reg_preds_flat)
         for i in range(OUTPUT_REGRESSION_DIM):
              if np.std(all_reg_targets_flat[:, i]) > 1e-8:
                   r2 = r2_score(all_reg_targets_flat[:, i], all_reg_preds_flat[:, i])
                   reg_r2_list.append(r2)
              else:
                   print(f"警告：{desc} 回归目标 {i+1} (或 {REGRESSION_LABELS[i] if REGRESSION_LABELS and i < len(REGRESSION_LABELS) else 'Unknown'}) 在评估集中是常量，无法计算 R2。")
         avg_reg_r2 = np.mean(reg_r2_list) if reg_r2_list else np.nan


    cls_accuracy = np.nan
    cls_f1 = np.nan
    cls_precision = np.nan
    cls_recall = np.nan
    cm = None

    if all_cls_targets_flat.size > 0:
         unique_classes = np.unique(all_cls_targets_flat)
         if len(unique_classes) >= 2:
              cls_accuracy = accuracy_score(all_cls_targets_flat, all_cls_preds_flat)
              cls_f1 = f1_score(all_cls_targets_flat, all_cls_preds_flat, average='binary', zero_division=0)
              cls_precision = precision_score(all_cls_targets_flat, all_cls_preds_flat, average='binary', zero_division=0)
              cls_recall = recall_score(all_cls_targets_flat, all_cls_preds_flat, average='binary', zero_division=0)

              all_possible_labels = np.unique(np.concatenate((all_cls_targets_flat, all_cls_preds_flat)))
              cm = confusion_matrix(all_cls_targets_flat, all_cls_preds_flat, labels=all_possible_labels)


         elif len(unique_classes) == 1:
             cls_accuracy = accuracy_score(all_cls_targets_flat, all_cls_preds_flat)
             print(f"警告：{desc} 分类目标只包含一个类别 ({unique_classes[0]})。无法计算 F1, Precision, Recall。")


    # --- 打印评估结果 ---
    print(f"\n--- {desc} 评估结果 ---")
    print(f"平均总损失: {avg_loss:.4f}")
    print(f"平均回归损失: {avg_reg_loss:.4f}")
    print(f"平均分类损失: {avg_cls_loss:.4f}")

    if OUTPUT_REGRESSION_DIM > 0:
        print("\n回归任务评估:")
        if not np.isnan(reg_mse): print(f"  MSE: {reg_mse:.4f}")
        if not np.isnan(reg_mae): print(f"  MAE: {reg_mae:.4f}")
        if not np.isnan(avg_reg_r2): print(f"  平均 R2: {avg_reg_r2:.4f}")
        if OUTPUT_REGRESSION_DIM > 1 and reg_r2_list:
             print("  各回归目标 R2:")
             for i in range(OUTPUT_REGRESSION_DIM):
                  # Check if R2 was calculated for this target (std > 0)
                  if i < len(reg_r2_list): # Assuming reg_r2_list only contains calculable R2s in order
                      print(f"    目标 {i+1} ({REGRESSION_LABELS[i] if REGRESSION_LABELS and i < len(REGRESSION_LABELS) else 'Unknown'}): {reg_r2_list[i]:.4f}")
                  elif np.std(all_reg_targets_flat[:, i]) <= 1e-8:
                       print(f"    目标 {i+1} ({REGRESSION_LABELS[i] if REGRESSION_LABELS and i < len(REGRESSION_LABELS) else 'Unknown'}): 无法计算 R2 (常量)")


    if OUTPUT_CLASSIFICATION_DIM > 0:
        print("\n分类任务评估:")
        if not np.isnan(cls_accuracy): print(f"  准确率: {cls_accuracy:.4f}")
        if not np.isnan(cls_f1):
             print(f"  F1 分数 (二分类，正类=1): {cls_f1:.4f}")
             print(f"  精确率 (二分类，正类=1): {cls_precision:.4f}")
             print(f"  召回率 (二分类，正类=1): {cls_recall:.4f}")

        if cm is not None:
             print("\n混淆矩阵:")
             print(cm)
             # 可视化混淆矩阵
             plt.figure(figsize=(6, 5))
             sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                         xticklabels=all_possible_labels,
                         yticklabels=all_possible_labels)
             plt.xlabel('预测标签')
             plt.ylabel('真实标签')
             plt.title(f'{desc} 混淆矩阵')
             # Don't call plt.show() here, will be shown at the end


    print("-" * 20)

    # Return everything needed for main function
    return avg_loss, avg_reg_loss, avg_cls_loss, reg_mse, avg_reg_r2, cls_accuracy, cls_f1, all_reg_preds_flat, all_reg_targets_flat, all_cls_preds_flat, all_cls_targets_flat, all_original_labels_np, all_batch_dates_list


# --- 主训练函数 ---
def main():
    set_seed(SEED)
    device = setup_device()

    print("\n加载数据...")
    try:
        # create_dataloaders returns train, val, test loaders (can be None)
        train_loader, val_loader, test_loader = create_dataloaders(NORMALIZATION_STATS_FILE, IMPUTATION_VALUES_FILE)
        if train_loader is None:
             print("错误：训练集 DataLoader 创建失败。请检查数据路径和日期范围。")
             return
        if val_loader is None:
             print("警告：验证集 DataLoader 创建失败。将跳过验证阶段。")
             # Ensure val_loader is None if creation failed
             val_loader = None
        if test_loader is None:
             print("警告：测试集 DataLoader 创建失败。将跳过测试阶段。")
             # Ensure test_loader is None if creation failed
             test_loader = None

        print("数据加载完成。")
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return

    # Instantiate model
    stock_pool = []
    NUM_STOCKS = None
    if train_loader and train_loader.dataset and hasattr(train_loader.dataset, 'num_stocks'):
         NUM_STOCKS = train_loader.dataset.num_stocks
         if hasattr(train_loader.dataset, 'stock_pool'):
              stock_pool = train_loader.dataset.stock_pool
              print(f"已获取股票池列表 ({len(stock_pool)} 只股票)。")
         else:
              print("警告：无法从训练数据集中获取股票池列表。特定股票可视化可能无法进行。")
         print(f"从训练数据集中确定节点数量: {NUM_STOCKS}")
    else:
         print("错误：无法从训练数据集中确定节点数量。请检查 dataset.py 中的 stock_pool 加载逻辑或 train_loader 是否成功创建。")
         return


    model = StockPredictorGNN(num_stocks=NUM_STOCKS, dropout_rate=DROPOUT_RATE)
    model.to(device)
    print("\n模型实例化完成.")

    # Define loss functions
    criterion_reg = nn.MSELoss()
    if OUTPUT_CLASSIFICATION_DIM > 0 and OUTPUT_CLASSIFICATION_DIM != 2:
         print(f"警告：CONFIG 中 OUTPUT_CLASSIFICATION_DIM ({OUTPUT_CLASSIFICATION_DIM}) 与二分类任务期望的维度 (2) 不匹配。")
    elif OUTPUT_CLASSIFICATION_DIM == 0:
         print("提示：CONFIG 中 OUTPUT_CLASSIFICATION_DIM 为 0，将跳过分类任务训练和评估。")

    # If classification task exists, define classification criterion
    criterion_cls = nn.CrossEntropyLoss() if OUTPUT_CLASSIFICATION_DIM > 0 else None


    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # --- Training Loop ---
    print("\n开始训练...")
    # 删除早停参数初始化相关的行:
    # best_val_f1 = -float('inf')
    # best_epoch = -1

    # checkpoint_dir = "checkpoints" # 可以保留这个目录用于保存最终模型或其他 checkpoints
    # os.makedirs(checkpoint_dir, exist_ok=True)
    # best_model_path = os.path.join(checkpoint_dir, "best_model.pth") # 删除保存最佳模型的路径变量

    # Lists to store history for plotting
    train_total_losses = []
    train_reg_losses = []
    train_cls_losses = []
    train_accuracies = []

    val_total_losses = []
    val_reg_losses = []
    val_cls_losses = []
    val_accuracies = []
    val_f1_scores = [] # 可以保留用于验证集 F1 分数绘图

    total_training_start_time = time.time()

    # 删除早停参数相关的行:
    # patience = 10
    # epochs_no_improve = 0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        epoch_start_time = time.time()

        # Train
        # train_one_epoch now returns train_accuracy
        train_loss, train_reg_loss, train_cls_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion_reg, criterion_cls, optimizer, device
        )
        train_total_losses.append(train_loss)
        train_reg_losses.append(train_reg_loss)
        # Append training classification loss and accuracy only if classification task exists
        if OUTPUT_CLASSIFICATION_DIM > 0:
             train_cls_losses.append(train_cls_loss)
             train_accuracies.append(train_accuracy)

        print(f"训练 Epoch {epoch+1} 完成 - 平均总损失: {train_loss:.4f}, 回归损失: {train_reg_loss:.4f}, 分类损失: {train_cls_loss:.4f}, 准确率: {train_accuracy:.4f}")


        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch+1} 耗时: {epoch_duration:.2f}秒")

        if epoch < EPOCHS - 1:
            elapsed_time = epoch_end_time - total_training_start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_epochs = EPOCHS - (epoch + 1)
            estimated_remaining_time = avg_epoch_time * remaining_epochs
            print(f"已用时: {elapsed_time:.2f}秒, 平均 Epoch 耗时: {avg_epoch_time:.2f}秒, 预计剩余时间: {estimated_remaining_time:.2f}秒")


        # 验证 (If val_loader exists)
        if val_loader:
            print("\n运行验证集...")
            # evaluate_model returns 13 values
            val_loss, val_reg_loss, val_cls_loss, val_reg_mse, val_reg_r2, val_accuracy, val_f1, \
            _, _, _, _, _, _ = evaluate_model( # Ignore detailed prediction/target data here
                model, val_loader, criterion_reg, criterion_cls, device, desc=f"Evaluating Val (Epoch {epoch+1})"
            )
            val_total_losses.append(val_loss)
            val_reg_losses.append(val_reg_loss)
            # Append validation classification loss, accuracy, and F1 only if classification task exists
            if OUTPUT_CLASSIFICATION_DIM > 0:
                val_cls_losses.append(val_cls_loss)
                val_accuracies.append(val_accuracy)
                val_f1_scores.append(val_f1) # Record F1 score for plotting


            # 删除基于验证集性能的早停检查和最佳模型保存逻辑:
            # if OUTPUT_CLASSIFICATION_DIM > 0:
            #     if not np.isnan(val_f1):
            #         if val_f1 > best_val_f1:
            #             best_val_f1 = val_f1
            #             best_epoch = epoch + 1
            #             torch.save(model.state_dict(), best_model_path)
            #             print(f"\n>>> 保存最优模型到 {best_model_path} (验证集 F1: {best_val_f1:.4f} 在 Epoch {best_epoch}) <<<")
            #             epochs_no_improve = 0
            #         else:
            #             epochs_no_improve += 1
            #
            #         # Check Early Stopping
            #         if epochs_no_improve >= patience:
            #             print(f"\nEarly stopping triggered after {patience} epochs with no improvement in Validation F1 score.")
            #             break # Exit the training loop
            #
            #     else:
            #          print("警告：验证集 F1 分数不可用，跳过 Early Stopping 检查本 Epoch。")
            #
            # else: # If no classification task, Early Stopping is based on total validation loss
            #      pass


        else:
             print("\n跳过验证，因为验证集 DataLoader 未创建。")
             # 如果没有验证集，早停功能依赖于验证指标，因此无法进行早停。模型将训练满 EPOCHS。


    total_training_end_time = time.time()
    total_training_duration = total_training_end_time - total_training_start_time
    print(f"\n所有 Epoch 训练完成！总训练耗时: {total_training_duration:.2f}秒")


    # --- 可视化训练和验证损失曲线 ---
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_total_losses) + 1), train_total_losses, label='Training Total Loss')
    if val_loader and val_total_losses:
        plt.plot(range(1, len(val_total_losses) + 1), val_total_losses, label='Validation Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss over Epochs')
    plt.legend()
    plt.grid(True)


    # --- 可视化训练和验证分类损失曲线 ---
    if OUTPUT_CLASSIFICATION_DIM > 0:
         plt.figure(figsize=(10, 6))
         plt.plot(range(1, len(train_cls_losses) + 1), train_cls_losses, label='Training Classification Loss')
         if val_loader and val_cls_losses:
              plt.plot(range(1, len(val_cls_losses) + 1), val_cls_losses, label='Validation Classification Loss')
         plt.xlabel('Epoch')
         plt.ylabel('Loss')
         plt.title('Classification Loss over Epochs')
         plt.legend()
         plt.grid(True)


    # --- 可视化训练和验证分类准确率曲线 ---
    if OUTPUT_CLASSIFICATION_DIM > 0:
         plt.figure(figsize=(10, 6))
         plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
         if val_loader and val_accuracies:
              valid_epochs_for_val_accuracy = [e for e in range(len(val_accuracies)) if not np.isnan(val_accuracies[e])]
              if valid_epochs_for_val_accuracy:
                  # Corrected list comprehension for plotting
                  plt.plot([e + 1 for e in valid_epochs_for_val_accuracy], [val_accuracies[e] for e in valid_epochs_for_val_accuracy], label='Validation Accuracy')
              else:
                  plt.title('Validation Accuracy over Epochs (No valid data)')
         plt.xlabel('Epoch')
         plt.ylabel('Accuracy')
         plt.title('Classification Accuracy over Epochs')
         plt.legend()
         plt.grid(True)


    # --- 可视化验证集 F1 分数曲线 ---
    # Only plot if validation was performed and classification task exists and data exists
    if val_loader and OUTPUT_CLASSIFICATION_DIM > 0 and val_f1_scores:
         plt.figure(figsize=(10, 6))
         # Filter out NaN values for plotting F1
         # CORRECTED list comprehension for plotting
         valid_epochs_for_f1 = [e for e in range(len(val_f1_scores)) if not np.isnan(val_f1_scores[e])]
         if valid_epochs_for_f1:
             # Use the valid epoch indices to access the corresponding F1 scores
             plt.plot([e + 1 for e in valid_epochs_for_f1], [val_f1_scores[e] for e in valid_epochs_for_f1], label='Validation F1 Score')
             plt.xlabel('Epoch')
             plt.ylabel('F1 Score')
             plt.title('Validation F1 Score over Epochs')
             plt.legend()
             plt.grid(True)
         else:
              plt.title('Validation F1 Score over Epochs (No valid data)')


    # --- 测试阶段 (如果 test_loader 存在) ---
    test_reg_preds_flat = np.array([])
    test_reg_targets_flat = np.array([])
    test_cls_preds_flat = np.array([])
    test_cls_targets_flat = np.array([])
    test_original_labels_np = np.array([])
    test_batch_dates_list = []

    if test_loader:
        # 删除加载最佳模型的代码:
        # print(f"\n开始在测试集上评估最终模型 (Epoch {best_epoch if best_epoch != -1 else 'Last'})...")
        # if os.path.exists(best_model_path):
        #     try:
        #         model.load_state_dict(torch.load(best_model_path, map_location=device))
        #         print("已加载最优模型状态字典。")
        #     except Exception as e:
        #          print(f"错误：加载最优模型文件 {best_model_path} 时出错: {e}。将使用训练结束时的模型进行测试。")
        # else:
        #     print(f"警告：未找到最优模型文件 {best_model_path}。将使用训练结束时的模型进行测试。")
        # 修改打印信息，表明使用训练结束时的模型
        print(f"\n开始在测试集上评估训练结束时的模型...")


        # evaluate_model returns 13 values
        test_loss, test_reg_loss, test_cls_loss, test_reg_mse, test_reg_r2, test_accuracy, test_f1, \
        test_reg_preds_flat, test_reg_targets_flat, test_cls_preds_flat, test_cls_targets_flat, \
        test_original_labels_np, test_batch_dates_list = evaluate_model(
            model, test_loader, criterion_reg, criterion_cls, device, desc="Evaluating Test"
        )

        print("\n--- 测试集最终评估结果总结 ---")
        print(f"总损失: {test_loss:.4f}")
        print(f"回归损失: {test_reg_loss:.4f}")
        print(f"分类损失: {test_cls_loss:.4f}")
        if not np.isnan(test_reg_mse): print(f"回归 MSE: {test_reg_mse:.4f}")
        if not np.isnan(test_reg_r2): print(f"回归 Avg R2: {test_reg_r2:.4f}")
        if not np.isnan(test_accuracy): print(f"分类 Accuracy: {test_accuracy:.4f}")
        if not np.isnan(test_f1): print(f"分类 F1 (二分类，正类=1): {test_f1:.4f}")
        print("-----------------------------")

        # --- 添加测试集分类结果可视化 (混淆矩阵和分类报告) ---
        if OUTPUT_CLASSIFICATION_DIM > 0 and test_cls_targets_flat.size > 0:
             print("\n--- 测试集分类报告 ---")
             # Generate classification report
             unique_test_labels = np.unique(test_cls_targets_flat)
             target_names = [CLASSIFICATION_LABELS[i] if CLASSIFICATION_LABELS and i < len(CLASSIFICATION_LABELS) else f'Class {i}' for i in unique_test_labels]

             # Filter predictions to match the range of target labels present
             valid_pred_mask = np.isin(test_cls_preds_flat, unique_test_labels)
             test_cls_preds_filtered_for_report = test_cls_preds_flat[valid_pred_mask]
             test_cls_targets_filtered_for_report = test_cls_targets_flat[valid_pred_mask]


             if test_cls_targets_filtered_for_report.size > 0:
                  try:
                      # Ensure labels parameter contains all possible labels that might appear in either true or pred
                      all_labels_in_data = np.unique(np.concatenate((test_cls_targets_filtered_for_report, test_cls_preds_filtered_for_report)))
                      # Use labels from config for target names if available
                      report_target_names = [CLASSIFICATION_LABELS[int(label)] if CLASSIFICATION_LABELS and int(label) < len(CLASSIFICATION_LABELS) else f'Class {int(label)}' for label in all_labels_in_data]

                      print(classification_report(test_cls_targets_filtered_for_report, test_cls_preds_filtered_for_report, labels=all_labels_in_data, target_names=report_target_names, zero_division=0))
                  except Exception as e:
                      print(f"警告：生成分类报告时出错: {e}")
                      # print("Raw test_cls_targets_flat:", np.unique(test_cls_targets_flat))
                      # print("Raw test_cls_preds_flat:", np.unique(test_cls_preds_flat))
                      # print("Unique test labels:", unique_test_labels)


             # Confusion matrix is already plotted within evaluate_model
             pass # No need to replot


        # --- 添加测试集回归预测可视化 ---
        if OUTPUT_REGRESSION_DIM > 0 and test_reg_preds_flat.size > 0:
             # Select a few sample stocks for detailed plotting
             num_sample_stocks = 5
             sample_stock_codes = []
             if stock_pool:
                  sample_stock_indices = random.sample(range(NUM_STOCKS), min(num_sample_stocks, NUM_STOCKS))
                  sample_stock_codes = [stock_pool[i] for i in sample_stock_indices]
                  print(f"\n绘制以下 {len(sample_stock_codes)} 只股票的回归预测对比 ({', '.join(sample_stock_codes)}):")
             else:
                  print("\n警告：股票池列表不可用，跳过特定股票的回归预测可视化。")


             if sample_stock_codes and test_loader.dataset and hasattr(test_loader.dataset, 'trading_days'):
                 test_trading_days = test_loader.dataset.trading_days
                 # Need stock pool from the test dataset to map codes to indices
                 test_dataset_stock_pool = test_loader.dataset.stock_pool if hasattr(test_loader.dataset, 'stock_pool') else stock_pool # Use test dataset pool if available, fallback to train pool

                 if test_dataset_stock_pool:
                      # Reconstruct dates corresponding to the flattened predictions/targets
                      prediction_dates_for_test = []
                      # The flattened data is indexed by (window_idx * NumStocks + stock_idx)
                      # The batch_dates_list contains lists of dates for each window in the processed order
                      # Flatten the list of lists of dates to get dates per time step across all windows
                      all_dates_flattened_timesteps = [date_obj for window_dates_list in test_batch_dates_list for date_obj in window_dates_list]

                      # The prediction date is for the PREDICTION_TIME_STEP within each window
                      # We need the date corresponding to index (window_idx * TIME_WINDOW_SIZE + PREDICTION_TIME_STEP)
                      # in the all_dates_flattened_timesteps list, then repeat it NUM_STOCKS times for the flattened prediction/target array
                      num_processed_windows = len(test_batch_dates_list) # Number of windows processed in DataLoader run

                      for i in range(num_processed_windows):
                           # Get the date for the prediction time step in the i-th processed window
                           date_index_in_flattened_timesteps = i * TIME_WINDOW_SIZE + PREDICTION_TIME_STEP
                           if date_index_in_flattened_timesteps < len(all_dates_flattened_timesteps):
                                prediction_date = all_dates_flattened_timesteps[date_index_in_flattened_timesteps]
                                prediction_dates_for_test.extend([prediction_date] * NUM_STOCKS) # Repeat for each stock in the window
                           # else: date index out of bounds, should not happen with correct data


                      dates_for_flattened_data = np.array(prediction_dates_for_test)


                      # Plot for each sample stock
                      for stock_code in sample_stock_codes:
                          if stock_code in test_dataset_stock_pool:
                               stock_idx = test_dataset_stock_pool.index(stock_code) # Get the index of the sample stock

                               # Get the flattened indices corresponding to this stock across all windows
                               flattened_indices_for_stock = np.arange(stock_idx, test_reg_preds_flat.shape[0], NUM_STOCKS)

                               if flattened_indices_for_stock.size > 0:
                                    # Extract predictions and targets for this stock
                                    preds_for_stock = test_reg_preds_flat[flattened_indices_for_stock, :]
                                    targets_for_stock = test_reg_targets_flat[flattened_indices_for_stock, :]
                                    dates_for_stock = dates_for_flattened_data[flattened_indices_for_stock]


                                    # Sort by date to plot a sensible time series
                                    if dates_for_stock.size > 0:
                                         sort_indices = np.argsort(dates_for_stock)
                                         sorted_dates = [dates_for_stock[i] for i in sort_indices]
                                         sorted_preds = preds_for_stock[sort_indices]
                                         sorted_targets = targets_for_stock[sort_indices]


                                         # Plot each regression target for this stock
                                         for i in range(OUTPUT_REGRESSION_DIM):
                                              plt.figure(figsize=(12, 6))
                                              plt.plot(sorted_dates, sorted_targets[:, i], label=f'real {REGRESSION_LABELS[i] if REGRESSION_LABELS and i < len(REGRESSION_LABELS) else f"Target {i+1}"}', marker='o', linestyle='-')
                                              plt.plot(sorted_dates, sorted_preds[:, i], label=f'pre {REGRESSION_LABELS[i] if REGRESSION_LABELS and i < len(REGRESSION_LABELS) else f"Target {i+1}"}', marker='x', linestyle='--')

                                              plt.xlabel('datetime')
                                              plt.ylabel(REGRESSION_LABELS[i] if REGRESSION_LABELS and i < len(REGRESSION_LABELS) else f'Target {i+1}')
                                              plt.title(f'{stock_code} comparison: {REGRESSION_LABELS[i] if REGRESSION_LABELS and i < len(REGRESSION_LABELS) else f"Target {i+1}"}')
                                              plt.legend()
                                              plt.grid(True)
                                              plt.xticks(rotation=45)
                                              plt.tight_layout()

                                    else:
                                         print(f"警告：股票 {stock_code} 在测试集上没有有效的日期数据用于可视化。")
                               else:
                                    print(f"警告：股票 {stock_code} 在测试集上没有找到对应的扁平化数据索引用于可视化。")

                          else:
                               print(f"警告：样本股票代码 {stock_code} 不在测试集 Dataset 的股票池中，跳过可视化。")

                 else:
                      print("警告：测试集 Dataset 的股票池属性不可用，跳过特定股票的回归预测可视化。")

             # Also keep the overall scatter plot for general performance view
             for i in range(OUTPUT_REGRESSION_DIM):
                  plt.figure(figsize=(8, 8))
                  plt.scatter(test_reg_targets_flat[:, i], test_reg_preds_flat[:, i], alpha=0.5, label='All Predictions')
                  min_val = np.min(test_reg_targets_flat[:, i][np.isfinite(test_reg_targets_flat[:, i])]) if np.isfinite(test_reg_targets_flat[:, i]).any() else 0
                  max_val = np.max(test_reg_targets_flat[:, i][np.isfinite(test_reg_targets_flat[:, i])]) if np.isfinite(test_reg_targets_flat[:, i]).any() else 1
                  min_pred = np.min(test_reg_preds_flat[:, i][np.isfinite(test_reg_preds_flat[:, i])]) if np.isfinite(test_reg_preds_flat[:, i]).any() else 0
                  max_pred = np.max(test_reg_preds_flat[:, i][np.isfinite(test_reg_preds_flat[:, i])]) if np.isfinite(test_reg_preds_flat[:, i]).any() else 1
                  plot_min = min(min_val, min_pred)
                  plot_max = max(max_val, max_pred)


                  plt.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', label='Ideal Prediction')
                  plt.xlabel(f'real (Target {i+1})')
                  plt.ylabel(f'pre (Prediction {i+1})')
                  target_label = REGRESSION_LABELS[i] if REGRESSION_LABELS and i < len(REGRESSION_LABELS) else f'Target {i+1}'
                  plt.title(f'test regression: {target_label} (real vs pre - AllStocks)')
                  plt.legend()
                  plt.grid(True)
                  plt.axis('equal') # Make scales equal for better visual comparison to y=x line
                  plt.xlim(plot_min, plot_max)
                  plt.ylim(plot_min, plot_max)


             else:
                  print("\n跳过测试集回归预测可视化，因为没有回归任务或数据。")


    else:
         print("\n跳过测试，因为测试集 DataLoader 未创建。")


    # --- 在脚本的最后统一显示所有 matplotlib 图形 ---
    print("\n显示所有生成的图表...")
    plt.show()


# ==================== 执行入口点 ====================
if __name__ == "__main__":
    main()