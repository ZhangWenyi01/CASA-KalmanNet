# CPDNet 绘图工具使用说明

## 功能概述

这个工具用于可视化CPDNet的结果数据，支持：
- 指定特定的pt文件进行绘图
- 自动检测数据结构和内容
- 生成黑白打印友好的图表
- 保存高质量PDF输出

## 使用方法

### 1. 查看可用的数据文件

```bash
python plot_all_CPDNet_results.py --list
```

这会列出`plot_data`文件夹中所有可用的CPDNet结果文件。

### 2. 绘制特定文件

```bash
# 绘制H参数为0.95的结果
python plot_all_CPDNet_results.py --file CPDNet_results_H_0.95.pt

# 或者简写（不需要.pt扩展名）
python plot_all_CPDNet_results.py --file CPDNet_results_H_0.95

# 指定不同的batch索引
python plot_all_CPDNet_results.py --file CPDNet_results_H_0.95 --batch 10

# 不保存PDF文件
python plot_all_CPDNet_results.py --file CPDNet_results_H_0.95 --no-save
```

### 3. 命令行参数说明

- `--file, -f`: 指定要绘制的pt文件名
- `--list, -l`: 列出所有可用的数据文件
- `--batch, -b`: 指定batch索引（默认：6）
- `--no-save`: 不保存PDF文件，只显示图表

## 数据文件结构

每个pt文件应包含以下数据：

- `predicted_cpd`: 预测的CPD概率
- `actual_cpd`: 实际的CPD概率
- `estimation_state`: 估计状态
- `true_state`: 真实状态
- `estimation_y`: 估计观测值
- `true_y`: 真实观测值
- `changepoint`: 变化点位置

## 输出说明

### 图表内容
1. **上半部分**: CPD概率预测结果
   - 预测概率 vs 实际概率
   - 变化点标记

2. **下半部分**: 状态估计结果
   - 估计状态 vs 真实状态
   - 估计观测值 vs 真实观测值
   - 变化点标记

### 文件保存
- 图表自动保存到`CPDNet_plots/`文件夹
- 文件名格式：`CPDNet_{参数类型}_{数值}.pdf`
- 高分辨率输出（300 DPI）

## 示例

```bash
# 查看所有可用文件
python plot_all_CPDNet_results.py --list

# 绘制Q参数为100的结果
python plot_all_CPDNet_results.py --file CPDNet_results_Q_100

# 绘制F参数为1.01的结果，使用batch索引5
python plot_all_CPDNet_results.py --file CPDNet_results_F_1.01 --batch 5
```

## 注意事项

1. 确保pt文件在`plot_data`文件夹中
2. 如果batch索引超出范围，会自动调整到最大值
3. 图表针对黑白打印进行了优化，使用不同的线型和标记
4. 支持自动检测数据结构和完整性
