import pandas as pd
import numpy as np
import os
from pathlib import Path
import statsmodels.api as sm
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import matplotlib.font_manager as fm

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置文件夹路径，包含 87 个 CSV 文件
data_folder = Path(r"F:\SNP+蛋白+类别")  # 你的文件夹路径
csv_files = list(data_folder.glob("*.csv"))

# 定义潜在的 SNP 和蛋白质特征列
potential_snp_columns = ["genotype"] + [f"Chr_{i}" for i in range(1, 21)] + ["Chr_X", "B Allele Freq_scaled", "Log R Ratio_scaled"]
protein_columns = ["pT217_F", "AB42_F", "AB40_F", "AB42_AB40_F", "pT217_AB42_F", "NfL_Q", "GFAP_Q"]
outcome = "ADNICategory"

# 处理单个文件的 MR 分析
def process_file(file):
    print(f"处理文件: {file.name}")
    results = []

    # 提取 SNP 名称（去掉 .csv 后缀）
    snp_name = file.stem

    # 加载数据
    df = pd.read_csv(file)

    # 数据清理：移除缺失值
    df = df.dropna()

    # 将 ADNICategory 转换为二元变量（Normal=0, MCI/AD=1）
    df["AD_binary"] = df[outcome].apply(lambda x: 0 if x == "Normal" else 1)

    # 动态选择存在的 SNP 列
    snp_columns = [col for col in potential_snp_columns if col in df.columns]
    if not snp_columns:
        print(f"错误: {file.name} 中没有可用的 SNP 列，跳过")
        return results

    # 检查 SNP 和蛋白质列是否为数值型
    for col in snp_columns + protein_columns:
        if col not in df.columns:
            print(f"警告: {file.name} 中缺少 {col} 列，跳过该列")
            if col in snp_columns:
                snp_columns.remove(col)
            elif col in protein_columns:
                protein_columns.remove(col)
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"警告: {file.name} 中的 {col} 列包含非数值数据，尝试转换为数值型")
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                print(f"错误: 无法转换 {col} 列为数值型，跳过该列")
                if col in snp_columns:
                    snp_columns.remove(col)
                elif col in protein_columns:
                    protein_columns.remove(col)

    # 对每个蛋白质特征进行 MR 分析
    for exposure in protein_columns:
        if exposure not in df.columns:
            print(f"警告: {file.name} 中缺少 {exposure} 列，跳过")
            continue

        # 筛选工具变量
        snp_effects = []
        for snp in snp_columns:
            try:
                # 检查 SNP 列是否有变化（避免全为相同值）
                if df[snp].nunique() <= 1:
                    print(f"警告: {file.name} 中的 {snp} 列值无变化，跳过")
                    continue

                X = sm.add_constant(df[snp])
                model = sm.OLS(df[exposure], X).fit()
                p_value = model.pvalues.iloc[1]  # 按位置索引
                beta = model.params.iloc[1]      # 按位置索引
                se = model.bse.iloc[1]           # 按位置索引

                # 检查标准误差是否有效
                if se == 0 or np.isnan(se):
                    print(f"警告: {file.name} 中 {snp} 对 {exposure} 的标准误差无效（se={se}），跳过")
                    continue

                f_stat = (beta / se) ** 2  # 计算 F 统计量

                if p_value < 0.05 and f_stat > 10:
                    snp_effects.append({
                        "SNP": snp,
                        "beta_exposure": beta,
                        "se_exposure": se,
                        "p_exposure": p_value,
                        "f_stat": f_stat
                    })
            except Exception as e:
                print(f"错误: {file.name} 中 {snp} 对 {exposure} 的回归失败，原因: {str(e)}")
                continue

        # 如果没有显著的 SNP，跳过该暴露变量
        if not snp_effects:
            print(f"{file.name} 中 {exposure} 没有显著的工具变量，跳过。")
            continue

        # 转换为 DataFrame
        snp_effects_df = pd.DataFrame(snp_effects)

        # 对结局变量（AD_binary）进行逻辑回归
        for index, row in snp_effects_df.iterrows():
            snp = row["SNP"]
            try:
                X = sm.add_constant(df[snp])
                model = sm.Logit(df["AD_binary"], X).fit(disp=0)
                snp_effects_df.loc[index, "beta_outcome"] = model.params.iloc[1]  # 按位置索引
                snp_effects_df.loc[index, "se_outcome"] = model.bse.iloc[1]      # 按位置索引
            except Exception as e:
                print(f"错误: {file.name} 中 {snp} 对 AD_binary 的逻辑回归失败，原因: {str(e)}")
                snp_effects_df.loc[index, "beta_outcome"] = np.nan
                snp_effects_df.loc[index, "se_outcome"] = np.nan

        # 移除无效的 SNP 效应
        snp_effects_df = snp_effects_df.dropna()
        if snp_effects_df.empty:
            continue

        # IVW 方法
        def ivw_mr(beta_exp, se_exp, beta_out, se_out):
            weights = 1 / se_out ** 2
            beta_ivw = np.sum(beta_out * beta_exp * weights) / np.sum(beta_exp ** 2 * weights)
            se_ivw = np.sqrt(1 / np.sum(beta_exp ** 2 * weights))
            return beta_ivw, se_ivw

        try:
            beta_ivw, se_ivw = ivw_mr(
                snp_effects_df["beta_exposure"],
                snp_effects_df["se_exposure"],
                snp_effects_df["beta_outcome"],
                snp_effects_df["se_outcome"]
            )

            # 计算 P 值
            z_score = beta_ivw / se_ivw
            p_value_ivw = 2 * (1 - norm.cdf(abs(z_score)))

            # 存储结果
            results.append({
                "SNP": snp_name,
                "exposure": exposure,
                "beta_ivw": beta_ivw,
                "se_ivw": se_ivw,
                "p_value_ivw": p_value_ivw,
                "n_snps": len(snp_effects_df)
            })
        except Exception as e:
            print(f"错误: {file.name} 中 {exposure} 的 IVW 计算失败，原因: {str(e)}")
            continue

    return results

# 并行处理所有文件
results = Parallel(n_jobs=-1)(delayed(process_file)(file) for file in csv_files)

# 展平结果
results_flat = [item for sublist in results for item in sublist]
results_df = pd.DataFrame(results_flat)

# 保存结果到 CSV 文件
results_df.to_csv( "mr_results.csv", index=False, encoding='utf-8-sig')

# 打印结果
print("\nMR 分析结果汇总：")
print(results_df)

# 可视化汇总结果
plt.figure(figsize=(12, 8))

# 按 exposure 和 SNP 分组，动态调整条形高度
exposures = results_df['exposure'].unique()
n_exposures = len(exposures)
base_height = 0.8 / n_exposures  # 每组条形的基础高度
colors = sns.color_palette("husl", n_exposures)  # 为每种蛋白质分配颜色

# 为每个 SNP 分配唯一的 Y 坐标
snps = results_df['SNP'].unique()
snp_to_y = {snp: idx for idx, snp in enumerate(snps)}

for idx, exposure in enumerate(exposures):
    subset = results_df[results_df['exposure'] == exposure]
    if subset.empty:
        continue
    # 归一化 n_snps 以映射到条形高度（base_height 到 base_height * 2）
    if subset['n_snps'].max() > subset['n_snps'].min():
        heights = base_height + base_height * (subset['n_snps'] - subset['n_snps'].min()) / (subset['n_snps'].max() - subset['n_snps'].min())
    else:
        heights = base_height * np.ones(len(subset))  # 默认高度
    # 计算 Y 位置
    y_positions = [snp_to_y[s] + idx * base_height for s in subset['SNP']]
    plt.barh(y_positions, subset['beta_ivw'], height=heights,
             color=colors[idx], label=exposure, alpha=0.8)

plt.yticks(np.arange(len(snps)) + 0.4, snps)  # 设置 Y 轴刻度为 SNP 名称
plt.title("Two-Sample MR 因果效应估计（所有蛋白质）")
plt.xlabel("IVW Beta (因果效应)")
plt.ylabel("SNP")
plt.legend(title="蛋白质", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()