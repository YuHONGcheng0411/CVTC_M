import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# 设置文件夹路径（替换为实际路径）
folder_path = r"F:\SNP+蛋白+类别"

# 初始化结果列表
results = []

# 定义蛋白特征列
protein_features = ['pT217_F', 'AB42_F', 'AB40_F', 'AB42_AB40_F', 'pT217_AB42_F', 'NfL_Q', 'GFAP_Q']

# 检查多重共线性
def check_collinearity(X):
    """检查特征矩阵的条件数以检测多重共线性"""
    try:
        cond_number = np.linalg.cond(X)
        return cond_number < 1e5
    except:
        return False

# 移除零方差和低方差特征
def remove_low_variance_features(X, threshold=1e-5):
    """移除方差低于阈值的特征"""
    variances = X.var()
    low_variance_cols = variances[variances < threshold].index
    if low_variance_cols.any():
        print(f"移除低方差特征: {low_variance_cols.tolist()}")
        X = X.drop(columns=low_variance_cols)
    return X

# 遍历文件夹中的CSV文件
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        snp_name = file_name.replace(".csv", "")
        data = pd.read_csv(os.path.join(folder_path, file_name))

        # 排除Unknown类别的样本
        data = data[data['ADNICategory'] != 'Unknown']

        # 将ADNICategory简化为二分类：AD vs. Non-AD (MCI+Normal)
        data['ADNICategory_binary'] = data['ADNICategory'].apply(lambda x: 1 if x == 'AD' else 0)

        # 检查必要列是否存在
        required_cols = ['genotype', 'B Allele Freq_scaled', 'Log R Ratio_scaled',
                        'ADNICategory_binary'] + protein_features
        if not all(col in data.columns for col in required_cols):
            print(f"文件 {file_name} 缺少必要列，跳过")
            continue

        # 获取从 'genotype' 到 'Log R Ratio_scaled' 的所有列
        try:
            start_col = data.columns.get_loc('genotype')
            end_col = data.columns.get_loc('Log R Ratio_scaled') + 1
            all_features = data.columns[start_col:end_col].tolist()
            exclude_cols = ['genotype', 'B Allele Freq_scaled', 'Log R Ratio_scaled']
            one_hot_features = [col for col in all_features if col not in exclude_cols]
            snp_features = ['genotype', 'B Allele Freq_scaled', 'Log R Ratio_scaled']
        except KeyError:
            print(f"文件 {file_name} 缺少 'genotype' 或 'Log R Ratio_scaled' 列，跳过")
            continue

        # 针对每个蛋白特征进行分析
        for protein in protein_features:
            # 选择SNP特征和当前蛋白特征
            X = data[snp_features + one_hot_features + [protein]].copy()
            y = data['ADNICategory_binary']

            # 数据清理：移除缺失值和无穷值
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.dropna()
            if X.empty or len(X) < 10:
                print(f"{snp_name} 与 {protein} 的数据不足或无效，跳过")
                results.append((snp_name, protein, np.nan, np.nan, np.nan, np.nan, np.nan))
                continue

            # 将 SNP 整体降维为单一特征
            snp_cols = snp_features + one_hot_features
            if snp_cols:
                # 检查 SNP 特征的方差
                X_snp = X[snp_cols]
                variances = X_snp.var()
                valid_snp_features = variances[variances > 1e-5].index.tolist()

                if valid_snp_features:
                    try:
                        scaler_snp = StandardScaler()
                        X_snp_scaled = scaler_snp.fit_transform(X[valid_snp_features])
                        pca_snp = PCA(n_components=1)
                        X_snp_pca = pca_snp.fit_transform(X_snp_scaled)
                        explained_variance = pca_snp.explained_variance_ratio_[0]
                        print(f"{snp_name} SNP 整体 PCA 降维，保留方差比例: {explained_variance:.4f}")
                        # 创建新特征矩阵，仅包含 SNP 的 PCA 特征和蛋白特征
                        X = X[[protein]].copy()
                        X['SNP_PCA'] = X_snp_pca
                    except Exception as e:
                        print(f"PCA 降维 {snp_name} 的 SNP 特征时出错: {e}")
                        print(f"SNP 特征方差: {variances.to_dict()}")
                        X = X[[protein]].copy()
                        X['SNP_PCA'] = 0  # 用零填充
                else:
                    print(f"{snp_name} SNP 特征全为零方差，跳过 PCA")
                    X = X[[protein]].copy()
                    X['SNP_PCA'] = 0  # 用零填充
            else:
                print(f"{snp_name} 无 SNP 特征，跳过 PCA")
                X = X[[protein]].copy()
                X['SNP_PCA'] = 0

            # 移除低方差特征
            X = remove_low_variance_features(X, threshold=1e-5)

            # 标准化特征（SVM需要）
            scaler = StandardScaler()
            try:
                X_scaled = scaler.fit_transform(X)
            except Exception as e:
                print(f"标准化 {snp_name} 与 {protein} 的特征时出错: {e}")
                results.append((snp_name, protein, np.nan, np.nan, np.nan, np.nan, np.nan))
                continue

            # 计算SVM分类准确率（5折交叉验证）
            try:
                svm = SVC(kernel='rbf', random_state=42)
                svm_accuracy = np.mean(cross_val_score(svm, X_scaled, y.loc[X.index], cv=5))
            except Exception as e:
                print(f"处理 {snp_name} 与 {protein} 的SVM准确率时出错: {e}")
                svm_accuracy = np.nan

            # 添加 SNP 整体与蛋白的交互项
            X['SNP_protein_interaction'] = X['SNP_PCA'] * X[protein]

            # 再次移除低方差交互项
            X = remove_low_variance_features(X, threshold=1e-5)

            # 检查多重共线性
            X = sm.add_constant(X)
            if not check_collinearity(X):
                print(f"{snp_name} 与 {protein} 的特征矩阵存在多重共线性，使用正则化逻辑回归")
                try:
                    lr = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
                    lr.fit(X.drop(columns=['const']), y.loc[X.index])
                    coef_interaction = lr.coef_[0][X.columns.get_loc('SNP_protein_interaction') - 1]
                    p_value_interaction = np.nan
                except Exception as e:
                    print(f"正则化逻辑回归 {snp_name} 与 {protein} 时出错: {e}")
                    coef_interaction = np.nan
                    p_value_interaction = np.nan
            else:
                try:
                    model = sm.Logit(y.loc[X.index], X).fit(disp=0, maxiter=100)
                    coef_interaction = model.params['SNP_protein_interaction']
                    p_value_interaction = model.pvalues['SNP_protein_interaction']
                except Exception as e:
                    print(f"处理 {snp_name} 与 {protein} 的逻辑回归时出错: {e}")
                    coef_interaction = np.nan
                    p_value_interaction = np.nan

            # 存储结果（添加交互系数的绝对值）
            results.append((snp_name, protein, svm_accuracy, coef_interaction, p_value_interaction,
                           explained_variance if 'explained_variance' in locals() else np.nan,
                           abs(coef_interaction) if not np.isnan(coef_interaction) else np.nan))
            print(f"{snp_name} 与 {protein}: SVM准确率 = {svm_accuracy:.4f}, "
                  f"SNP-蛋白交互系数 = {coef_interaction:.4f}, p值 = {p_value_interaction:.4f}, "
                  f"交互系数绝对值 = {abs(coef_interaction):.4f}")

# 计算综合排名
p_values = [r[4] for r in results if not np.isnan(r[4])]
accuracies = [r[2] for r in results if not np.isnan(r[2])]
explained_variances = [r[5] for r in results if not np.isnan(r[5])]
coef_abs_values = [r[6] for r in results if not np.isnan(r[6])]

if len(p_values) > 0 and len(accuracies) > 0 and len(explained_variances) > 0 and len(coef_abs_values) > 0:
    p_min, p_max = min(p_values), max(p_values)
    acc_min, acc_max = min(accuracies), max(accuracies)
    var_min, var_max = min(explained_variances), max(explained_variances)
    coef_abs_min, coef_abs_max = min(coef_abs_values), max(coef_abs_values)

    p_range = p_max - p_min if p_max != p_min else 1.0
    acc_range = acc_max - acc_min if acc_max != acc_min else 1.0
    var_range = var_max - var_min if var_max != var_min else 1.0
    coef_abs_range = coef_abs_max - coef_abs_min if coef_abs_max != coef_abs_min else 1.0

    for i, (snp, protein, acc, coef, p, var, coef_abs) in enumerate(results):
        norm_p = (p_max - p) / p_range if not np.isnan(p) else 0.0
        norm_acc = (acc - acc_min) / acc_range if not np.isnan(acc) else 0.0
        norm_var = (var - var_min) / var_range if not np.isnan(var) else 0.0
        norm_coef_abs = (coef_abs - coef_abs_min) / coef_abs_range if not np.isnan(coef_abs) else 0.0
        # 调整权重：p值25%，SVM准确率30%，PCA方差比例25%，交互系数绝对值20%
        combined_score = 0.25 * norm_p + 0.30 * norm_acc + 0.25 * norm_var + 0.20 * norm_coef_abs
        results[i] = (snp, protein, acc, coef, p, var, coef_abs, combined_score)
else:
    results = [(snp, protein, acc, coef, p, var, coef_abs, np.nan)
               for snp, protein, acc, coef, p, var, coef_abs in results]

# 按综合得分排序（降序）
results.sort(key=lambda x: x[7] if not np.isnan(x[7]) else float('-inf'), reverse=True)

# 打印综合排名
print("\nSNP-蛋白综合排名:")
print("排名\tSNP\t\t蛋白\t\tSVM准确率\tSNP-蛋白交互系数\tp值\tPCA方差比例\t交互系数绝对值\t综合得分")
for i, (snp, protein, acc, coef, p, var, coef_abs, score) in enumerate(results, 1):
    print(
        f"{i}\t{snp:<15}\t{protein:<15}\t{acc:.4f}\t\t{coef:.4f}\t\t{p:.4f}\t\t{var:.4f}\t\t{coef_abs:.4f}\t\t{score:.4f}")

# 将结果保存为 CSV 文件
results_df = pd.DataFrame(results, columns=['SNP', 'Protein', 'SVM_Accuracy', 'Interaction_Coefficient',
                                           'P_Value', 'PCA_Variance', 'Interaction_Coefficient_Abs', 'Combined_Score'])
results_df.to_csv('snp_protein_results_3.csv', index=False)