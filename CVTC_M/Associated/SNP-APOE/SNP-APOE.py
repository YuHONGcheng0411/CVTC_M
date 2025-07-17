import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

# 设置文件夹路径
folder_path = r"F:\SNP+APOE+类别"

# 初始化结果列表
results = []

# 初始化 LabelEncoder 用于 genotype 编码
label_encoder = LabelEncoder()

# 定义ApoE基因型编码函数
def encode_apoe_genotype(genotype):
    if genotype in ['2/2', '2/3', '3/3']:
        return 0  # 无ε4
    elif genotype in ['2/4', '3/4']:
        return 1  # 1个ε4
    elif genotype == '4/4':
        return 2  # 2个ε4
    else:
        return np.nan

# 遍历文件夹中的CSV文件
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        snp_name = file_name.replace(".csv", "")
        data = pd.read_csv(os.path.join(folder_path, file_name))

        # 排除Unknown类别的样本
        data = data[data['ADNICategory'] != 'Unknown']
        print(f"{snp_name}: After filtering Unknown, data shape = {data.shape}")

        # 将ADNICategory简化为二分类：AD vs. Non-AD (MCI+Normal)
        data['ADNICategory_binary'] = data['ADNICategory'].apply(lambda x: 1 if x == 'AD' else 0)

        # 选择APOE特征
        apoe_columns = ['GENOTYPE'] + list(data.loc[:, 'APVOLUME':'SITEID'].columns)
        X_apoe = data[apoe_columns].copy()

        # 对GENOTYPE进行生物学编码
        X_apoe['GENOTYPE_encoded'] = X_apoe['GENOTYPE'].apply(encode_apoe_genotype)
        X_apoe = X_apoe.dropna(subset=['GENOTYPE_encoded'])
        data = data.loc[X_apoe.index]  # 同步更新data的索引
        print(f"{snp_name}: After dropping NaN in GENOTYPE_encoded, X_apoe shape = {X_apoe.shape}, data shape = {data.shape}")

        # 对APOE特征进行PCA降维
        exclude_apoe_columns = ['APVOLUME', 'ID', 'SITEID', 'GENOTYPE']
        apoe_pca_columns = [col for col in apoe_columns if col not in exclude_apoe_columns]

        if apoe_pca_columns:
            X_apoe_pca = X_apoe[apoe_pca_columns].copy()
            variances = X_apoe_pca.var()
            valid_apoe_pca_columns = variances[variances > 1e-10].index.tolist()

            if valid_apoe_pca_columns:
                X_apoe_pca = X_apoe_pca[valid_apoe_pca_columns]
                scaler_apoe_pca = StandardScaler()
                try:
                    X_apoe_pca_scaled = scaler_apoe_pca.fit_transform(X_apoe_pca)
                    pca_apoe = PCA(n_components=0.8)  # 保留80%方差
                    X_apoe_pca_transformed = pca_apoe.fit_transform(X_apoe_pca_scaled)
                    for i in range(X_apoe_pca_transformed.shape[1]):
                        X_apoe[f'APOE_PCA_component_{i+1}'] = X_apoe_pca_transformed[:, i]
                    final_apoe_columns = ['APVOLUME', 'ID', 'SITEID', 'GENOTYPE_encoded'] + [f'APOE_PCA_component_{i+1}' for i in range(X_apoe_pca_transformed.shape[1])]
                except Exception as e:
                    print(f"Error in APOE PCA for {snp_name}: {e}, skipping PCA")
                    final_apoe_columns = ['APVOLUME', 'ID', 'SITEID', 'GENOTYPE_encoded']
            else:
                print(f"Warning: No valid APOE PCA features for {snp_name}, skipping PCA")
                final_apoe_columns = ['APVOLUME', 'ID', 'SITEID', 'GENOTYPE_encoded']
        else:
            final_apoe_columns = ['APVOLUME', 'ID', 'SITEID', 'GENOTYPE_encoded']

        # 选择SNP特征
        snp_columns = data.loc[:, 'genotype':'Log R Ratio_scaled'].columns
        X_snp = data[snp_columns].copy()

        # 对genotype进行编码
        X_snp['genotype_encoded'] = label_encoder.fit_transform(X_snp['genotype'])

        # 对SNP特征进行PCA降维
        exclude_snp_columns = ['genotype', 'B Allele Freq_scaled', 'Log R Ratio_scaled']
        snp_pca_columns = [col for col in snp_columns if col not in exclude_snp_columns]

        if snp_pca_columns:
            X_snp_pca = X_snp[snp_pca_columns].copy()
            variances = X_snp_pca.var()
            valid_snp_pca_columns = variances[variances > 1e-10].index.tolist()

            if valid_snp_pca_columns:
                X_snp_pca = X_snp_pca[valid_snp_pca_columns]
                scaler_snp_pca = StandardScaler()
                try:
                    X_snp_pca_scaled = scaler_snp_pca.fit_transform(X_snp_pca)
                    pca_snp = PCA(n_components=0.8)  # 保留80%方差
                    X_snp_pca_transformed = pca_snp.fit_transform(X_snp_pca_scaled)
                    for i in range(X_snp_pca_transformed.shape[1]):
                        X_snp[f'SNP_PCA_component_{i+1}'] = X_snp_pca_transformed[:, i]
                    final_snp_columns = ['genotype_encoded', 'B Allele Freq_scaled', 'Log R Ratio_scaled'] + [f'SNP_PCA_component_{i+1}' for i in range(X_snp_pca_transformed.shape[1])]
                except Exception as e:
                    print(f"Error in SNP PCA for {snp_name}: {e}, skipping PCA")
                    final_snp_columns = ['genotype_encoded', 'B Allele Freq_scaled', 'Log R Ratio_scaled']
            else:
                print(f"Warning: No valid SNP PCA features for {snp_name}, skipping PCA")
                final_snp_columns = ['genotype_encoded', 'B Allele Freq_scaled', 'Log R Ratio_scaled']
        else:
            final_snp_columns = ['genotype_encoded', 'B Allele Freq_scaled', 'Log R Ratio_scaled']

        # 合并X_snp和X_apoe，确保索引一致
        X = pd.concat([X_snp[final_snp_columns], X_apoe[final_apoe_columns]], axis=1, join='inner')
        print(f"{snp_name}: After concat, X shape = {X.shape}")

        # 确保X和data的索引对齐
        common_indices = X.index.intersection(data.index)
        if len(common_indices) == 0:
            print(f"Error: No common indices between X and data for {snp_name}, skipping")
            results.append((snp_name, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue

        # 根据共同索引重新选择X和y
        X = X.loc[common_indices]
        y = data['ADNICategory_binary'].loc[common_indices]
        print(f"{snp_name}: After index alignment, X shape = {X.shape}, y shape = {y.shape}, common indices = {len(common_indices)}")

        # 检查样本量是否足够
        if len(y) < 5:
            print(f"Error: Sample size too small for {snp_name} ({len(y)} samples), skipping")
            results.append((snp_name, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue

        # 标准化特征
        scaler = StandardScaler()
        try:
            X_scaled = scaler.fit_transform(X)
        except Exception as e:
            print(f"Error in standardization for {snp_name}: {e}, skipping")
            results.append((snp_name, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue

        # SVM分类（优化超参数）
        param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
        grid_search = GridSearchCV(SVC(kernel='rbf', random_state=42), param_grid, cv=5)
        try:
            grid_search.fit(X_scaled, y)
            svm_accuracy = grid_search.best_score_
        except Exception as e:
            print(f"Error computing SVM accuracy for {snp_name}: {e}")
            svm_accuracy = np.nan

        # 添加交互项并重新标准化
        X['genotype_APOE_interaction'] = X['genotype_encoded'] * X['GENOTYPE_encoded']
        try:
            X_scaled = scaler.fit_transform(X)  # 重新标准化包含交互项的X
        except Exception as e:
            print(f"Error in standardization after adding interaction for {snp_name}: {e}, skipping")
            results.append((snp_name, svm_accuracy, np.nan, np.nan, np.nan, np.nan))
            continue

        # 随机森林模型
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_scaled, y)
            # 检查交互项是否在特征中并获取其重要性
            if 'genotype_APOE_interaction' in X.columns:
                interaction_idx = X.columns.get_loc('genotype_APOE_interaction')
                interaction_importance = rf.feature_importances_[interaction_idx]
            else:
                print(f"Warning: Interaction term not found in X columns for {snp_name}")
                interaction_importance = np.nan
            # 使用交叉验证评估交互效应
            cv_folds = min(5, len(y))  # 动态调整折数
            rf_scores = cross_val_score(rf, X_scaled, y, cv=cv_folds, scoring='accuracy')
            interaction_score = np.mean(rf_scores)  # 作为交互效应的代理
        except Exception as e:
            print(f"Error in random forest for {snp_name}: {e}")
            interaction_importance, interaction_score = np.nan, np.nan

        # 存储结果
        results.append((snp_name, svm_accuracy, interaction_importance, interaction_score, abs(interaction_importance) if not np.isnan(interaction_importance) else np.nan, np.nan))
        print(f"{snp_name}: SVM Accuracy = {svm_accuracy:.4f}, Interaction Importance = {interaction_importance:.4f}, "
              f"Interaction Score = {interaction_score:.4f}")

# FDR校正交互分数
interaction_scores = [r[3] for r in results if not np.isnan(r[3])]
if interaction_scores:
    reject, score_corrected, _, _ = multipletests(interaction_scores, method='fdr_bh')
    score_corrected_dict = dict(zip([r[0] for r in results if not np.isnan(r[3])], score_corrected))
    results = [(snp, acc, imp, score_corrected_dict.get(snp, score), abs_imp, combined) for snp, acc, imp, score, abs_imp, combined in results]

# 计算综合排名
scores = [r[3] for r in results if not np.isnan(r[3])]
accuracies = [r[1] for r in results if not np.isnan(r[1])]
imp_abs_values = [r[4] for r in results if not np.isnan(r[4])]

if len(scores) > 0 and len(accuracies) > 0 and len(imp_abs_values) > 0:
    score_min, score_max = min(scores), max(scores)
    acc_min, acc_max = min(accuracies), max(accuracies)
    imp_abs_min, imp_abs_max = min(imp_abs_values), max(imp_abs_values)

    score_range = score_max - score_min if score_max != score_min else 1.0
    acc_range = acc_max - acc_min if acc_max != acc_min else 1.0
    imp_abs_range = imp_abs_max - imp_abs_min if imp_abs_max != imp_abs_min else 1.0

    for i, (snp, acc, imp, score, abs_imp, _) in enumerate(results):
        norm_score = (score_max - score) / score_range if not np.isnan(score) else 0.0
        norm_acc = (acc - acc_min) / acc_range if not np.isnan(acc) else 0.0
        norm_imp_abs = (abs_imp - imp_abs_min) / imp_abs_range if not np.isnan(abs_imp) else 0.0
        combined_score = 0.2 * norm_score + 0.5 * norm_acc + 0.3 * norm_imp_abs
        results[i] = (snp, acc, imp, score, abs_imp, combined_score)
else:
    results = [(snp, acc, imp, score, abs_imp, np.nan) for snp, acc, imp, score, abs_imp, _ in results]

# 按综合得分排序
results.sort(key=lambda x: x[5] if not np.isnan(x[5]) else float('-inf'), reverse=True)

# 打印结果
print("\nComprehensive SNP-APOE Ranking:")
print("Rank\tSNP\t\tSVM Accuracy\tInteraction Importance\tInteraction Score\tInteraction Importance Abs\tCombined Score")
for i, (snp, acc, imp, score, abs_imp, combined_score) in enumerate(results, 1):
    print(f"{i}\t{snp:<15}\t{acc:.4f}\t\t{imp:.4f}\t\t{score:.4f}\t\t{abs_imp:.4f}\t\t{combined_score:.4f}")

# # 保存结果
# results_df = pd.DataFrame(results, columns=['SNP', 'SVM_Accuracy', 'Interaction_Importance', 'Interaction_Score', 'Interaction_Importance_Abs', 'Combined_Score'])
# results_df.insert(0, 'Rank', range(1, len(results_df) + 1))
# results_df.to_csv('SNP_APOE_Ranking_random_forest_no_smote.csv', index=False)