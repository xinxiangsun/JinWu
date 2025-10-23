
import sys
import subprocess
from typing import Optional, Union, cast, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - ensure 3D projection is available

# 全局字体设置：优先使用可显示中文的字体，避免中文字符缺失警告
matplotlib.rcParams['font.sans-serif'] = [
    'WenQuanYi Zen Hei',  # Ubuntu/Debian 常见
    'Noto Sans CJK SC',   # Google Noto 字体
    'SimHei',             # 黑体
    'Microsoft YaHei',    # 微软雅黑（Windows）
    'DejaVu Sans'         # 兜底
]
matplotlib.rcParams['axes.unicode_minus'] = False

# 尝试在已安装字体中选择一个可用的中文字体，提升中文显示效果
def _ensure_cjk_font():
    try:
        from matplotlib import font_manager
        preferred = [
            'WenQuanYi Zen Hei',
            'Noto Sans CJK SC',
            'Noto Sans CJK JP',
            'Noto Sans CJK TC',
            'Source Han Sans CN',
            'SimHei',
            'Microsoft YaHei'
        ]
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in preferred:
            if name in available:
                matplotlib.rcParams['font.family'] = [name, 'sans-serif']
                break
    except Exception:
        # 静默失败，不影响主流程
        pass

_ensure_cjk_font()

class ClusterAnalyzer:
    """
    一个通用的多维数据聚类分析器。

    封装数据预处理、最佳聚类数评估、K-Means 聚类与 PCA 可视化完整流程。
    """

    def __init__(self, data: Union[pd.DataFrame, np.ndarray]):
        """初始化分析器。"""
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise TypeError("数据必须是 pandas DataFrame 或 numpy ndarray 类型。")
        self.data: pd.DataFrame = pd.DataFrame(data).copy()
        self.scaled_data: Optional[np.ndarray] = None
        self.kmeans_model: Optional[KMeans] = None
        self.labels: Optional[np.ndarray] = None
        print(
            f"ClusterAnalyzer 初始化成功，加载了 {self.data.shape[0]} 条数据，每条数据有 {self.data.shape[1]} 个特征。"
        )

    def _preprocess_data(self) -> None:
        """对数据进行标准化处理。"""
        print("步骤 1: 正在对数据进行标准化...")
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.data)
        print("数据标准化完成。")

    def find_optimal_clusters(self, max_k: int = 10) -> int:
        """使用肘部法则和轮廓系数来寻找最佳的聚类数量 (k)。"""
        if self.scaled_data is None:
            self._preprocess_data()

        # 类型保障
        assert self.scaled_data is not None
        X = cast(np.ndarray, self.scaled_data)

        print(f"\n步骤 2: 正在寻找最佳聚类数量 (k)，最大尝试到 k={max_k}...")
        k_values = list(range(2, max_k + 1))
        inertia_values: list[float] = []
        sil_scores: list[float] = []

        for k in k_values:
            km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
            km.fit(X)
            inertia_values.append(float(km.inertia_))
            sil_scores.append(float(silhouette_score(X, km.labels_)))

        # 绘制评估图
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(k_values, inertia_values, 'bo-')
        plt.xlabel('聚类数量 (k)')
        plt.ylabel('簇内误差平方和 (Inertia)')
        plt.title('肘部法则 (Elbow Method)')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(k_values, sil_scores, 'ro-')
        plt.xlabel('聚类数量 (k)')
        plt.ylabel('轮廓系数 (Silhouette Score)')
        plt.title('轮廓系数法')
        plt.grid(True)

        plt.suptitle('寻找最佳聚类数量 (k) 的评估图', fontsize=16)
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.show()

        best_k = k_values[int(np.argmax(sil_scores))]
        print(f"分析建议: 根据轮廓系数，最佳的 k 值可能是 {best_k} (得分最高)。")
        print("请结合肘部法则图中的'拐点'和轮廓系数图中的峰值，自行决定最终的 k 值。")
        return best_k

    def fit_predict(self, n_clusters: int) -> pd.DataFrame:
        """使用指定的聚类数量执行 K-Means 聚类。"""
        if self.scaled_data is None:
            self._preprocess_data()
        assert self.scaled_data is not None
        X = cast(np.ndarray, self.scaled_data)

        print(f"\n步骤 3: 正在使用 k={n_clusters} 进行K-Means聚类...")
        self.kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
        self.labels = self.kmeans_model.fit_predict(X)

        result = self.data.copy()
        result['cluster'] = self.labels
        print("聚类完成，已将聚类标签添加到原始数据中。")
        return result

    def visualize_clusters(self) -> None:
        """使用 PCA 降维后，将聚类结果可视化。"""
        if self.kmeans_model is None or self.labels is None or self.scaled_data is None:
            raise RuntimeError("请先调用 fit_predict() 方法进行聚类，然后再进行可视化。")

        X = cast(np.ndarray, self.scaled_data)
        y = cast(np.ndarray, self.labels)

        print("\n步骤 4: 正在使用PCA降维并进行可视化...")
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X)

        pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
        pca_df['cluster'] = y
        centroids_pca = pca.transform(cast(KMeans, self.kmeans_model).cluster_centers_)

        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x='PC1', y='PC2',
            hue='cluster',
            palette=sns.color_palette("hsv", n_colors=int(len(np.unique(y)))),
            data=pca_df,
            legend="full",
            alpha=0.7
        )

        plt.scatter(
            centroids_pca[:, 0], centroids_pca[:, 1],
            s=200, c='black', marker='X', label='聚类中心 (Centroids)'
        )

        plt.title('多维数据聚类结果可视化 (PCA降维后)', fontsize=16)
        plt.xlabel(f'主成分 1 (解释方差: {pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'主成分 2 (解释方差: {pca.explained_variance_ratio_[1]:.2%})')
        plt.legend(title='聚类标签')
        plt.grid(True)
        plt.show()
        print("可视化完成。")

    # ====== 新增：PCA 与高维可视化辅助 ======
    def _ensure_scaled(self) -> np.ndarray:
        """确保标准化数据就绪，并返回 ndarray。"""
        if self.scaled_data is None:
            self._preprocess_data()
        assert self.scaled_data is not None
        return cast(np.ndarray, self.scaled_data)

    def get_pca_loadings(self, n_components: int = 2) -> pd.DataFrame:
        """
        获取指定主成分数量的载荷矩阵（各特征在主成分上的权重）。

        返回：index 为原始特征名，列为 PC1..PCn 的 DataFrame。
        """
        X = self._ensure_scaled()
        n_components = max(1, min(n_components, X.shape[1]))
        pca = PCA(n_components=n_components)
        pca.fit(X)
        loadings = pd.DataFrame(
            pca.components_.T,
            index=self.data.columns,
            columns=[f"PC{i+1}" for i in range(n_components)],
        )
        loadings["loading_norm"] = np.linalg.norm(loadings.values, axis=1)
        return loadings.sort_values("loading_norm", ascending=False)

    def get_principal_components(self, n_components: int = 2) -> pd.DataFrame:
        """返回样本在前 n 个主成分空间中的坐标（PC 分数）。"""
        X = self._ensure_scaled()
        n_components = max(1, min(n_components, X.shape[1]))
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(X)
        pc_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(n_components)])
        if self.labels is not None:
            pc_df["cluster"] = self.labels
        pc_df.attrs["explained_variance_ratio_"] = pca.explained_variance_ratio_
        return pc_df

    def visualize_pca_scree(self, max_components: int = 10) -> None:
        """绘制 PCA 碎石图（方差贡献率与累积贡献率）。"""
        X = self._ensure_scaled()
        n = min(max_components, X.shape[1])
        pca = PCA(n_components=n)
        pca.fit(X)
        var_ratio = pca.explained_variance_ratio_
        cum_ratio = np.cumsum(var_ratio)

        plt.figure(figsize=(10, 5))
        plt.bar(range(1, n + 1), var_ratio, alpha=0.7, label='单个主成分方差贡献率')
        plt.plot(range(1, n + 1), cum_ratio, 'o-', color='red', label='累积方差贡献率')
        plt.xticks(range(1, n + 1))
        plt.xlabel('主成分序号')
        plt.ylabel('方差贡献率')
        plt.title('PCA 碎石图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def visualize_pca_biplot(self, top_features: int = 8, scale_arrows: float = 1.0) -> None:
        """
        绘制 2D PCA biplot：样本在 PC1-PC2 的散点 + 特征载荷箭头。
        自动选取载荷范数最大的前 top_features 个特征。
        """
        if self.kmeans_model is None or self.labels is None:
            print("提示：未进行聚类，biplot 仍会绘制，但无颜色分组。")

        X = self._ensure_scaled()
        pca = PCA(n_components=2)
        scores = pca.fit_transform(X)
        loadings = pd.DataFrame(pca.components_.T, index=self.data.columns, columns=['PC1', 'PC2'])
        loadings['norm'] = np.linalg.norm(loadings[['PC1', 'PC2']].values, axis=1)
        top = loadings.sort_values('norm', ascending=False).head(max(2, top_features))

        plt.figure(figsize=(12, 8))
        if self.labels is not None:
            sns.scatterplot(x=scores[:, 0], y=scores[:, 1], hue=self.labels, palette='hsv', alpha=0.7)
            plt.legend(title='聚类标签')
        else:
            plt.scatter(scores[:, 0], scores[:, 1], alpha=0.7)

        # 以样本分数范围缩放箭头长度
        x_scale = (scores[:, 0].max() - scores[:, 0].min())
        y_scale = (scores[:, 1].max() - scores[:, 1].min())
        for feat, row in top.iterrows():
            plt.arrow(0, 0, row['PC1'] * x_scale * 0.2 * scale_arrows, row['PC2'] * y_scale * 0.2 * scale_arrows,
                      color='black', alpha=0.6, head_width=0.02 * max(x_scale, y_scale))
            plt.text(row['PC1'] * x_scale * 0.22 * scale_arrows, row['PC2'] * y_scale * 0.22 * scale_arrows,
                     str(feat), color='black', fontsize=9)

        plt.xlabel(f'PC1 (方差贡献: {pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 (方差贡献: {pca.explained_variance_ratio_[1]:.1%})')
        plt.title('PCA Biplot（含特征载荷）')
        plt.grid(True, alpha=0.3)
        plt.show()

    def visualize_pairwise_top_features(self, top_k: int = 5) -> List[str]:
        """
        自动选择与前两主成分关联度最高的前 top_k 个特征，绘制两两散点矩阵（Pairplot）。
        返回选中特征名列表。
        """
        if top_k < 2:
            top_k = 2

        loadings = self.get_pca_loadings(n_components=2)
        selected = loadings.head(top_k).index.tolist()

        plot_df = self.data[selected].copy()
        if self.labels is not None:
            plot_df['cluster'] = self.labels
            sns.pairplot(plot_df, hue='cluster', corner=True, diag_kind='hist', plot_kws={'alpha': 0.7})
        else:
            sns.pairplot(plot_df, corner=True, diag_kind='hist', plot_kws={'alpha': 0.7})
        plt.suptitle('自动选择的高权重特征两两散点矩阵', y=1.02)
        plt.show()
        return selected

    def visualize_clusters_3d(self) -> None:
        """使用 3D PCA（前三主成分）绘制三维散点图。"""
        if self.scaled_data is None:
            self._preprocess_data()
        if self.scaled_data is None:
            raise RuntimeError('标准化数据不可用。')
        X = cast(np.ndarray, self.scaled_data)
        n = min(3, X.shape[1])
        if n < 3:
            print('特征少于 3 个，跳过 3D 可视化。')
            return
        pca = PCA(n_components=3)
        scores = pca.fit_transform(X)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        if self.labels is not None:
            for lab in np.unique(self.labels):
                pts = scores[self.labels == lab]
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linestyle='', marker='o', label=f'cluster {lab}', markersize=3, alpha=0.9)
            ax.legend()
        else:
            ax.plot(scores[:, 0], scores[:, 1], scores[:, 2], linestyle='', marker='o', markersize=3, alpha=0.9)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
        ax.set_title('PCA 三维可视化')
        plt.tight_layout()
        plt.show()
