import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="K-Means + PCA Explorer | A.Masmi",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("wine_clustered_data.csv")


@st.cache_data
def load_scores():
    return pd.read_csv("kmeans_scores.csv")


@st.cache_resource
def load_scaler():
    return joblib.load("scaler.joblib")


@st.cache_resource
def load_metadata():
    return joblib.load("kmeans_pca_metadata.joblib")


df = load_data()
scores = load_scores()
scaler = load_scaler()
metadata = load_metadata()

feature_names = metadata["feature_names"]


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 4rem;
    padding-bottom: 4rem;
}

.main-title {
    font-size: 3.2rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}

.subtitle {
    font-size: 1.15rem;
    color: #94a3b8;
    margin-bottom: 2rem;
}

.info-card {
    padding: 18px;
    border: 1px solid #334155;
    border-radius: 12px;
    margin-top: 10px;
    margin-bottom: 20px;
}

.cluster-box {
    padding: 18px;
    border: 1px solid #334155;
    border-radius: 12px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Interactive K-Means + PCA Explorer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Unsupervised Machine Learning Demo • A.Masmi</div>',
    unsafe_allow_html=True
)

st.write(
    """
    Explore how K-Means clustering discovers structure in a real
    multidimensional dataset and how PCA transforms 13 features into
    visual 2D and 3D representations.
    """
)


# ============================================================
# TOP METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Samples",
    f"{metadata['samples']:,}"
)

c2.metric(
    "Features",
    metadata["features"]
)

c3.metric(
    "Best K",
    metadata["best_k"]
)

c4.metric(
    "Best Silhouette",
    f"{metadata['best_silhouette']:.3f}"
)

st.divider()


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Interactive Clustering",
    "PCA 2D",
    "PCA 3D",
    "Cluster Insights",
    "About Project"
])


# ============================================================
# TAB 1
# INTERACTIVE K-MEANS
# ============================================================

with tab1:

    st.header("Interactive K-Means Explorer")

    st.write(
        """
        Change the number of clusters and watch the clustering
        metrics and PCA visualization update automatically.
        """
    )

    selected_k = st.slider(
        "Choose number of clusters (K)",
        min_value=2,
        max_value=10,
        value=3,
        step=1
    )

    X = df[feature_names].copy()

    X_scaled = scaler.transform(X)

    dynamic_model = KMeans(
        n_clusters=selected_k,
        random_state=42,
        n_init=20
    )

    dynamic_labels = dynamic_model.fit_predict(
        X_scaled
    )

    dynamic_silhouette = silhouette_score(
        X_scaled,
        dynamic_labels
    )

    dynamic_inertia = dynamic_model.inertia_

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Selected K",
        selected_k
    )

    m2.metric(
        "Silhouette Score",
        f"{dynamic_silhouette:.4f}"
    )

    m3.metric(
        "Inertia",
        f"{dynamic_inertia:.1f}"
    )

    if selected_k == metadata["best_k"]:

        st.success(
            "This K produced the highest silhouette score "
            "among the tested values."
        )

    else:

        st.info(
            f"The notebook selected K={metadata['best_k']} "
            "as the best value using silhouette score."
        )


    # --------------------------------------------------------
    # ELBOW + SILHOUETTE
    # --------------------------------------------------------

    graph1, graph2 = st.columns(2)

    with graph1:

        elbow = px.line(
            scores,
            x="K",
            y="Inertia",
            markers=True,
            title="Elbow Method"
        )

        elbow.add_vline(
            x=selected_k,
            line_dash="dash"
        )

        elbow.update_xaxes(
            dtick=1
        )

        st.plotly_chart(
            elbow,
            use_container_width=True
        )


    with graph2:

        silhouette_fig = px.line(
            scores,
            x="K",
            y="Silhouette_Score",
            markers=True,
            title="Silhouette Score by K"
        )

        silhouette_fig.add_vline(
            x=selected_k,
            line_dash="dash"
        )

        silhouette_fig.update_xaxes(
            dtick=1
        )

        st.plotly_chart(
            silhouette_fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # DYNAMIC PCA
    # --------------------------------------------------------

    st.subheader(
        f"Live PCA Cluster View — K={selected_k}"
    )

    dynamic_pca = PCA(
        n_components=2
    )

    X_dynamic_pca = dynamic_pca.fit_transform(
        X_scaled
    )

    dynamic_df = pd.DataFrame({
        "PC1": X_dynamic_pca[:, 0],
        "PC2": X_dynamic_pca[:, 1],
        "Cluster": dynamic_labels.astype(str),
        "Original Class": df["original_class_name"]
    })

    dynamic_fig = px.scatter(
        dynamic_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        symbol="Original Class",
        hover_data=[
            "Cluster",
            "Original Class"
        ],
        title=
        f"K-Means Clusters Projected with PCA — K={selected_k}"
    )

    dynamic_fig.update_traces(
        marker=dict(size=11)
    )

    dynamic_fig.update_layout(
        height=650
    )

    st.plotly_chart(
        dynamic_fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # CLUSTER SIZES
    # --------------------------------------------------------

    cluster_sizes = (
        pd.Series(dynamic_labels)
        .value_counts()
        .sort_index()
        .reset_index()
    )

    cluster_sizes.columns = [
        "Cluster",
        "Samples"
    ]

    cluster_sizes["Cluster"] = (
        cluster_sizes["Cluster"]
        .astype(str)
    )

    size_fig = px.bar(
        cluster_sizes,
        x="Cluster",
        y="Samples",
        text="Samples",
        title="Cluster Sizes"
    )

    st.plotly_chart(
        size_fig,
        use_container_width=True
    )


# ============================================================
# TAB 2
# PCA 2D
# ============================================================

with tab2:

    st.header(
        "PCA 2D Visualization"
    )

    st.write(
        """
        PCA reduces the original 13-dimensional feature space
        to two principal components so that the K-Means clusters
        can be visualized.
        """
    )

    X = df[feature_names]
    X_scaled = scaler.transform(X)

    pca2 = PCA(
        n_components=2
    )

    X2 = pca2.fit_transform(
        X_scaled
    )

    kmeans3 = KMeans(
        n_clusters=metadata["best_k"],
        random_state=42,
        n_init=20
    )

    labels3 = kmeans3.fit_predict(
        X_scaled
    )

    pca2_df = pd.DataFrame({
        "PC1": X2[:, 0],
        "PC2": X2[:, 1],
        "Cluster": labels3.astype(str),
        "Original Class":
            df["original_class_name"]
    })

    v1, v2 = st.columns(2)

    v1.metric(
        "PC1 Variance",
        f"{pca2.explained_variance_ratio_[0] * 100:.1f}%"
    )

    v2.metric(
        "PC2 Variance",
        f"{pca2.explained_variance_ratio_[1] * 100:.1f}%"
    )

    fig2d = px.scatter(
        pca2_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        symbol="Original Class",
        hover_data=[
            "Cluster",
            "Original Class"
        ],
        title=
        "Best K-Means Clustering in 2D PCA Space"
    )

    fig2d.update_traces(
        marker=dict(size=12)
    )

    fig2d.update_layout(
        height=700
    )

    st.plotly_chart(
        fig2d,
        use_container_width=True
    )

    st.markdown(
        """
        **How to read this graph**

        Each point represents one wine sample.

        The **color** represents the cluster discovered by K-Means.

        The **symbol** represents the original known wine class.

        K-Means does not use the original class labels during training.
        """
    )


# ============================================================
# TAB 3
# PCA 3D
# ============================================================

with tab3:

    st.header(
        "Interactive 3D PCA Explorer"
    )

    st.write(
        """
        Drag the graph with your mouse to rotate the cluster space.
        Scroll to zoom and hover over individual samples.
        """
    )

    X = df[feature_names]
    X_scaled = scaler.transform(X)

    pca3 = PCA(
        n_components=3
    )

    X3 = pca3.fit_transform(
        X_scaled
    )

    kmeans_3d = KMeans(
        n_clusters=metadata["best_k"],
        random_state=42,
        n_init=20
    )

    labels_3d = kmeans_3d.fit_predict(
        X_scaled
    )

    pca3_df = pd.DataFrame({
        "PC1": X3[:, 0],
        "PC2": X3[:, 1],
        "PC3": X3[:, 2],
        "Cluster": labels_3d.astype(str),
        "Original Class":
            df["original_class_name"]
    })

    a, b, c = st.columns(3)

    a.metric(
        "PC1",
        f"{pca3.explained_variance_ratio_[0] * 100:.1f}% variance"
    )

    b.metric(
        "PC2",
        f"{pca3.explained_variance_ratio_[1] * 100:.1f}% variance"
    )

    c.metric(
        "PC3",
        f"{pca3.explained_variance_ratio_[2] * 100:.1f}% variance"
    )

    fig3d = px.scatter_3d(
        pca3_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Cluster",
        symbol="Original Class",
        hover_data=[
            "Cluster",
            "Original Class"
        ],
        title=
        "Rotatable 3D PCA Cluster Visualization"
    )

    fig3d.update_traces(
        marker=dict(
            size=6,
            opacity=0.85
        )
    )

    fig3d.update_layout(
        height=800
    )

    st.plotly_chart(
        fig3d,
        use_container_width=True
    )

    st.info(
        "Try clicking and dragging directly on the graph "
        "to rotate the clusters in three dimensions."
    )


# ============================================================
# TAB 4
# CLUSTER INSIGHTS
# ============================================================

with tab4:

    st.header(
        "Cluster Insights"
    )

    X = df[feature_names]
    X_scaled = scaler.transform(X)

    final_model = KMeans(
        n_clusters=metadata["best_k"],
        random_state=42,
        n_init=20
    )

    final_labels = final_model.fit_predict(
        X_scaled
    )

    insight_df = df.copy()

    insight_df["Cluster"] = (
        final_labels.astype(str)
    )


    # --------------------------------------------------------
    # CLUSTER DISTRIBUTION
    # --------------------------------------------------------

    counts = (
        insight_df["Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    counts.columns = [
        "Cluster",
        "Samples"
    ]

    fig_counts = px.pie(
        counts,
        names="Cluster",
        values="Samples",
        hole=0.45,
        title="Cluster Distribution"
    )

    st.plotly_chart(
        fig_counts,
        use_container_width=True
    )


    # --------------------------------------------------------
    # ORIGINAL CLASS VS CLUSTER
    # --------------------------------------------------------

    st.subheader(
        "K-Means Clusters vs Original Classes"
    )

    comparison = pd.crosstab(
        insight_df["original_class_name"],
        insight_df["Cluster"]
    )

    st.dataframe(
        comparison,
        use_container_width=True
    )

    heatmap = go.Figure(
        data=go.Heatmap(
            z=comparison.values,
            x=[
                f"Cluster {x}"
                for x in comparison.columns
            ],
            y=comparison.index,
            text=comparison.values,
            texttemplate="%{text}"
        )
    )

    heatmap.update_layout(
        title=
        "Original Wine Classes vs Discovered Clusters",
        height=500
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True
    )


    # --------------------------------------------------------
    # FEATURE PROFILES
    # --------------------------------------------------------

    st.subheader(
        "Cluster Feature Profiles"
    )

    selected_feature = st.selectbox(
        "Choose a feature to compare",
        feature_names
    )

    feature_summary = (
        insight_df
        .groupby("Cluster")[
            selected_feature
        ]
        .mean()
        .reset_index()
    )

    feature_fig = px.bar(
        feature_summary,
        x="Cluster",
        y=selected_feature,
        text_auto=".2f",
        title=
        f"Average {selected_feature} by Cluster"
    )

    st.plotly_chart(
        feature_fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # DATA EXPLORER
    # --------------------------------------------------------

    st.subheader(
        "Explore the Dataset"
    )

    selected_cluster = st.selectbox(
        "Filter by cluster",
        ["All"] +
        sorted(
            insight_df["Cluster"]
            .unique()
            .tolist()
        )
    )

    if selected_cluster == "All":

        filtered_df = insight_df

    else:

        filtered_df = insight_df[
            insight_df["Cluster"]
            == selected_cluster
        ]

    st.write(
        f"Samples shown: {len(filtered_df)}"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 5
# ABOUT
# ============================================================

with tab5:

    st.header(
        "About This Project"
    )

    st.markdown(
        f"""
        ### Project Goal

        Demonstrate an end-to-end unsupervised machine-learning
        workflow using K-Means clustering and Principal Component
        Analysis.

        ### Dataset

        **scikit-learn Wine dataset**

        **Samples:** {metadata['samples']}

        **Numerical features:** {metadata['features']}

        The dataset contains chemical measurements from wine
        samples.

        ### K-Means

        K values from **2 through 10** were evaluated.

        The highest silhouette score was:

        **{metadata['best_silhouette']:.4f} at K={metadata['best_k']}**

        ### PCA

        PCA was used to reduce the original 13-dimensional
        feature space.

        **{metadata['pcs_for_90_percent_variance']} principal components**
        preserve at least **90% of the variance**.

        Two and three principal components are used separately
        for visualization.

        ### Technologies

        **Python**

        **pandas**

        **NumPy**

        **scikit-learn**

        **K-Means**

        **PCA**

        **Plotly**

        **Streamlit**

        **JupyterLab**

        **Git & GitHub**

        ### What This Demo Shows

        - Unsupervised learning
        - Feature standardization
        - K-Means clustering
        - Elbow method
        - Silhouette analysis
        - Dimensionality reduction
        - PCA explained variance
        - Interactive 2D visualization
        - Interactive 3D visualization
        - Cluster interpretation

        ### Portfolio Project

        Developed by **A.Masmi** as an interactive
        Machine Learning and Data Science portfolio project.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#94a3b8;
        padding:10px;
    ">
        A.Masmi • Machine Learning & Data Science Portfolio
    </div>
    """,
    unsafe_allow_html=True
)
