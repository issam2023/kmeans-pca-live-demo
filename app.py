import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Customer Clustering & PCA | A. Masmi",
    page_icon="🧩",
    layout="wide"
)

now = datetime.now()
session_date = now.strftime("%B %d, %Y")
session_time = now.strftime("%I:%M %p")

# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
}

.hero {
    padding: 30px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,.25);
    margin-bottom: 20px;
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
}

.hero-subtitle {
    font-size: 1.1rem;
    opacity: .78;
    margin-top: 8px;
}

.workflow {
    padding: 14px;
    text-align: center;
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 12px;
    margin-bottom: 20px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

header_html = f"""<div class="hero">
<div class="hero-title">🧩 AI Customer Clustering & PCA Analytics Dashboard</div>
<div class="hero-subtitle">
25,000 Customers • K-Means • PCA • Cluster Profiling • Unsupervised Learning
</div>
<br>
<strong>Developed by A. Masmi</strong><br>
jovina&#64;gmx.us<br>
Session: {session_date} • {session_time}
</div>"""

st.markdown(
    header_html,
    unsafe_allow_html=True
)

st.markdown("""
<div class="workflow">
Customer Data → Diagnose → Scale → Find Best K → Cluster → PCA → Profile → Predict
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_customer_data():

    df = pd.read_csv(
        "customer_behavior_25000.csv"
    )

    return df


df = load_customer_data()

feature_columns = [
    col
    for col in df.columns
    if col != "customer_id"
]

X = df[
    feature_columns
].copy()

# ============================================================
# SCALE DATA
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)

# ============================================================
# K EVALUATION
# ============================================================

@st.cache_data
def evaluate_k(data):

    results = []

    for k in range(2, 11):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=15
        )

        labels = model.fit_predict(
            data
        )

        results.append({
            "K":
                k,

            "Inertia":
                model.inertia_,

            "Silhouette":
                silhouette_score(
                    data,
                    labels,
                    sample_size=5000,
                    random_state=42
                ),

            "Davies-Bouldin":
                davies_bouldin_score(
                    data,
                    labels
                ),

            "Calinski-Harabasz":
                calinski_harabasz_score(
                    data,
                    labels
                )
        })

    return pd.DataFrame(
        results
    )


scores_df = evaluate_k(
    X_scaled
)

best_k = int(
    scores_df.loc[
        scores_df["Silhouette"].idxmax(),
        "K"
    ]
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Clustering Controls"
)

selected_k = st.sidebar.slider(
    "Number of clusters (K)",
    min_value=2,
    max_value=10,
    value=best_k
)

st.sidebar.success(
    f"Recommended K = {best_k}"
)

st.sidebar.markdown(
    """
### Why StandardScaler?

K-Means uses Euclidean distance.

Without scaling, features such as annual income could dominate
features such as satisfaction score.
"""
)

# ============================================================
# K-MEANS
# ============================================================

kmeans = KMeans(
    n_clusters=selected_k,
    random_state=42,
    n_init=20
)

labels = kmeans.fit_predict(
    X_scaled
)

# ============================================================
# METRICS
# ============================================================

silhouette = silhouette_score(
    X_scaled,
    labels,
    sample_size=5000,
    random_state=42
)

db_score = davies_bouldin_score(
    X_scaled,
    labels
)

ch_score = calinski_harabasz_score(
    X_scaled,
    labels
)

# ============================================================
# PCA
# ============================================================

pca_full = PCA()

pca_full.fit(
    X_scaled
)

explained_variance = (
    pca_full
    .explained_variance_ratio_
)

cumulative_variance = np.cumsum(
    explained_variance
)

pca = PCA(
    n_components=3
)

pca_values = pca.fit_transform(
    X_scaled
)

pca_df = pd.DataFrame({
    "PC1":
        pca_values[:, 0],

    "PC2":
        pca_values[:, 1],

    "PC3":
        pca_values[:, 2],

    "Cluster":
        labels.astype(str),

    "customer_id":
        df["customer_id"]
})

# ============================================================
# TOP KPIs
# ============================================================

k1, k2, k3, k4, k5, k6 = st.columns(6)

k1.metric(
    "Customers",
    f"{len(df):,}"
)

k2.metric(
    "Features",
    len(feature_columns)
)

k3.metric(
    "Selected K",
    selected_k
)

k4.metric(
    "Silhouette",
    f"{silhouette:.3f}"
)

k5.metric(
    "Davies-Bouldin",
    f"{db_score:.3f}"
)

k6.metric(
    "PCA 3D Variance",
    f"{pca.explained_variance_ratio_.sum():.1%}"
)

st.divider()

# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "🏠 Overview",
    "🩺 Data Quality",
    "🧹 Scaling",
    "🎯 Best K",
    "🧩 Cluster Profiles",
    "📉 PCA Analysis",
    "🌐 2D / 3D Explorer",
    "📊 Evaluation",
    "🔮 New Customer"
])

# ============================================================
# OVERVIEW
# ============================================================

with tabs[0]:

    st.header(
        "Customer Dataset Overview"
    )

    st.write(
        """
This synthetic dataset contains **25,000 customer records**
with behavioral and commercial features designed for
unsupervised customer segmentation.
"""
    )

    o1, o2, o3, o4 = st.columns(4)

    o1.metric(
        "Rows",
        f"{len(df):,}"
    )

    o2.metric(
        "Columns",
        len(df.columns)
    )

    o3.metric(
        "Clustering Features",
        len(feature_columns)
    )

    o4.metric(
        "Missing Values",
        int(
            df
            .isna()
            .sum()
            .sum()
        )
    )

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Descriptive Statistics"
    )

    st.dataframe(
        X.describe().T.round(2),
        use_container_width=True
    )

    feature = st.selectbox(
        "Explore a feature",
        feature_columns
    )

    fig = px.histogram(
        df,
        x=feature,
        nbins=40,
        marginal="box",
        title=f"Distribution of {feature}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ============================================================
# DATA QUALITY
# ============================================================

with tabs[1]:

    st.header(
        "Data Quality"
    )

    quality_df = pd.DataFrame({
        "Feature":
            df.columns,

        "Type":
            df.dtypes.astype(str).values,

        "Missing":
            df.isna().sum().values,

        "Unique":
            df.nunique().values
    })

    st.dataframe(
        quality_df,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "No missing values detected."
    )

    st.info(
        """
`customer_id` is excluded from clustering because it is an
identifier, not a behavioral feature.
"""
    )

# ============================================================
# SCALING
# ============================================================

with tabs[2]:

    st.header(
        "Feature Standardization"
    )

    scaling_df = pd.DataFrame({
        "Feature":
            feature_columns,

        "Original Mean":
            X.mean().values,

        "Original Std":
            X.std().values,

        "Scaled Mean":
            X_scaled.mean(
                axis=0
            ),

        "Scaled Std":
            X_scaled.std(
                axis=0
            )
    })

    st.dataframe(
        scaling_df.round(3),
        use_container_width=True,
        hide_index=True
    )

    selected_feature = st.selectbox(
        "Feature to compare",
        feature_columns,
        key="scaling_feature"
    )

    index = feature_columns.index(
        selected_feature
    )

    compare_df = pd.concat([
        pd.DataFrame({
            "Value":
                X[
                    selected_feature
                ],

            "Version":
                "Original"
        }),

        pd.DataFrame({
            "Value":
                X_scaled[
                    :,
                    index
                ],

            "Version":
                "Standardized"
        })
    ])

    fig_scale = px.histogram(
        compare_df,
        x="Value",
        color="Version",
        barmode="overlay",
        opacity=.55,
        nbins=40,
        title="Original vs Standardized Distribution"
    )

    st.plotly_chart(
        fig_scale,
        use_container_width=True
    )

# ============================================================
# BEST K
# ============================================================

with tabs[3]:

    st.header(
        "Automatic K Selection"
    )

    st.success(
        f"🏆 Recommended number of clusters: K = {best_k}"
    )

    st.dataframe(
        scores_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    fig_sil = px.line(
        scores_df,
        x="K",
        y="Silhouette",
        markers=True,
        title="Silhouette Score by K"
    )

    fig_sil.add_vline(
        x=best_k,
        line_dash="dash",
        annotation_text=f"Best K = {best_k}"
    )

    st.plotly_chart(
        fig_sil,
        use_container_width=True
    )

    fig_elbow = px.line(
        scores_df,
        x="K",
        y="Inertia",
        markers=True,
        title="Elbow Method"
    )

    st.plotly_chart(
        fig_elbow,
        use_container_width=True
    )

# ============================================================
# CLUSTER PROFILES
# ============================================================

with tabs[4]:

    st.header(
        "Customer Cluster Profiles"
    )

    clustered = X.copy()

    clustered[
        "Cluster"
    ] = labels

    cluster_counts = (
        clustered[
            "Cluster"
        ]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    cluster_counts.columns = [
        "Cluster",
        "Customers"
    ]

    fig_counts = px.bar(
        cluster_counts,
        x="Cluster",
        y="Customers",
        title="Customers per Cluster"
    )

    st.plotly_chart(
        fig_counts,
        use_container_width=True
    )

    profiles = (
        clustered
        .groupby(
            "Cluster"
        )
        .mean()
        .round(2)
    )

    st.subheader(
        "Mean Customer Profile"
    )

    st.dataframe(
        profiles,
        use_container_width=True
    )

    profile_feature = st.selectbox(
        "Compare clusters by feature",
        feature_columns,
        key="profile_feature"
    )

    profile_plot = (
        clustered
        .groupby(
            "Cluster"
        )[
            profile_feature
        ]
        .mean()
        .reset_index()
    )

    fig_profile = px.bar(
        profile_plot,
        x="Cluster",
        y=profile_feature,
        title=f"Cluster Comparison — {profile_feature}"
    )

    st.plotly_chart(
        fig_profile,
        use_container_width=True
    )

# ============================================================
# PCA
# ============================================================

with tabs[5]:

    st.header(
        "Principal Component Analysis"
    )

    variance_df = pd.DataFrame({
        "Component": [
            f"PC{i+1}"
            for i in range(
                len(
                    explained_variance
                )
            )
        ],

        "Explained Variance":
            explained_variance,

        "Cumulative Variance":
            cumulative_variance
    })

    p1, p2, p3 = st.columns(3)

    p1.metric(
        "PC1 Variance",
        f"{explained_variance[0]:.1%}"
    )

    p2.metric(
        "PC1 + PC2",
        f"{cumulative_variance[1]:.1%}"
    )

    p3.metric(
        "PC1 + PC2 + PC3",
        f"{cumulative_variance[2]:.1%}"
    )

    fig_var = px.bar(
        variance_df,
        x="Component",
        y="Explained Variance",
        title="Explained Variance by Principal Component"
    )

    st.plotly_chart(
        fig_var,
        use_container_width=True
    )

    fig_cumulative = px.line(
        variance_df,
        x="Component",
        y="Cumulative Variance",
        markers=True,
        title="Cumulative Explained Variance"
    )

    fig_cumulative.add_hline(
        y=.90,
        line_dash="dash",
        annotation_text="90%"
    )

    st.plotly_chart(
        fig_cumulative,
        use_container_width=True
    )

    components_90 = int(
        np.argmax(
            cumulative_variance
            >= .90
        )
        + 1
    )

    st.success(
        f"{components_90} principal components explain at least 90% of the variance."
    )

# ============================================================
# 2D / 3D
# ============================================================

with tabs[6]:

    st.header(
        "Interactive PCA Cluster Explorer"
    )

    view = st.radio(
        "View",
        [
            "2D PCA",
            "3D PCA"
        ],
        horizontal=True
    )

    sample_size = st.slider(
        "Points to display",
        min_value=1000,
        max_value=10000,
        value=5000,
        step=1000
    )

    plot_df = pca_df.sample(
        sample_size,
        random_state=42
    )

    if view == "2D PCA":

        fig_2d = px.scatter(
            plot_df,
            x="PC1",
            y="PC2",
            color="Cluster",
            hover_data=[
                "customer_id"
            ],
            title="Customer Segments — PCA 2D"
        )

        st.plotly_chart(
            fig_2d,
            use_container_width=True
        )

    else:

        fig_3d = px.scatter_3d(
            plot_df,
            x="PC1",
            y="PC2",
            z="PC3",
            color="Cluster",
            hover_data=[
                "customer_id"
            ],
            title="Customer Segments — PCA 3D"
        )

        st.plotly_chart(
            fig_3d,
            use_container_width=True
        )

# ============================================================
# EVALUATION
# ============================================================

with tabs[7]:

    st.header(
        "Clustering Evaluation"
    )

    e1, e2, e3 = st.columns(3)

    e1.metric(
        "Silhouette Score",
        f"{silhouette:.4f}"
    )

    e2.metric(
        "Davies-Bouldin",
        f"{db_score:.4f}"
    )

    e3.metric(
        "Calinski-Harabasz",
        f"{ch_score:.1f}"
    )

    st.markdown("""
### Metric interpretation

**Silhouette Score**  
Higher is better. Measures cluster cohesion and separation.

**Davies-Bouldin Index**  
Lower is better.

**Calinski-Harabasz Score**  
Higher generally indicates better-defined clusters.
""")

# ============================================================
# NEW CUSTOMER
# ============================================================

with tabs[8]:

    st.header(
        "New Customer Cluster Prediction"
    )

    st.write(
        """
Enter behavioral information for a new customer.
The same StandardScaler is applied before K-Means assigns
the customer to the nearest cluster.
"""
    )

    inputs = {}

    columns = st.columns(
        3
    )

    for i, feature_name in enumerate(
        feature_columns
    ):

        with columns[
            i % 3
        ]:

            inputs[
                feature_name
            ] = st.number_input(
                feature_name
                .replace(
                    "_",
                    " "
                )
                .title(),

                value=float(
                    X[
                        feature_name
                    ]
                    .median()
                ),

                format="%.2f"
            )

    new_customer = pd.DataFrame(
        [inputs]
    )

    new_scaled = scaler.transform(
        new_customer
    )

    cluster_prediction = int(
        kmeans.predict(
            new_scaled
        )[0]
    )

    distances = kmeans.transform(
        new_scaled
    )[0]

    n1, n2 = st.columns(
        2
    )

    n1.metric(
        "Assigned Cluster",
        cluster_prediction
    )

    n2.metric(
        "Distance to Centroid",
        f"{distances[cluster_prediction]:.3f}"
    )

    distance_df = pd.DataFrame({
        "Cluster":
            [
                str(i)
                for i in range(
                    selected_k
                )
            ],

        "Distance":
            distances
    })

    fig_distance = px.bar(
        distance_df,
        x="Cluster",
        y="Distance",
        title="Distance to Cluster Centroids"
    )

    st.plotly_chart(
        fig_distance,
        use_container_width=True
    )

# ============================================================
# RECOMMENDATION
# ============================================================

st.divider()

recommended_row = scores_df.loc[
    scores_df[
        "Silhouette"
    ].idxmax()
]

st.subheader(
    "🤖 Clustering Recommendation"
)

st.success(
    f"""
Recommended configuration: **K = {best_k}**

K={best_k} achieved the highest Silhouette Score
(**{recommended_row['Silhouette']:.3f}**) among K=2 to K=10.

The dashboard still allows users to change K manually and
compare the clustering results.
"""
)

# ============================================================
# FOOTER
# ============================================================

st.divider()

footer_html = f"""<div style="text-align:center; padding:20px; opacity:.75; line-height:1.7;">
<strong>AI Customer Clustering & PCA Analytics Dashboard</strong><br>
Developed by A. Masmi • jovina&#64;gmx.us<br>
25,000 Customers • K-Means • PCA • StandardScaler • Plotly • Streamlit<br>
Session: {session_date} • {session_time}<br>
<a href="https://issam2023.github.io/masmi-portfolio/#projects"
target="_blank">AI & Data Science Portfolio</a>
</div>"""

st.markdown(
    footer_html,
    unsafe_allow_html=True
)
