import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from preprocessor import DataPreprocessor
from content_based_filter import ContentBasedFilter
from collaborative_filter import CollaborativeFilter
from knowledge_based_filter import KnowledgeBasedFilter
from hybrid_recommender import HybridRecommender
from analytics import Analytics
from evaluation import Evaluator

# Page configuration
st.set_page_config(
    page_title="LinkedIn Job Recommender System",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for LinkedIn-like styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #0077B5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2F2F2F;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #F3F2F0;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(hash_funcs={pd.DataFrame: lambda _: None})
def load_and_preprocess_data():
    """Load and preprocess the LinkedIn jobs dataset"""
    loader = DataLoader()
    data = loader.load_data()
    print(data.columns)  # Show all column names
    print(data.head())   # Preview first few rows
    print("*"*100)

    
    if data is not None:
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess(data)
        return processed_data
    return None
@st.cache_data(hash_funcs={pd.DataFrame: lambda _: None})
def initialize_recommenders(data):
    """Initialize all recommendation systems"""
    cb_filter = ContentBasedFilter()
    cb_filter.fit(data)
    
    cf_filter = CollaborativeFilter()
    cf_filter.fit(data)
    
    kb_filter = KnowledgeBasedFilter()
    kb_filter.fit(data)
    
    hybrid = HybridRecommender(cb_filter, cf_filter, kb_filter)
    
    return cb_filter, cf_filter, kb_filter, hybrid

def main():
    st.markdown('<div class="main-header">💼 LinkedIn Job Recommender System</div>', unsafe_allow_html=True)
    st.markdown("### A Hybrid Recommendation System combining Collaborative Filtering, Content-Based Filtering, and Knowledge-Based Approaches")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🏠 Home",
        "📊 Data Analytics",
        "🎯 Get Recommendations", 
        "📈 System Evaluation",
        "📋 Dataset Overview"
    ])
    
    # Load data
    with st.spinner("Loading and preprocessing data..."):
        data = load_and_preprocess_data()
    
    if data is None:
        st.error("Failed to load dataset. Please check your internet connection and try again.")
        st.info("The system will attempt to download the LinkedIn jobs dataset from Kaggle using kagglehub.")
        return
    
    # Initialize recommenders
    with st.spinner("Initializing recommendation systems..."):
        cb_filter, cf_filter, kb_filter, hybrid = initialize_recommenders(data)
    
    if page == "🏠 Home":
        show_home_page(data)
    elif page == "📊 Data Analytics":
        show_analytics_page(data)
    elif page == "🎯 Get Recommendations":
        show_recommendations_page(data, hybrid)
    elif page == "📈 System Evaluation":
        show_evaluation_page(data, cb_filter, cf_filter, kb_filter, hybrid)
    elif page == "📋 Dataset Overview":
        show_dataset_overview(data)

def show_home_page(data):
    st.markdown('<div class="section-header">🏠 Welcome to the LinkedIn Job Recommender System</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Jobs", f"{len(data):,}")
    with col2:
        st.metric("Unique Companies", f"{data['company'].nunique():,}")
    with col3:
        st.metric("Job Locations", f"{data['job_location'].nunique():,}")
    
    st.markdown("---")
    
    # System Architecture
    st.markdown('<div class="section-header">🏗️ System Architecture</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Content-Based Filtering**")
        st.markdown("""
        - Analyzes job descriptions and required skills
        - Uses TF-IDF vectorization for text analysis
        - Recommends jobs similar to user's profile and preferences
        - Considers job titles, descriptions, and skill requirements
        """)
        
        st.markdown("**🤝 Collaborative Filtering**")
        st.markdown("""
        - Uses matrix factorization (SVD) on user-job interactions
        - Finds patterns in user behavior and preferences
        - Recommends jobs liked by similar users
        - Handles sparse data with advanced algorithms
        """)
    
    with col2:
        st.markdown("**🧠 Knowledge-Based Filtering**")
        st.markdown("""
        - Applies explicit user constraints and preferences
        - Filters by location, salary range, experience level
        - Uses domain knowledge about job requirements
        - Provides explainable recommendations
        """)
        
        st.markdown("**🔄 Hybrid Approach**")
        st.markdown("""
        - Combines all three filtering approaches
        - Uses weighted averaging for final recommendations
        - Balances different recommendation strategies
        - Provides diverse and accurate job suggestions
        """)

def show_analytics_page(data):
    st.markdown('<div class="section-header">📊 Job Analytics Dashboard</div>', unsafe_allow_html=True)

    analytics = Analytics(data)

    # --- Top Companies & Locations ---
    col1, col2 = st.columns(2)
    with col1:
        fig_companies = analytics.create_top_companies_plot(15)
        if fig_companies:
            st.plotly_chart(fig_companies, use_container_width=True)

    with col2:
        fig_locations = analytics.create_top_locations_plot(15)
        if fig_locations:
            st.plotly_chart(fig_locations, use_container_width=True)

    # --- Job Level & Job Type Distribution ---
    col1, col2 = st.columns(2)
    with col1:
        fig_level = analytics.create_experience_level_plot()
        if fig_level:
            st.plotly_chart(fig_level, use_container_width=True)

    with col2:
        fig_type = analytics.create_job_type_plot()
        if fig_type:
            st.plotly_chart(fig_type, use_container_width=True)

    # # --- Top Skills ---
    # st.markdown("#### Top 20 Skills in Job Market")
    # fig_skills = analytics.create_top_skills_plot(20)
    # if fig_skills:
    #     st.plotly_chart(fig_skills, use_container_width=True)

    # # --- Job Posting Trends ---
    # st.markdown("#### Job Posting Trends Over Time")
    # fig_trends = analytics.create_job_posting_trends_plot()
    # if fig_trends:
    #     st.plotly_chart(fig_trends, use_container_width=True)

    # # --- Remote Jobs ---
    # st.markdown("#### Remote Job Statistics")
    # remote_stats = analytics.get_remote_job_statistics()
    # st.write(remote_stats)


def show_recommendations_page(data, hybrid):
    st.markdown('<div class="section-header">🎯 Get Personalized Job Recommendations</div>', unsafe_allow_html=True)
    
    # --- User Input Form ---
    st.markdown("#### Tell us about yourself and your preferences:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Basic Information**")
        preferred_location = st.selectbox(
            "Preferred Location", 
            ["Any"] + sorted(data['job_location'].dropna().unique().tolist())
        )
        
        job_type = st.selectbox(
            "Job Type Preference", 
            ["Any"] + sorted(data['job_type'].dropna().unique().tolist())
        )
        
        job_level = st.selectbox(
            "Job Level Preference",
            ["Any"] + sorted(data['job_level'].dropna().unique().tolist())
        )

        # Skills input
        user_skills = st.text_input(
            "Your Skills (comma-separated)",
            placeholder="Python, SQL, Machine Learning"
        )

    st.markdown("#### Customize Recommendation Approach")
    col1, col2, col3 = st.columns(3)
    with col1:
        cb_weight = st.slider("Content-Based Weight", 0.0, 1.0, 0.4, 0.1)
    with col2:
        cf_weight = st.slider("Collaborative Weight", 0.0, 1.0, 0.3, 0.1)
    with col3:
        kb_weight = st.slider("Knowledge-Based Weight", 0.0, 1.0, 0.3, 0.1)

    # Normalize weights
    total_weight = cb_weight + cf_weight + kb_weight
    if total_weight > 0:
        cb_weight /= total_weight
        cf_weight /= total_weight
        kb_weight /= total_weight

    if st.button("🔍 Get My Job Recommendations", type="primary"):
        with st.spinner("Generating personalized recommendations..."):

            # Build user profile
            user_profile = {
                'job_location': preferred_location if preferred_location != "Any" else None,
                'job_type': job_type if job_type != "Any" else None,
                'job_level': job_level if job_level != "Any" else None,
                'skills': [s.strip().title() for s in user_skills.split(',')] if user_skills else None
            }

            # Filter dataset to required columns
            data_filtered = data.dropna(subset=['job_location', 'job_type', 'job_level']).reset_index(drop=True)

            # Get recommendations from hybrid recommender
            recommendations = hybrid.recommend(
                user_profile,
                n_recommendations=10,
                weights={'content': cb_weight, 'collaborative': cf_weight, 'knowledge': kb_weight}
            )

            if recommendations is not None and not recommendations.empty:
                st.success(f"Found {len(recommendations)} job recommendations for you!")

                for idx, row in recommendations.iterrows():
                    # Use the actual 'job_idx' column and cast to int for iloc
                    job_idx = int(row['job_idx'])
                    job = data_filtered.iloc[job_idx]

                    score_value = float(row.get('hybrid_score', 0))

                    with st.expander(f"🎯 {idx+1}. {job['job_title']} at {job['company']} (Match: {score_value:.1%})"):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown(f"**📍 Location:** {job['job_location']}")
                            st.markdown(f"**📊 Level:** {job['job_level']}")
                            st.markdown(f"**💼 Type:** {job['job_type']}")
                            if 'job_skills' in job and pd.notna(job['job_skills']):
                                st.markdown(f"**🛠 Skills Required:** {job['job_skills']}")
                        with col2:
                            if pd.notna(job.get('job_link')):
                                st.markdown(f"**🔗 Apply:** [View Job]({job['job_link']})")
            else:
                st.warning("No jobs found matching your criteria. Try adjusting your preferences.")



def show_evaluation_page(data, cb_filter, cf_filter, kb_filter, hybrid):
    st.markdown('<div class="section-header">📈 System Evaluation Dashboard</div>', unsafe_allow_html=True)
    
    
    
    # Generate evaluation metrics
    with st.spinner("Evaluating recommendation systems..."):
        evaluator = Evaluator(data)
        evaluation_results = evaluator.evaluate_all_systems(cb_filter, cf_filter, kb_filter, hybrid)
    
    # Display evaluation metrics
    st.markdown("#### Recommendation System Performance Comparison")
    
    # Create metrics comparison
    systems = list(evaluation_results.keys())
    metrics = ['precision', 'recall', 'f1_score', 'diversity', 'coverage']
    
    comparison_data = []
    for system in systems:
        for metric in metrics:
            if metric in evaluation_results[system]:
                comparison_data.append({
                    'System': system,
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': evaluation_results[system][metric]
                })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create comparison chart
        fig = px.bar(comparison_df, x='System', y='Value', color='Metric', 
                    title="Performance Metrics Comparison", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed metrics table
        st.markdown("#### Detailed Performance Metrics")
        metrics_table = comparison_df.pivot(index='System', columns='Metric', values='Value')
        st.dataframe(metrics_table.round(4))
    
    # System strengths and weaknesses
    st.markdown("#### System Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Content-Based Filtering**")
        st.markdown("✅ Strengths:")
        st.markdown("- No cold start problem for new users")
        st.markdown("- Transparent recommendations")
        st.markdown("- Good for niche preferences")
        st.markdown("❌ Weaknesses:")
        st.markdown("- Limited diversity")
        st.markdown("- Requires good content features")
        
        st.markdown("**🧠 Knowledge-Based Filtering**")
        st.markdown("✅ Strengths:")
        st.markdown("- Handles constraints well")
        st.markdown("- Explainable recommendations")
        st.markdown("- No data dependency")
        st.markdown("❌ Weaknesses:")
        st.markdown("- Requires domain expertise")
        st.markdown("- May be too rigid")
    
    with col2:
        st.markdown("**🤝 Collaborative Filtering**")
        st.markdown("✅ Strengths:")
        st.markdown("- Discovers hidden patterns")
        st.markdown("- Good for popular items")
        st.markdown("- Learns from user behavior")
        st.markdown("❌ Weaknesses:")
        st.markdown("- Cold start problem")
        st.markdown("- Requires sufficient data")
        
        st.markdown("**🔄 Hybrid System**")
        st.markdown("✅ Strengths:")
        st.markdown("- Combines all advantages")
        st.markdown("- Balances different approaches")
        st.markdown("- Robust performance")
        st.markdown("❌ Weaknesses:")
        st.markdown("- More complex to tune")
        st.markdown("- Higher computational cost")


def show_dataset_overview(data, sample_size=10):
    with st.spinner("Loading dataset overview..."):
        st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)
        
        # ---------------- Basic info ----------------
        st.markdown("#### Dataset Information")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Records", f"{len(data):,}")
        with col2: st.metric("Features", f"{len(data.columns)}")
        with col3: st.metric("Memory Usage", f"{data.memory_usage(deep=False).sum() / 1024**2:.1f} MB")
        with col4: st.metric("Missing Values", f"{data.isnull().sum().sum():,}")
        
        # ---------------- Column info ----------------
        st.markdown("#### Column Information")
        col_info = []
        for col in data.columns:
            sample = data[col].iloc[0] if not data[col].empty else np.nan
            # Handle unhashable / mixed-type columns safely
            try:
                unique_vals = data[col].nunique(dropna=True)
            except Exception:
                unique_vals = np.nan
            col_info.append({
                'Column': col,
                'Data Type': str(data[col].dtype),
                'Non-Null Count': data[col].count(),
                'Null Count': data[col].isnull().sum(),
                'Unique Values': unique_vals,
                'Sample Value': str(sample)[:50] + ("..." if len(str(sample)) > 50 else "")
            })
        st.dataframe(pd.DataFrame(col_info), width='stretch')
        
        # ---------------- Missing values plot ----------------
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if not missing_data.empty:
            fig = px.bar(
                x=missing_data.values, 
                y=missing_data.index, 
                orientation='h',
                title="Missing Values by Column"
            )
            # Only update layout, no deprecated keyword args
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig)  # no extra config
        else:
            st.success("No missing values found!")
        
        # ---------------- Sample rows ----------------
        st.markdown(f"#### Sample Data (First {sample_size} rows)")
        st.dataframe(data.head(sample_size), width='stretch')

if __name__ == "__main__":
    main()
