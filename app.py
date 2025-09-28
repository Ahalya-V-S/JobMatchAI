import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the LinkedIn jobs dataset"""
    loader = DataLoader()
    data = loader.load_data()
    
    if data is not None:
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess(data)
        return processed_data
    return None

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
        st.metric("Job Locations", f"{data['location'].nunique():,}")
    
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
    st.markdown('<div class="section-header">📊 Data Analytics Dashboard</div>', unsafe_allow_html=True)
    
    analytics = Analytics(data)
    
    # Job distribution by experience level
    st.markdown("#### Job Distribution by Experience Level")
    exp_dist = analytics.get_experience_distribution()
    fig_exp = px.pie(values=exp_dist.values, names=exp_dist.index, 
                     title="Jobs by Experience Level", color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig_exp, use_container_width=True)
    
    # Top companies and locations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top 15 Companies by Job Postings")
        top_companies = analytics.get_top_companies(15)
        fig_companies = px.bar(x=top_companies.values, y=top_companies.index, 
                              orientation='h', title="Top Companies")
        fig_companies.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_companies, use_container_width=True)
    
    with col2:
        st.markdown("#### Top 15 Job Locations")
        top_locations = analytics.get_top_locations(15)
        fig_locations = px.bar(x=top_locations.values, y=top_locations.index, 
                              orientation='h', title="Top Locations")
        fig_locations.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_locations, use_container_width=True)
    
    # Salary analysis
    st.markdown("#### Salary Distribution Analysis")
    salary_stats = analytics.get_salary_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Salary", f"${salary_stats['mean']:,.0f}")
    with col2:
        st.metric("Median Salary", f"${salary_stats['median']:,.0f}")
    with col3:
        st.metric("Min Salary", f"${salary_stats['min']:,.0f}")
    with col4:
        st.metric("Max Salary", f"${salary_stats['max']:,.0f}")
    
    # Salary distribution plot
    salary_data = data.dropna(subset=['salary_min', 'salary_max'])
    if not salary_data.empty:
        fig_salary = px.histogram(salary_data, x='salary_min', nbins=50, 
                                title="Salary Distribution (Minimum Salary)")
        st.plotly_chart(fig_salary, use_container_width=True)
    
    # Skills analysis
    st.markdown("#### Most In-Demand Skills")
    top_skills = analytics.get_top_skills(20)
    if not top_skills.empty:
        fig_skills = px.bar(x=top_skills.values, y=top_skills.index, 
                           orientation='h', title="Top 20 Skills in Job Market")
        fig_skills.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_skills, use_container_width=True)

def show_recommendations_page(data, hybrid):
    st.markdown('<div class="section-header">🎯 Get Personalized Job Recommendations</div>', unsafe_allow_html=True)
    
    # User input form
    st.markdown("#### Tell us about yourself and your preferences:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Basic preferences
        st.markdown("**Basic Information**")
        experience_level = st.selectbox("Experience Level", 
                                      ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive"])
        
        preferred_location = st.selectbox("Preferred Location", 
                                        ["Any"] + sorted(data['location'].dropna().unique().tolist()))
        
        job_type = st.selectbox("Job Type Preference", 
                              ["Any", "Full-time", "Part-time", "Contract", "Temporary", "Internship"])
        
        # Salary preferences
        st.markdown("**Salary Expectations**")
        min_salary = st.number_input("Minimum Salary ($)", min_value=0, value=50000, step=5000)
        max_salary = st.number_input("Maximum Salary ($)", min_value=min_salary, value=150000, step=5000)
    
    with col2:
        # Skills and interests
        st.markdown("**Skills and Interests**")
        
        # Get available skills from data
        all_skills = []
        if 'skills' in data.columns:
            for skills_str in data['skills'].dropna():
                if isinstance(skills_str, str):
                    all_skills.extend([skill.strip() for skill in skills_str.split(',')])
        
        unique_skills = sorted(list(set(all_skills)))[:100]  # Limit to top 100 skills
        
        selected_skills = st.multiselect("Select your key skills (max 10)", 
                                       unique_skills, max_selections=10)
        
        industry_preference = st.text_input("Industry Preference (optional)", 
                                          placeholder="e.g., Technology, Healthcare, Finance")
        
        company_size_pref = st.selectbox("Company Size Preference", 
                                       ["Any", "Startup (1-50)", "Small (51-200)", 
                                        "Medium (201-1000)", "Large (1000+)"])
    
    # Recommendation weights
    st.markdown("#### Customize Recommendation Approach")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cb_weight = st.slider("Content-Based Weight", 0.0, 1.0, 0.4, 0.1,
                             help="Focus on job content similarity")
    with col2:
        cf_weight = st.slider("Collaborative Weight", 0.0, 1.0, 0.3, 0.1,
                             help="Focus on what similar users liked")
    with col3:
        kb_weight = st.slider("Knowledge-Based Weight", 0.0, 1.0, 0.3, 0.1,
                             help="Focus on matching your constraints")
    
    # Normalize weights
    total_weight = cb_weight + cf_weight + kb_weight
    if total_weight > 0:
        cb_weight /= total_weight
        cf_weight /= total_weight
        kb_weight /= total_weight
    
    # Generate recommendations
    if st.button("🔍 Get My Job Recommendations", type="primary"):
        with st.spinner("Generating personalized recommendations..."):
            
            # Create user profile
            user_profile = {
                'experience_level': experience_level,
                'location': preferred_location if preferred_location != "Any" else None,
                'job_type': job_type if job_type != "Any" else None,
                'min_salary': min_salary,
                'max_salary': max_salary,
                'skills': selected_skills,
                'industry': industry_preference if industry_preference else None,
                'company_size': company_size_pref if company_size_pref != "Any" else None
            }
            
            # Get recommendations
            recommendations = hybrid.recommend(
                user_profile, 
                n_recommendations=10,
                weights={'content': cb_weight, 'collaborative': cf_weight, 'knowledge': kb_weight}
            )
            
            if recommendations is not None and len(recommendations) > 0:
                st.success(f"Found {len(recommendations)} job recommendations for you!")
                
                # Display recommendations
                for idx, (job_idx, score) in enumerate(recommendations.iterrows()):
                    job = data.iloc[job_idx]
                    
                    with st.expander(f"🎯 {idx+1}. {job['title']} at {job['company']} (Match: {score:.1%})"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**📍 Location:** {job['location']}")
                            if pd.notna(job.get('experience_level')):
                                st.markdown(f"**📊 Experience:** {job['experience_level']}")
                            if pd.notna(job.get('employment_type')):
                                st.markdown(f"**💼 Type:** {job['employment_type']}")
                            
                            if pd.notna(job.get('description')):
                                description = str(job['description'])[:300] + "..." if len(str(job['description'])) > 300 else str(job['description'])
                                st.markdown(f"**📝 Description:** {description}")
                            
                            if pd.notna(job.get('skills')):
                                st.markdown(f"**🛠️ Required Skills:** {job['skills']}")
                        
                        with col2:
                            if pd.notna(job.get('salary_min')) and pd.notna(job.get('salary_max')):
                                st.markdown(f"**💰 Salary Range:**")
                                st.markdown(f"${job['salary_min']:,.0f} - ${job['salary_max']:,.0f}")
                            
                            if pd.notna(job.get('application_url')):
                                st.markdown(f"**🔗 Apply:** [View Job]({job['application_url']})")
            else:
                st.warning("No jobs found matching your criteria. Try adjusting your preferences.")

def show_evaluation_page(data, cb_filter, cf_filter, kb_filter, hybrid):
    st.markdown('<div class="section-header">📈 System Evaluation Dashboard</div>', unsafe_allow_html=True)
    
    evaluator = Evaluator(data)
    
    # Generate evaluation metrics
    with st.spinner("Evaluating recommendation systems..."):
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

def show_dataset_overview(data):
    st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)
    
    # Dataset basic info
    st.markdown("#### Dataset Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    with col2:
        st.metric("Features", f"{len(data.columns)}")
    with col3:
        st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    with col4:
        st.metric("Missing Values", f"{data.isnull().sum().sum():,}")
    
    # Column information
    st.markdown("#### Column Information")
    col_info = []
    for col in data.columns:
        col_info.append({
            'Column': col,
            'Data Type': str(data[col].dtype),
            'Non-Null Count': data[col].count(),
            'Null Count': data[col].isnull().sum(),
            'Unique Values': data[col].nunique(),
            'Sample Values': str(data[col].dropna().iloc[0])[:50] + "..." if len(str(data[col].dropna().iloc[0])) > 50 else str(data[col].dropna().iloc[0]) if not data[col].dropna().empty else "N/A"
        })
    
    col_df = pd.DataFrame(col_info)
    st.dataframe(col_df, use_container_width=True)
    
    # Missing values heatmap
    st.markdown("#### Missing Values Analysis")
    missing_data = data.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if not missing_data.empty:
        fig_missing = px.bar(x=missing_data.values, y=missing_data.index, 
                           orientation='h', title="Missing Values by Column")
        fig_missing.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.success("No missing values found in the dataset!")
    
    # Sample data
    st.markdown("#### Sample Data (First 10 rows)")
    st.dataframe(data.head(10), use_container_width=True)

if __name__ == "__main__":
    main()
