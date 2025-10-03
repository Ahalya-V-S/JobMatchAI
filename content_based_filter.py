import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import streamlit as st

class ContentBasedFilter:
    """Content-based filtering using TF-IDF and job features"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.feature_matrix = None
        self.scaler = StandardScaler()
        self.job_indices = None
        self.data = None
    
    def fit(self, data):
        """Train the content-based filter"""
        st.info("Training Content-Based Filter...")
        self.data = data.copy()
        self.job_indices = data.index
        
        # 1. Process text features using TF-IDF
        self._create_tfidf_features()
        
        # 2. Process numerical/categorical features
        self._create_feature_matrix()
        
        st.success("Content-Based Filter training completed!")

    def _create_tfidf_features(self):
        """Create TF-IDF features from job_title, job_summary, job_location, job_type, job_level, job_skills"""
        if self.data is None:
            st.error("No data available for TF-IDF feature creation")
            return

        text_features = []

        for idx in self.data.index:
            job = self.data.loc[idx]
            text_parts = []

            if pd.notna(job.get('job_title')):
                text_parts.append(str(job['job_title']))
            if pd.notna(job.get('job_summary')):
                text_parts.append(str(job['job_summary']))
            if pd.notna(job.get('job_location')):
                text_parts.append(str(job['job_location']))
            if pd.notna(job.get('job_type')):
                text_parts.append(str(job['job_type']))
            if pd.notna(job.get('job_level')):
                text_parts.append(str(job['job_level']))
            if pd.notna(job.get('job_skills')):
                text_parts.append(str(job['job_skills']))

            combined_text = ' '.join(text_parts)
            text_features.append(combined_text if combined_text.strip() else " ")

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_features)
        st.info(f"Created TF-IDF matrix with shape: {self.tfidf_matrix.shape}")

    def _create_feature_matrix(self):
        """Create numerical/categorical feature matrix"""
        if self.data is None:
            st.error("No data available for feature matrix creation")
            return

        features = []

        for idx in self.data.index:
            job = self.data.loc[idx]
            job_features = []

            # Placeholder for salary (set 0 if not available)
            salary_min = job.get('salary_min', 0) if pd.notna(job.get('salary_min')) else 0
            salary_max = job.get('salary_max', 0) if pd.notna(job.get('salary_max')) else 0
            salary_avg = (salary_min + salary_max)/2 if salary_max > 0 else 0
            job_features.extend([salary_min, salary_max, salary_avg])

            # Job type one-hot: Hybrid, Remote, Onsite
            emp_type = job.get('job_type', 'Unknown')
            emp_types = ['Hybrid', 'Remote', 'Onsite']
            for et in emp_types:
                job_features.append(1 if emp_type == et else 0)

            # Job level: Associate=1, Mid senior=2
            job_level = job.get('job_level', 'Unknown')
            level_mapping = {'Associate': 1, 'Mid senior': 2, 'Unknown': 0}
            job_features.append(level_mapping.get(job_level, 0))

            # Remote flag
            is_remote = 1 if job.get('is_remote', False) else 0
            job_features.append(is_remote)

            # Number of skills
            skills_count = len(str(job.get('job_skills', '')).split(',')) if pd.notna(job.get('job_skills')) else 0
            job_features.append(skills_count)

            features.append(job_features)

        self.feature_matrix = self.scaler.fit_transform(features)
        st.info(f"Created feature matrix with shape: {self.feature_matrix.shape}")

    def recommend(self, user_profile, n_recommendations=10):
        """Generate content-based recommendations for a user profile"""
        if self.tfidf_matrix is None or self.feature_matrix is None:
            return None

        user_text = self._create_user_text_profile(user_profile)
        user_vector = self.tfidf_vectorizer.transform([user_text])

        text_similarities = cosine_similarity(user_vector, self.tfidf_matrix)[0]
        user_features = self._create_user_feature_profile(user_profile)
        user_feature_vector = self.scaler.transform([user_features])
        feature_similarities = cosine_similarity(user_feature_vector, self.feature_matrix)[0]

        combined_similarities = 0.7*text_similarities + 0.3*feature_similarities
        valid_jobs = self._apply_user_constraints(user_profile)

        filtered_similarities = [(idx, sim) for idx, sim in enumerate(combined_similarities) if idx in valid_jobs]
        filtered_similarities.sort(key=lambda x: x[1], reverse=True)
        top_jobs = filtered_similarities[:n_recommendations]

        return pd.DataFrame({
            'job_idx': [job[0] for job in top_jobs],
            'similarity_score': [job[1] for job in top_jobs]
        })

    def _create_user_text_profile(self, user_profile):
        text_parts = []
        if user_profile.get('job_title'):
            text_parts.append(user_profile['job_title'])
        if user_profile.get('job_summary'):
            text_parts.append(user_profile['job_summary'])
        if user_profile.get('location'):
            text_parts.append(user_profile['location'])
        if user_profile.get('job_type'):
            text_parts.append(user_profile['job_type'])
        if user_profile.get('job_level'):
            text_parts.append(user_profile['job_level'])
        if user_profile.get('skills'):
            text_parts.append(' '.join(user_profile['skills']))
        return ' '.join(text_parts)

    def _create_user_feature_profile(self, user_profile):
        features = []
        min_salary = user_profile.get('min_salary', 0)
        max_salary = user_profile.get('max_salary', 200000)
        avg_salary = (min_salary + max_salary)/2
        features.extend([min_salary, max_salary, avg_salary])

        # Job type one-hot
        job_type = user_profile.get('job_type', 'Hybrid')
        emp_types = ['Hybrid', 'Remote', 'Onsite']
        for et in emp_types:
            features.append(1 if job_type == et else 0)

        # Job level
        job_level = user_profile.get('job_level', 'Unknown')
        level_mapping = {'Associate': 1, 'Mid senior': 2, 'Unknown': 0}
        features.append(level_mapping.get(job_level, 0))

        # Remote flag
        features.append(0.5)

        # Number of skills
        skills_count = len(user_profile.get('skills', []))
        features.append(skills_count)

        return features

    def _apply_user_constraints(self, user_profile):
        if self.data is None:
            return set()
        valid_jobs = set(range(len(self.data)))

        # Filter by location
        if user_profile.get('location'):
            location_mask = self.data['job_location'].str.contains(user_profile['location'], case=False, na=False)
            valid_jobs &= set(self.data[location_mask].index)

        # Filter by job type
        if user_profile.get('job_type'):
            job_type_mask = self.data['job_type'] == user_profile['job_type']
            valid_jobs &= set(self.data[job_type_mask].index)

        # Filter by job level
        if user_profile.get('job_level'):
            level_mapping = {'Associate': 1, 'Mid senior': 2}
            user_level_val = level_mapping.get(user_profile['job_level'], 0)
            job_levels = self.data['job_level'].map(level_mapping).fillna(0)
            valid_jobs &= set(self.data[job_levels == user_level_val].index)

        if not valid_jobs:
            valid_jobs = set(range(len(self.data)))  # fallback to all jobs

        return valid_jobs

    def get_job_similarities(self, job_idx, n_similar=10):
        if self.tfidf_matrix is None:
            return None

        job_similarities = cosine_similarity(
            self.tfidf_matrix[job_idx:job_idx+1],
            self.tfidf_matrix
        )[0]

        feature_similarities = cosine_similarity(
            self.feature_matrix[job_idx:job_idx+1],
            self.feature_matrix
        )[0]

        combined_similarities = 0.7*job_similarities + 0.3*feature_similarities

        similar_indices = np.argsort(combined_similarities)[::-1][1:n_similar+1]
        similar_scores = combined_similarities[similar_indices]

        return pd.DataFrame({
            'job_idx': similar_indices,
            'similarity_score': similar_scores
        })
