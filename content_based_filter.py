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
        
        # 2. Process categorical and numerical features
        self._create_feature_matrix()
        
        st.success("Content-Based Filter training completed!")
    
    def _create_tfidf_features(self):
        """Create TF-IDF features from job descriptions and titles"""
        if self.data is None:
            st.error("No data available for TF-IDF feature creation")
            return

        text_features = []

        for idx in self.data.index:
            job = self.data.loc[idx]
            text_parts = []

            if pd.notna(job.get('title')):
                text_parts.extend([str(job['title'])]*3)
            if pd.notna(job.get('description')):
                text_parts.append(str(job['description']))
            if pd.notna(job.get('skills')):
                text_parts.extend([str(job['skills'])]*2)
            if pd.notna(job.get('company')):
                text_parts.append(str(job['company']))
            if pd.notna(job.get('location')):
                text_parts.append(str(job['location']))

            combined_text = ' '.join(text_parts)
            text_features.append(combined_text)

        # Remove empty strings
        text_features = [doc.strip() for doc in text_features if doc.strip()]

        if not text_features:
            raise ValueError("No valid text data found for TF-IDF. Check your dataset columns.")

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
        """Create numerical feature matrix for additional similarity"""
        if self.data is None:
            st.error("No data available for feature matrix creation")
            return

        features = []

        for idx in self.data.index:
            job = self.data.loc[idx]
            job_features = []

            # Salary features
            salary_min = job.get('salary_min', 0) if pd.notna(job.get('salary_min')) else 0
            salary_max = job.get('salary_max', 0) if pd.notna(job.get('salary_max')) else 0
            salary_avg = (salary_min + salary_max)/2 if salary_max > 0 else 0
            job_features.extend([salary_min, salary_max, salary_avg])

            # Experience level encoding
            exp_level = job.get('experience_level', 'Unknown')
            exp_mapping = {
                'Entry level': 1,
                'Associate': 2,
                'Mid-Senior level': 3,
                'Director': 4,
                'Executive': 5,
                'Unknown': 0
            }
            job_features.append(exp_mapping.get(exp_level, 0))

            # Employment type one-hot
            emp_type = job.get('employment_type', 'Unknown')
            emp_types = ['Full-time', 'Part-time', 'Contract', 'Temporary', 'Internship']
            for et in emp_types:
                job_features.append(1 if emp_type == et else 0)

            # Skills count
            skills_count = job.get('skills_count', 0) if pd.notna(job.get('skills_count')) else 0
            job_features.append(skills_count)

            # Remote flag
            is_remote = 1 if job.get('is_remote', False) else 0
            job_features.append(is_remote)

            features.append(job_features)

        # Normalize features
        self.feature_matrix = self.scaler.fit_transform(features)
        st.info(f"Created feature matrix with shape: {self.feature_matrix.shape}")
    
    def get_content_similarity(self, job_idx1, job_idx2):
        """Calculate content similarity between two jobs"""
        if self.tfidf_matrix is None or self.feature_matrix is None:
            return 0.0

        tfidf_sim = cosine_similarity(
            self.tfidf_matrix[job_idx1:job_idx1+1],
            self.tfidf_matrix[job_idx2:job_idx2+1]
        )[0][0]

        feature_sim = cosine_similarity(
            self.feature_matrix[job_idx1:job_idx1+1],
            self.feature_matrix[job_idx2:job_idx2+1]
        )[0][0]

        combined_sim = 0.7*tfidf_sim + 0.3*feature_sim
        return combined_sim

    def recommend(self, user_profile, n_recommendations=10):
        """Generate content-based recommendations for a user profile"""
        if self.tfidf_matrix is None or self.feature_matrix is None:
            return None

        user_text = self._create_user_text_profile(user_profile)
        if self.tfidf_vectorizer is None:
            return None
        user_vector = self.tfidf_vectorizer.transform([user_text])

        text_similarities = cosine_similarity(user_vector, self.tfidf_matrix)[0]

        user_features = self._create_user_feature_profile(user_profile)
        user_feature_vector = self.scaler.transform([user_features])

        feature_similarities = cosine_similarity(user_feature_vector, self.feature_matrix)[0]

        combined_similarities = 0.7*text_similarities + 0.3*feature_similarities

        valid_jobs = self._apply_user_constraints(user_profile)

        filtered_similarities = [
            (idx, sim) for idx, sim in enumerate(combined_similarities) if idx in valid_jobs
        ]

        filtered_similarities.sort(key=lambda x: x[1], reverse=True)
        top_jobs = filtered_similarities[:n_recommendations]

        return pd.DataFrame({
            'job_idx': [job[0] for job in top_jobs],
            'similarity_score': [job[1] for job in top_jobs]
        })

    def _create_user_text_profile(self, user_profile):
        text_parts = []
        if user_profile.get('skills'):
            skills_text = ' '.join(user_profile['skills'])
            text_parts.extend([skills_text]*3)
        if user_profile.get('industry'):
            text_parts.append(user_profile['industry'])
        if user_profile.get('experience_level'):
            text_parts.append(user_profile['experience_level'])
        if user_profile.get('location'):
            text_parts.append(user_profile['location'])
        return ' '.join(text_parts)

    def _create_user_feature_profile(self, user_profile):
        features = []
        min_salary = user_profile.get('min_salary', 0)
        max_salary = user_profile.get('max_salary', 200000)
        avg_salary = (min_salary + max_salary)/2
        features.extend([min_salary, max_salary, avg_salary])

        exp_level = user_profile.get('experience_level', 'Unknown')
        exp_mapping = {
            'Entry level': 1,
            'Associate': 2,
            'Mid-Senior level': 3,
            'Director': 4,
            'Executive': 5,
            'Unknown': 0
        }
        features.append(exp_mapping.get(exp_level, 0))

        job_type = user_profile.get('job_type', 'Full-time')
        emp_types = ['Full-time', 'Part-time', 'Contract', 'Temporary', 'Internship']
        for et in emp_types:
            features.append(1 if job_type == et else 0)

        skills_count = len(user_profile.get('skills', []))
        features.append(skills_count)

        # Remote flexibility
        features.append(0.5)

        return features

    def _apply_user_constraints(self, user_profile):
        if self.data is None:
            return set()
        valid_jobs = set(range(len(self.data)))

        if user_profile.get('location') and user_profile['location'] != 'Any':
            location_mask = self.data['location'].str.contains(user_profile['location'], case=False, na=False)
            valid_jobs &= set(self.data[location_mask].index)

        if user_profile.get('min_salary'):
            salary_mask = (self.data['salary_max'] >= user_profile['min_salary']) | self.data['salary_max'].isna()
            valid_jobs &= set(self.data[salary_mask].index)

        if user_profile.get('max_salary'):
            salary_mask = (self.data['salary_min'] <= user_profile['max_salary']) | self.data['salary_min'].isna()
            valid_jobs &= set(self.data[salary_mask].index)

        if user_profile.get('job_type') and user_profile['job_type'] != 'Any':
            job_type_mask = self.data['employment_type'] == user_profile['job_type']
            valid_jobs &= set(self.data[job_type_mask].index)

        if user_profile.get('experience_level'):
            exp_level = user_profile['experience_level']
            if exp_level == 'Entry level':
                exp_mask = self.data['experience_level'].isin(['Entry level','Associate'])
            elif exp_level == 'Associate':
                exp_mask = self.data['experience_level'].isin(['Entry level','Associate','Mid-Senior level'])
            elif exp_level == 'Mid-Senior level':
                exp_mask = self.data['experience_level'].isin(['Associate','Mid-Senior level','Director'])
            elif exp_level == 'Director':
                exp_mask = self.data['experience_level'].isin(['Mid-Senior level','Director','Executive'])
            else:
                exp_mask = pd.Series([True]*len(self.data), index=self.data.index)
            valid_jobs &= set(self.data[exp_mask].index)

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
