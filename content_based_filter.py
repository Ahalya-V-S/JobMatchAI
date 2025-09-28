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
        # Combine title, description, and skills for text analysis
        text_features = []
        
        for idx in self.data.index:
            job = self.data.loc[idx]
            
            text_parts = []
            
            # Add title (with higher weight by repeating)
            if pd.notna(job.get('title')):
                text_parts.extend([str(job['title'])] * 3)
            
            # Add description
            if pd.notna(job.get('description')):
                text_parts.append(str(job['description']))
            
            # Add skills (with higher weight)
            if pd.notna(job.get('skills')):
                text_parts.extend([str(job['skills'])] * 2)
            
            # Add company
            if pd.notna(job.get('company')):
                text_parts.append(str(job['company']))
            
            # Add location
            if pd.notna(job.get('location')):
                text_parts.append(str(job['location']))
            
            combined_text = ' '.join(text_parts)
            text_features.append(combined_text)
        
        # Create TF-IDF matrix
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_features)
        st.info(f"Created TF-IDF matrix with shape: {self.tfidf_matrix.shape}")
    
    def _create_feature_matrix(self):
        """Create numerical feature matrix for additional similarity"""
        features = []
        
        for idx in self.data.index:
            job = self.data.loc[idx]
            job_features = []
            
            # Salary features
            salary_min = job.get('salary_min', 0) if pd.notna(job.get('salary_min')) else 0
            salary_max = job.get('salary_max', 0) if pd.notna(job.get('salary_max')) else 0
            salary_avg = (salary_min + salary_max) / 2 if salary_max > 0 else 0
            
            job_features.extend([salary_min, salary_max, salary_avg])
            
            # Experience level (numerical encoding)
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
            
            # Employment type (one-hot encoding)
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
        # Get TF-IDF similarity
        tfidf_sim = cosine_similarity(
            self.tfidf_matrix[job_idx1:job_idx1+1],
            self.tfidf_matrix[job_idx2:job_idx2+1]
        )[0][0]
        
        # Get feature similarity
        feature_sim = cosine_similarity(
            self.feature_matrix[job_idx1:job_idx1+1],
            self.feature_matrix[job_idx2:job_idx2+1]
        )[0][0]
        
        # Combine similarities (TF-IDF weighted more heavily)
        combined_sim = 0.7 * tfidf_sim + 0.3 * feature_sim
        
        return combined_sim
    
    def recommend(self, user_profile, n_recommendations=10):
        """Generate content-based recommendations for a user profile"""
        if self.tfidf_matrix is None or self.feature_matrix is None:
            return None
        
        # Create user profile vector
        user_text = self._create_user_text_profile(user_profile)
        user_vector = self.tfidf_vectorizer.transform([user_text])
        
        # Calculate similarities with all jobs
        text_similarities = cosine_similarity(user_vector, self.tfidf_matrix)[0]
        
        # Create user feature vector
        user_features = self._create_user_feature_profile(user_profile)
        user_feature_vector = self.scaler.transform([user_features])
        
        # Calculate feature similarities
        feature_similarities = cosine_similarity(user_feature_vector, self.feature_matrix)[0]
        
        # Combine similarities
        combined_similarities = 0.7 * text_similarities + 0.3 * feature_similarities
        
        # Apply user constraints (knowledge-based filtering)
        valid_jobs = self._apply_user_constraints(user_profile)
        
        # Filter similarities to only valid jobs
        filtered_similarities = []
        for idx, sim in enumerate(combined_similarities):
            if idx in valid_jobs:
                filtered_similarities.append((idx, sim))
        
        # Sort by similarity and get top recommendations
        filtered_similarities.sort(key=lambda x: x[1], reverse=True)
        top_jobs = filtered_similarities[:n_recommendations]
        
        # Return as DataFrame with scores
        recommendations = pd.DataFrame({
            'job_idx': [job[0] for job in top_jobs],
            'similarity_score': [job[1] for job in top_jobs]
        })
        
        return recommendations
    
    def _create_user_text_profile(self, user_profile):
        """Create text representation of user profile"""
        text_parts = []
        
        # Add skills (most important)
        if user_profile.get('skills'):
            skills_text = ' '.join(user_profile['skills'])
            text_parts.extend([skills_text] * 3)  # Higher weight
        
        # Add industry preference
        if user_profile.get('industry'):
            text_parts.append(user_profile['industry'])
        
        # Add experience level
        if user_profile.get('experience_level'):
            text_parts.append(user_profile['experience_level'])
        
        # Add location preference
        if user_profile.get('location'):
            text_parts.append(user_profile['location'])
        
        return ' '.join(text_parts)
    
    def _create_user_feature_profile(self, user_profile):
        """Create numerical feature representation of user profile"""
        features = []
        
        # Salary preferences
        min_salary = user_profile.get('min_salary', 0)
        max_salary = user_profile.get('max_salary', 200000)
        avg_salary = (min_salary + max_salary) / 2
        
        features.extend([min_salary, max_salary, avg_salary])
        
        # Experience level
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
        
        # Employment type preference
        job_type = user_profile.get('job_type', 'Full-time')
        emp_types = ['Full-time', 'Part-time', 'Contract', 'Temporary', 'Internship']
        for et in emp_types:
            features.append(1 if job_type == et else 0)
        
        # Skills count (number of skills user has)
        skills_count = len(user_profile.get('skills', []))
        features.append(skills_count)
        
        # Remote preference (assume flexible)
        features.append(0.5)
        
        return features
    
    def _apply_user_constraints(self, user_profile):
        """Apply hard constraints based on user preferences"""
        valid_jobs = set(range(len(self.data)))
        
        # Location constraint
        if user_profile.get('location') and user_profile['location'] != 'Any':
            location_mask = self.data['location'].str.contains(
                user_profile['location'], case=False, na=False
            )
            valid_jobs &= set(self.data[location_mask].index)
        
        # Salary constraint
        if user_profile.get('min_salary'):
            # Filter jobs where salary_max >= user's minimum
            salary_mask = (self.data['salary_max'] >= user_profile['min_salary']) | self.data['salary_max'].isna()
            valid_jobs &= set(self.data[salary_mask].index)
        
        if user_profile.get('max_salary'):
            # Filter jobs where salary_min <= user's maximum
            salary_mask = (self.data['salary_min'] <= user_profile['max_salary']) | self.data['salary_min'].isna()
            valid_jobs &= set(self.data[salary_mask].index)
        
        # Job type constraint
        if user_profile.get('job_type') and user_profile['job_type'] != 'Any':
            job_type_mask = self.data['employment_type'] == user_profile['job_type']
            valid_jobs &= set(self.data[job_type_mask].index)
        
        # Experience level constraint (flexible matching)
        if user_profile.get('experience_level'):
            exp_level = user_profile['experience_level']
            # Allow some flexibility in experience matching
            if exp_level == 'Entry level':
                exp_mask = self.data['experience_level'].isin(['Entry level', 'Associate'])
            elif exp_level == 'Associate':
                exp_mask = self.data['experience_level'].isin(['Entry level', 'Associate', 'Mid-Senior level'])
            elif exp_level == 'Mid-Senior level':
                exp_mask = self.data['experience_level'].isin(['Associate', 'Mid-Senior level', 'Director'])
            elif exp_level == 'Director':
                exp_mask = self.data['experience_level'].isin(['Mid-Senior level', 'Director', 'Executive'])
            else:
                exp_mask = pd.Series([True] * len(self.data), index=self.data.index)
            
            valid_jobs &= set(self.data[exp_mask].index)
        
        return valid_jobs
    
    def get_job_similarities(self, job_idx, n_similar=10):
        """Get similar jobs to a given job"""
        if self.tfidf_matrix is None:
            return None
        
        # Calculate similarities with all other jobs
        job_similarities = cosine_similarity(
            self.tfidf_matrix[job_idx:job_idx+1],
            self.tfidf_matrix
        )[0]
        
        # Get feature similarities
        feature_similarities = cosine_similarity(
            self.feature_matrix[job_idx:job_idx+1],
            self.feature_matrix
        )[0]
        
        # Combine similarities
        combined_similarities = 0.7 * job_similarities + 0.3 * feature_similarities
        
        # Get top similar jobs (excluding the job itself)
        similar_indices = np.argsort(combined_similarities)[::-1][1:n_similar+1]
        similar_scores = combined_similarities[similar_indices]
        
        return pd.DataFrame({
            'job_idx': similar_indices,
            'similarity_score': similar_scores
        })
