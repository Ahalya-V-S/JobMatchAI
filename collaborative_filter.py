import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

class CollaborativeFilter:
    """Collaborative filtering using matrix factorization (SVD)"""
    
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.svd_model = None
        self.user_item_matrix = None
        self.user_mapper = None
        self.item_mapper = None
        self.reverse_user_mapper = None
        self.reverse_item_mapper = None
        self.user_factors = None
        self.item_factors = None
        self.data = None
    
    def fit(self, data):
        """Train the collaborative filtering model"""
        st.info("Training Collaborative Filter...")
        
        self.data = data.copy()
        
        # Generate synthetic user-item interactions
        interactions_df = self._generate_interactions(data)
        
        if interactions_df is None or len(interactions_df) == 0:
            st.warning("No interactions generated for collaborative filtering")
            return
        
        # Create user-item matrix
        self._create_user_item_matrix(interactions_df)
        
        # Apply matrix factorization
        self._apply_matrix_factorization()
        
        st.success("Collaborative Filter training completed!")
    
    def _generate_interactions(self, data):
        """Generate synthetic user-item interactions based on job characteristics"""
        np.random.seed(42)
        
        n_users = min(500, len(data) // 20)  # Reasonable number of users
        n_jobs = len(data)
        
        interactions = []
        
        # Define user archetypes based on preferences
        user_archetypes = [
            {'type': 'entry_level_tech', 'exp': 'Entry level', 'category': 'Engineering', 'salary_pref': 'low'},
            {'type': 'senior_manager', 'exp': 'Director', 'category': 'Management', 'salary_pref': 'high'},
            {'type': 'data_analyst', 'exp': 'Associate', 'category': 'Data', 'salary_pref': 'medium'},
            {'type': 'sales_rep', 'exp': 'Associate', 'category': 'Sales', 'salary_pref': 'medium'},
            {'type': 'designer', 'exp': 'Mid-Senior level', 'category': 'Design', 'salary_pref': 'medium'},
            {'type': 'hr_specialist', 'exp': 'Associate', 'category': 'HR', 'salary_pref': 'medium'},
            {'type': 'finance_analyst', 'exp': 'Associate', 'category': 'Finance', 'salary_pref': 'medium'},
            {'type': 'marketing_manager', 'exp': 'Mid-Senior level', 'category': 'Marketing', 'salary_pref': 'medium'},
        ]
        
        for user_id in range(n_users):
            # Assign archetype to user
            archetype = np.random.choice(user_archetypes)
            
            # Each user interacts with 10-30 jobs
            n_interactions = np.random.randint(10, 31)
            
            # Sample jobs with preference bias
            job_indices = self._sample_jobs_with_bias(data, archetype, n_interactions)
            
            for job_idx in job_indices:
                job = data.iloc[job_idx]
                
                # Generate rating based on job-user fit
                rating = self._generate_rating(job, archetype)
                
                interactions.append({
                    'user_id': user_id,
                    'job_id': job_idx,
                    'rating': rating
                })
        
        return pd.DataFrame(interactions)
    
    def _sample_jobs_with_bias(self, data, archetype, n_interactions):
        """Sample jobs with bias towards user preferences"""
        n_jobs = len(data)
        
        # Calculate preference scores for all jobs
        preference_scores = np.zeros(n_jobs)
        
        for idx in range(n_jobs):
            job = data.iloc[idx]
            score = 1.0  # Base score
            
            # Experience level preference
            if 'experience_level' in data.columns:
                if job.get('experience_level') == archetype['exp']:
                    score += 2.0
                elif archetype['exp'] == 'Entry level' and job.get('experience_level') == 'Associate':
                    score += 1.0
                elif archetype['exp'] == 'Associate' and job.get('experience_level') in ['Entry level', 'Mid-Senior level']:
                    score += 1.0
                elif archetype['exp'] == 'Mid-Senior level' and job.get('experience_level') in ['Associate', 'Director']:
                    score += 1.0
            
            # Job category preference
            if 'job_category' in data.columns:
                if job.get('job_category') == archetype['category']:
                    score += 2.0
            
            # Salary preference
            if 'salary_avg' in data.columns and pd.notna(job.get('salary_avg')):
                salary = job['salary_avg']
                if archetype['salary_pref'] == 'low' and salary < 60000:
                    score += 1.0
                elif archetype['salary_pref'] == 'medium' and 60000 <= salary <= 120000:
                    score += 1.0
                elif archetype['salary_pref'] == 'high' and salary > 120000:
                    score += 1.0
            
            preference_scores[idx] = score
        
        # Normalize scores to probabilities
        probabilities = preference_scores / preference_scores.sum()
        
        # Sample jobs based on preferences
        job_indices = np.random.choice(n_jobs, size=n_interactions, replace=False, p=probabilities)
        
        return job_indices
    
    def _generate_rating(self, job, archetype):
        """Generate rating based on job-user archetype fit"""
        base_rating = np.random.uniform(2.5, 3.5)  # Base rating
        
        # Adjust rating based on fit
        # Experience level fit
        if job.get('experience_level') == archetype['exp']:
            base_rating += 1.0
        elif archetype['exp'] == 'Entry level' and job.get('experience_level') == 'Associate':
            base_rating += 0.5
        elif archetype['exp'] == 'Associate' and job.get('experience_level') in ['Entry level', 'Mid-Senior level']:
            base_rating += 0.5
        elif archetype['exp'] == 'Mid-Senior level' and job.get('experience_level') in ['Associate', 'Director']:
            base_rating += 0.5
        
        # Job category fit
        if job.get('job_category') == archetype['category']:
            base_rating += 1.0
        
        # Salary fit
        if 'salary_avg' in job and pd.notna(job.get('salary_avg')):
            salary = job['salary_avg']
            if archetype['salary_pref'] == 'low' and salary < 60000:
                base_rating += 0.5
            elif archetype['salary_pref'] == 'medium' and 60000 <= salary <= 120000:
                base_rating += 0.5
            elif archetype['salary_pref'] == 'high' and salary > 120000:
                base_rating += 0.5
            else:
                base_rating -= 0.5  # Salary mismatch penalty
        
        # Add some noise
        base_rating += np.random.uniform(-0.5, 0.5)
        
        # Ensure rating is in valid range
        return max(1.0, min(5.0, base_rating))
    
    def _create_user_item_matrix(self, interactions_df):
        """Create user-item interaction matrix"""
        # Create mappings
        unique_users = interactions_df['user_id'].unique()
        unique_items = interactions_df['job_id'].unique()
        
        self.user_mapper = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapper = {item: idx for idx, item in enumerate(unique_items)}
        self.reverse_user_mapper = {idx: user for user, idx in self.user_mapper.items()}
        self.reverse_item_mapper = {idx: item for item, idx in self.item_mapper.items()}
        
        # Map user and item IDs
        interactions_df['user_idx'] = interactions_df['user_id'].map(self.user_mapper)
        interactions_df['item_idx'] = interactions_df['job_id'].map(self.item_mapper)
        
        # Create sparse matrix
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        self.user_item_matrix = csr_matrix(
            (interactions_df['rating'], (interactions_df['user_idx'], interactions_df['item_idx'])),
            shape=(n_users, n_items)
        )
        
        st.info(f"Created user-item matrix with shape: {self.user_item_matrix.shape}")
    
    def _apply_matrix_factorization(self):
        """Apply SVD matrix factorization"""
        self.svd_model = TruncatedSVD(n_components=self.n_components, random_state=42)
        
        # Fit SVD on user-item matrix
        self.user_factors = self.svd_model.fit_transform(self.user_item_matrix)
        self.item_factors = self.svd_model.components_.T
        
        st.info(f"SVD decomposition completed with {self.n_components} components")
    
    def recommend(self, user_profile, n_recommendations=10):
        """Generate collaborative filtering recommendations"""
        if self.svd_model is None or self.user_factors is None:
            return None
        
        # Create synthetic user based on profile
        synthetic_user = self._create_synthetic_user(user_profile)
        
        if synthetic_user is None:
            return None
        
        # Find similar users
        user_similarities = cosine_similarity([synthetic_user], self.user_factors)[0]
        
        # Get top similar users
        top_user_indices = np.argsort(user_similarities)[::-1][:20]  # Top 20 similar users
        
        # Aggregate recommendations from similar users
        job_scores = {}
        
        for user_idx in top_user_indices:
            user_similarity = user_similarities[user_idx]
            
            # Get jobs this user rated highly (>= 4.0)
            user_ratings = self.user_item_matrix[user_idx].toarray()[0]
            high_rated_items = np.where(user_ratings >= 4.0)[0]
            
            for item_idx in high_rated_items:
                job_id = self.reverse_item_mapper[item_idx]
                rating = user_ratings[item_idx]
                
                if job_id not in job_scores:
                    job_scores[job_id] = 0
                
                # Weight the rating by user similarity
                job_scores[job_id] += rating * user_similarity
        
        # Sort jobs by aggregated scores
        if not job_scores:
            return None
        
        sorted_jobs = sorted(job_scores.items(), key=lambda x: x[1], reverse=True)
        top_jobs = sorted_jobs[:n_recommendations]
        
        # Return as DataFrame
        recommendations = pd.DataFrame({
            'job_idx': [job[0] for job in top_jobs],
            'cf_score': [job[1] for job in top_jobs]
        })
        
        return recommendations
    
    def _create_synthetic_user(self, user_profile):
        """Create a synthetic user vector based on user profile"""
        if self.item_factors is None:
            return None
        
        # Find jobs that match user profile closely
        matching_jobs = self._find_matching_jobs(user_profile)
        
        if not matching_jobs:
            return None
        
        # Create user vector as average of matching job factors
        synthetic_user = np.zeros(self.n_components)
        
        for job_id in matching_jobs:
            if job_id in self.item_mapper:
                item_idx = self.item_mapper[job_id]
                synthetic_user += self.item_factors[item_idx]
        
        synthetic_user /= len(matching_jobs)  # Average
        
        return synthetic_user
    
    def _find_matching_jobs(self, user_profile):
        """Find jobs that match user profile for creating synthetic user"""
        matching_jobs = []
        
        # Look for jobs that match user preferences
        for job_idx in range(len(self.data)):
            job = self.data.iloc[job_idx]
            match_score = 0
            
            # Experience level match
            if user_profile.get('experience_level') and job.get('experience_level') == user_profile['experience_level']:
                match_score += 2
            
            # Skills match
            if user_profile.get('skills') and 'skills_list' in job:
                user_skills = set(user_profile['skills'])
                job_skills = set(job.get('skills_list', []))
                if user_skills & job_skills:  # Intersection
                    match_score += len(user_skills & job_skills)
            
            # Salary range match
            if (user_profile.get('min_salary') and job.get('salary_min') and 
                job['salary_min'] >= user_profile['min_salary']):
                match_score += 1
            
            # Location match
            if (user_profile.get('location') and job.get('location') and 
                user_profile['location'] in job['location']):
                match_score += 1
            
            # If job has a good match score, include it
            if match_score >= 2:
                matching_jobs.append(job_idx)
        
        return matching_jobs[:50]  # Limit to top 50 matching jobs
    
    def get_user_similarities(self, user_id):
        """Get similar users to a given user"""
        if self.user_factors is None or user_id not in self.user_mapper:
            return None
        
        user_idx = self.user_mapper[user_id]
        user_vector = self.user_factors[user_idx:user_idx+1]
        
        similarities = cosine_similarity(user_vector, self.user_factors)[0]
        
        # Get top similar users (excluding the user itself)
        similar_indices = np.argsort(similarities)[::-1][1:11]  # Top 10 similar users
        similar_scores = similarities[similar_indices]
        
        similar_users = [self.reverse_user_mapper[idx] for idx in similar_indices]
        
        return pd.DataFrame({
            'user_id': similar_users,
            'similarity_score': similar_scores
        })
    
    def get_item_similarities(self, job_id):
        """Get similar items to a given job"""
        if self.item_factors is None or job_id not in self.item_mapper:
            return None
        
        item_idx = self.item_mapper[job_id]
        item_vector = self.item_factors[item_idx:item_idx+1]
        
        similarities = cosine_similarity(item_vector, self.item_factors)[0]
        
        # Get top similar items (excluding the item itself)
        similar_indices = np.argsort(similarities)[::-1][1:11]  # Top 10 similar items
        similar_scores = similarities[similar_indices]
        
        similar_items = [self.reverse_item_mapper[idx] for idx in similar_indices]
        
        return pd.DataFrame({
            'job_id': similar_items,
            'similarity_score': similar_scores
        })
