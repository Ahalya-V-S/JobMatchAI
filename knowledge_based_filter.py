import pandas as pd
import streamlit as st

class KnowledgeBasedFilter:
    """Knowledge-based filtering using only job_level, job_type, and location"""
    
    def __init__(self):
        self.data = None
    
    def fit(self, data):
        st.info("Initializing Knowledge-Based Filter...")
        self.data = data.copy()
        st.success("Knowledge-Based Filter initialization completed!")
    
    def recommend(self, user_profile, n_recommendations=10):
        if self.data is None:
            return None
        
        filtered = self.data.copy()
        
        # Filter by job_level
        if user_profile.get('job_level') and user_profile['job_level'] != 'Any':
            filtered['level_match'] = (filtered['job_level'] == user_profile['job_level']).astype(float)
        else:
            filtered['level_match'] = 0.5  # neutral score
        
        # Filter by job_type
        if user_profile.get('job_type') and user_profile['job_type'] != 'Any':
            filtered['type_match'] = (filtered['job_type'] == user_profile['job_type']).astype(float)
        else:
            filtered['type_match'] = 0.5
        
        # Filter by location
        if user_profile.get('location') and user_profile['location'] != 'Any':
            filtered['location_match'] = filtered['job_location'].str.contains(
                user_profile['location'], case=False, na=False
            ).astype(float)
        else:
            filtered['location_match'] = 0.5
        
        # Compute overall score (weighted)
        filtered['kb_score'] = 0.4*filtered['level_match'] + 0.3*filtered['type_match'] + 0.3*filtered['location_match']
        
        # Include job_idx so hybrid recommender works
        filtered['job_idx'] = filtered.index
        
        top_jobs = filtered.sort_values(by='kb_score', ascending=False).head(n_recommendations)
        return top_jobs[['job_idx', 'kb_score']]
