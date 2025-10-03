import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta, timezone


class KnowledgeBasedFilter:
    """Enhanced Knowledge-based filtering with extra rules"""

    def __init__(self):
        self.data = None

    def fit(self, data):
        st.info("Initializing Knowledge-Based Filter...")
        self.data = data.copy()
        # Convert dates to datetime
        for col in ['first_seen', 'last_processed_time']:
            if col in self.data.columns:
                self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
        st.success("Knowledge-Based Filter initialization completed!")


    def recommend(self, user_profile, n_recommendations=10):
        if self.data is None:
            return None

        filtered = self.data.copy()

        # --- Job level ---
        if user_profile.get('job_level') and user_profile['job_level'] != 'Any':
            filtered['level_match'] = (filtered['job_level'] == user_profile['job_level']).astype(float)
        else:
            filtered['level_match'] = 0.5

        # --- Job type ---
        if user_profile.get('job_type') and user_profile['job_type'] != 'Any':
            filtered['type_match'] = (filtered['job_type'] == user_profile['job_type']).astype(float)
        else:
            filtered['type_match'] = 0.5

        # --- Location ---
        if user_profile.get('location') and user_profile['location'] != 'Any':
            filtered['location_match'] = filtered['job_location'].str.contains(
                user_profile['location'], case=False, na=False
            ).astype(float)
        else:
            filtered['location_match'] = 0.5

        # --- Skills match ---
        if user_profile.get('skills'):
            filtered['skill_match'] = filtered['job_skills'].fillna('').apply(
                lambda x: len(set(user_profile['skills']).intersection(set(s.strip() for s in x.split(',')))) / max(len(user_profile['skills']),1)
            )
        else:
            filtered['skill_match'] = 0.5

        # --- Recent jobs boost (last 30 days) ---
        recent_threshold = datetime.now(timezone.utc) - timedelta(days=30)
        if 'last_processed_time' in filtered.columns:
            filtered['last_processed_time'] = pd.to_datetime(filtered['last_processed_time'], utc=True)
            filtered['recent_boost'] = (filtered['last_processed_time'] >= recent_threshold).astype(float)
        else:
            filtered['recent_boost'] = 0.5

        # --- Remote/Hybrid preference ---
        if user_profile.get('job_type') in ['Remote', 'Hybrid']:
            filtered['remote_boost'] = (filtered['job_type'] == user_profile['job_type']).astype(float)
        else:
            filtered['remote_boost'] = 0.5

        # --- Overall weighted score ---
        filtered['kb_score'] = (
            0.3*filtered['level_match'] +
            0.2*filtered['type_match'] +
            0.2*filtered['location_match'] +
            0.1*filtered['skill_match'] +
            0.1*filtered['recent_boost'] +
            0.1*filtered['remote_boost']
        )

        # Include job_idx for hybrid recommender
        filtered['job_idx'] = filtered.index

        top_jobs = filtered.sort_values(by='kb_score', ascending=False).head(n_recommendations)
        return top_jobs[['job_idx', 'kb_score']]
