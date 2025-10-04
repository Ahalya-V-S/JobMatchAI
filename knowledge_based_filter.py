import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta, timezone
import difflib

class KnowledgeBasedFilter:
    """Robust Knowledge-Based Filtering for job recommendations"""

    def __init__(self):
        self.data = None
        self.state_abbrev = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
            'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
            'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
            'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
            'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
            'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
            'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
            'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
            'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
        }

    def fit(self, data: pd.DataFrame):
        """Initialize the filter and prepare dates"""
        st.info("Initializing Knowledge-Based Filter...")
        self.data = data.copy()
        for col in ['first_seen', 'last_processed_time']:
            if col in self.data.columns:
                self.data[col] = pd.to_datetime(self.data[col], errors='coerce', utc=True)
        st.success("Knowledge-Based Filter ready!")

    def _normalize_skills(self, skills):
        if pd.isna(skills):
            return []
        if isinstance(skills, list):
            return [s.strip().lower() for s in skills if s.strip()]
        return [s.strip().lower() for s in str(skills).split(',') if s.strip()]

    def _normalize_location(self, loc):
        if pd.isna(loc):
            return ''
        loc = str(loc).strip().upper()
        if loc in self.state_abbrev:
            return self.state_abbrev[loc].lower()
        return loc.lower()

    def _location_match(self, user_loc, job_loc):
        user_loc_norm = self._normalize_location(user_loc)
        job_loc_norm = self._normalize_location(job_loc)
        return difflib.SequenceMatcher(None, user_loc_norm, job_loc_norm).ratio()

    def recommend(self, user_profile: dict, n_recommendations: int = 10):
        if self.data is None or self.data.empty:
            st.warning("No data available for Knowledge-Based Filter")
            return pd.DataFrame(columns=['job_idx', 'kb_score'])

        df = self.data.copy()

        # --- Job level match ---
        user_level = user_profile.get('job_level', 'Any')
        df['level_match'] = 0.5
        if user_level != 'Any' and 'job_level' in df.columns:
            df['level_match'] = (df['job_level'] == user_level).astype(float)

        # --- Job type match ---
        user_type = user_profile.get('job_type', 'Any')
        df['type_match'] = 0.5
        if user_type != 'Any' and 'job_type' in df.columns:
            df['type_match'] = (df['job_type'] == user_type).astype(float)

        # --- Location match (fuzzy) ---
        user_loc = user_profile.get('location', 'Any')
        df['location_match'] = 0.5
        if user_loc != 'Any' and 'job_location' in df.columns:
            df['location_match'] = df['job_location'].apply(lambda x: self._location_match(user_loc, x))

        # --- Skills match ---
        user_skills = [s.lower() for s in user_profile.get('skills', [])]
        df['skill_match'] = 0.5
        if user_skills and 'job_skills' in df.columns:
            df['skill_match'] = df['job_skills'].apply(
                lambda x: len(set(user_skills).intersection(set(self._normalize_skills(x)))) / max(len(user_skills),1)
            )

        # --- Recent jobs boost (last 30 days) ---
        recent_threshold = datetime.now(timezone.utc) - timedelta(days=30)
        df['recent_boost'] = 0.5
        if 'last_processed_time' in df.columns:
            df['recent_boost'] = (df['last_processed_time'] >= recent_threshold).astype(float)

        # --- Remote/Hybrid preference ---
        df['remote_boost'] = 0.5
        if user_type in ['Remote', 'Hybrid'] and 'job_type' in df.columns:
            df['remote_boost'] = (df['job_type'] == user_type).astype(float)

        # --- Compute overall weighted score ---
        df['kb_score'] = (
            0.3*df['level_match'] +
            0.2*df['type_match'] +
            0.2*df['location_match'] +
            0.1*df['skill_match'] +
            0.1*df['recent_boost'] +
            0.1*df['remote_boost']
        )

        df['job_idx'] = df.index

        # Return top N
        top_jobs = df.sort_values(by='kb_score', ascending=False).head(n_recommendations)
        return top_jobs[['job_idx', 'kb_score']]
