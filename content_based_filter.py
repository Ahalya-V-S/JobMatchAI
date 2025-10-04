import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import streamlit as st
import difflib

class ContentBasedFilter:
    """Content-based filtering using TF-IDF and job features"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.feature_matrix = None
        self.scaler = StandardScaler()
        self.job_indices = None
        self.data = None
        self.location_col = None  # Resolved column name for location
        # State abbreviations to full names
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
        # Common city abbreviations to full names
        self.city_abbrev = {
            'NYC': 'New York', 'LA': 'Los Angeles', 'SF': 'San Francisco', 'CHI': 'Chicago',
            'BOS': 'Boston', 'DC': 'Washington', 'ATL': 'Atlanta', 'MIA': 'Miami',
            'SEA': 'Seattle', 'DEN': 'Denver', 'PHX': 'Phoenix', 'HOU': 'Houston',
            'DAL': 'Dallas', 'PHL': 'Philadelphia', 'DET': 'Detroit', 'MIN': 'Minneapolis',
            'STL': 'St. Louis', 'CLT': 'Charlotte', 'LAS': 'Las Vegas', 'ORL': 'Orlando'
        }
    
    def fit(self, data):
        """Train the content-based filter"""
        st.info("Training Content-Based Filter...")
        self.data = data.copy()
        self.job_indices = data.index

        # Resolve location column
        if 'job_location' in self.data.columns:
            self.location_col = 'job_location'
        elif 'location' in self.data.columns:
            self.location_col = 'location'
        else:
            st.warning("No location column ('job_location' or 'location') found in dataset")
            self.location_col = None
        
        # Process text features using TF-IDF
        self._create_tfidf_features()
        
        # Process numerical/categorical features
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
            if self.location_col and pd.notna(job.get(self.location_col)):
                text_parts.append(str(job[self.location_col]))
            if pd.notna(job.get('job_type')):
                text_parts.append(str(job['job_type']))
            if pd.notna(job.get('job_level')):
                text_parts.append(str(job['job_level']))
            if pd.notna(job.get('job_skills')):
                # Handle job_skills as list or string
                skills = job['job_skills'] if isinstance(job['job_skills'], list) else str(job['job_skills']).split(',')
                text_parts.append(' '.join([str(skill).strip() for skill in skills]))

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

            # Placeholder for salary
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
            skills = job.get('job_skills', []) if isinstance(job.get('job_skills'), list) else str(job.get('job_skills', '')).split(',')
            skills_count = len([skill for skill in skills if skill.strip()])
            job_features.append(skills_count)

            features.append(job_features)

        self.feature_matrix = self.scaler.fit_transform(features)
        st.info(f"Created feature matrix with shape: {self.feature_matrix.shape}")

    def _normalize_location(self, loc):
        """Normalize location by expanding abbreviations"""
        if pd.isna(loc):
            return ''
        loc = str(loc).strip().upper()
        if loc in self.state_abbrev:
            return self.state_abbrev[loc].lower()
        if loc in self.city_abbrev:
            return self.city_abbrev[loc].lower()
        return loc.lower()

    def match_locations(self, query, threshold=60):
        """Return indices of jobs with fuzzy-matched locations"""
        if self.data is None or not self.location_col:
            return set()
        results = set()
        query_norm = self._normalize_location(query)
        for idx, val in self.data[self.location_col].items():
            val_norm = self._normalize_location(val)
            score = difflib.SequenceMatcher(None, query_norm, val_norm).ratio() * 100
            if score >= threshold:
                results.add(idx)
        return results

    def recommend(self, user_profile, n_recommendations=10):
        """Generate content-based recommendations for a user profile"""
        if self.tfidf_matrix is None or self.feature_matrix is None:
            st.error("Model not trained; cannot generate recommendations")
            return pd.DataFrame({'job_idx': [], 'similarity_score': []})

        user_text = self._create_user_text_profile(user_profile)
        user_vector = self.tfidf_vectorizer.transform([user_text])

        text_similarities = cosine_similarity(user_vector, self.tfidf_matrix)[0]
        user_features = self._create_user_feature_profile(user_profile)
        user_feature_vector = self.scaler.transform([user_features])
        feature_similarities = cosine_similarity(user_feature_vector, self.feature_matrix)[0]

        combined_similarities = 0.7 * text_similarities + 0.3 * feature_similarities
        valid_jobs = self._apply_user_constraints(user_profile)

        filtered_similarities = [(idx, sim) for idx, sim in enumerate(combined_similarities) if idx in valid_jobs]
        filtered_similarities.sort(key=lambda x: x[1], reverse=True)
        top_jobs = filtered_similarities[:n_recommendations]

        if not top_jobs:
            st.warning(f"No recommendations found for profile: {user_profile}")
        else:
            st.info(f"Found {len(top_jobs)} content-based recommendations")

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
            text_parts.append(' '.join([str(skill).strip() for skill in user_profile['skills']]))
        return ' '.join(text_parts)

    def _create_user_feature_profile(self, user_profile):
        features = []
        min_salary = user_profile.get('min_salary', 0)
        max_salary = user_profile.get('max_salary', 200000)
        avg_salary = (min_salary + max_salary) / 2
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
        skills_count = len([skill for skill in user_profile.get('skills', []) if skill.strip()])
        features.append(skills_count)

        return features

    def _apply_user_constraints(self, user_profile):
        if self.data is None:
            st.warning("No data available for applying constraints")
            return set()
        valid_jobs = set(range(len(self.data)))

        # Filter by location
        if user_profile.get('location') and self.location_col:
            matched_indices = self.match_locations(user_profile['location'])
            if not matched_indices:
                st.warning(f"No jobs found matching location '{user_profile['location']}'")
            valid_jobs &= matched_indices

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
            st.warning("No jobs matched the constraints; falling back to all jobs")
            valid_jobs = set(range(len(self.data)))

        return valid_jobs

    def get_job_similarities(self, job_idx, n_similar=10):
        if self.tfidf_matrix is None:
            st.error("TF-IDF matrix not initialized")
            return None

        job_similarities = cosine_similarity(
            self.tfidf_matrix[job_idx:job_idx+1],
            self.tfidf_matrix
        )[0]

        feature_similarities = cosine_similarity(
            self.feature_matrix[job_idx:job_idx+1],
            self.feature_matrix
        )[0]

        combined_similarities = 0.7 * job_similarities + 0.3 * feature_similarities

        similar_indices = np.argsort(combined_similarities)[::-1][1:n_similar+1]
        similar_scores = combined_similarities[similar_indices]

        return pd.DataFrame({
            'job_idx': similar_indices,
            'similarity_score': similar_scores
        })