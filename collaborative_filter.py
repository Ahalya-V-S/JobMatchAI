import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

class CollaborativeFilter:
    """Collaborative filtering using SVD and synthetic user profile matching"""

    def __init__(self, n_components=20):
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
        st.info("Training Collaborative Filter...")
        self.data = data.copy()

        interactions_df = self._generate_interactions(self.data)
        if interactions_df is None or len(interactions_df) == 0:
            st.warning("No interactions generated for collaborative filtering")
            return

        self._create_user_item_matrix(interactions_df)
        self._apply_matrix_factorization()
        st.success("Collaborative Filter training completed!")

    def _generate_interactions(self, data):
        np.random.seed(42)
        n_users = min(200, len(data)//10)
        interactions = []

        user_levels = data['job_level'].unique().tolist() if 'job_level' in data.columns else ['Any']
        user_types = data['job_type'].unique().tolist() if 'job_type' in data.columns else ['Any']

        for user_id in range(n_users):
            level_pref = np.random.choice(user_levels)
            type_pref = np.random.choice(user_types)
            n_interactions = np.random.randint(5, 15)

            scores = np.ones(len(data))
            if 'job_level' in data.columns:
                scores += (data['job_level'] == level_pref).astype(float) * 2
            if 'job_type' in data.columns:
                scores += (data['job_type'] == type_pref).astype(float) * 1.5

            job_indices = np.random.choice(len(data), size=n_interactions, replace=False, p=scores/scores.sum())
            for job_idx in job_indices:
                rating = np.random.uniform(3,5)
                interactions.append({'user_id': user_id, 'job_id': job_idx, 'rating': rating})

        return pd.DataFrame(interactions)

    def _create_user_item_matrix(self, interactions_df):
        unique_users = interactions_df['user_id'].unique()
        all_job_ids = np.arange(len(self.data))

        self.user_mapper = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapper = {item: idx for idx, item in enumerate(all_job_ids)}
        self.reverse_user_mapper = {idx: user for user, idx in self.user_mapper.items()}
        self.reverse_item_mapper = {idx: item for item, idx in self.item_mapper.items()}

        interactions_df['user_idx'] = interactions_df['user_id'].map(self.user_mapper)
        interactions_df['item_idx'] = interactions_df['job_id'].map(self.item_mapper)

        self.user_item_matrix = csr_matrix(
            (interactions_df['rating'], (interactions_df['user_idx'], interactions_df['item_idx'])),
            shape=(len(unique_users), len(all_job_ids))
        )
        st.info(f"User-item matrix shape: {self.user_item_matrix.shape}")

    def _apply_matrix_factorization(self):
        self.svd_model = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.user_factors = self.svd_model.fit_transform(self.user_item_matrix)
        self.item_factors = self.svd_model.components_.T
        st.info(f"SVD completed with {self.n_components} components")

    def recommend(self, user_profile, n_recommendations=10):
        if self.item_factors is None:
            return None

        # Step 1: Assign higher scores to jobs matching the user's profile
        job_scores = np.ones(self.user_item_matrix.shape[1])
        if 'job_level' in self.data.columns and user_profile.get('job_level'):
            job_scores += (self.data['job_level'] == user_profile['job_level']).astype(float) * 2
        if 'job_type' in self.data.columns and user_profile.get('job_type'):
            job_scores += (self.data['job_type'] == user_profile['job_type']).astype(float) * 1.5
        if 'location' in self.data.columns and user_profile.get('location'):
            job_scores += self.data['job_location'].str.contains(user_profile['location'], case=False, na=False).astype(float) * 2

        # Step 2: Create synthetic user vector in latent space
        synthetic_user = np.dot(job_scores, self.item_factors) / np.sum(job_scores)

        # Step 3: Find similar users in latent space
        user_similarities = cosine_similarity([synthetic_user], self.user_factors)[0]
        top_user_idx = np.argsort(user_similarities)[::-1][:5]

        # Step 4: Aggregate top jobs from similar users
        final_job_scores = {}
        for uidx in top_user_idx:
            ratings = self.user_item_matrix[uidx].toarray()[0]
            for jdx, r in enumerate(ratings):
                if r >= 4.0:  # Consider only strong preferences
                    final_job_scores[jdx] = final_job_scores.get(jdx, 0) + r * user_similarities[uidx]

        # Step 5: Return top N jobs
        top_jobs = sorted(final_job_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        return pd.DataFrame({'job_idx':[j[0] for j in top_jobs], 'cf_score':[j[1] for j in top_jobs]})
