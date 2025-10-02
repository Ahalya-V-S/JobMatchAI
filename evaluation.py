import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class Evaluator:
    """Evaluation metrics for job recommendation systems using dataset columns:
       ['job_link', 'last_processed_time', 'got_summary', 'got_ner',
        'is_being_worked', 'job_title', 'company', 'job_location', 'first_seen',
        'search_city', 'search_country', 'search_position', 'job_level',
        'job_type']
    """

    def __init__(self, data):
        self.data = data
        self.test_users = None
        self.test_interactions = None
        self._prepare_test_data()

    def _prepare_test_data(self):
        """Generate synthetic test users"""
        np.random.seed(42)
        n_test_users = 50
        test_users = []

        job_levels = self.data['job_level'].dropna().unique().tolist()
        job_types = self.data['job_type'].dropna().unique().tolist()
        locations = self.data['job_location'].dropna().unique().tolist()

        for user_id in range(n_test_users):
            user_profile = {
                'user_id': user_id,
                'preferred_job_level': np.random.choice(job_levels) if job_levels else 'Any',
                'preferred_job_type': np.random.choice(job_types) if job_types else 'Any',
                'job_location': np.random.choice(locations) if locations else 'Any',
                'keywords': self._generate_random_keywords()
            }
            test_users.append(user_profile)

        self.test_users = test_users
        self._generate_ground_truth()

    def _generate_random_keywords(self):
        """Generate random keywords from job_title and company columns"""
        all_keywords = []
        for col in ['job_title', 'company']:
            if col in self.data.columns:
                sample_size = min(50, len(self.data))
                for val in self.data[col].dropna().sample(sample_size):
                    all_keywords.extend(val.lower().split())
        unique_keywords = list(set(all_keywords))
        n_keywords = np.random.randint(1, min(5, len(unique_keywords))) if unique_keywords else 0
        return np.random.choice(unique_keywords, size=n_keywords, replace=False).tolist() if n_keywords > 0 else []

    def _generate_ground_truth(self):
        """Generate ground truth interactions for test users"""
        ground_truth = []

        for user in self.test_users:
            relevant_jobs = self._find_relevant_jobs(user)
            for job_idx in relevant_jobs:
                ground_truth.append({
                    'user_id': user['user_id'],
                    'job_idx': job_idx,
                    'relevant': 1
                })

        self.test_interactions = pd.DataFrame(ground_truth)

    def _find_relevant_jobs(self, user_profile):
        """Find relevant jobs based on level, type, location, and keywords"""
        relevant_jobs = []

        for idx in range(len(self.data)):
            job = self.data.iloc[idx]
            score = 0

            # Job level match
            if job.get('job_level') == user_profile['preferred_job_level']:
                score += 3

            # Job type match
            if job.get('job_type') == user_profile['preferred_job_type']:
                score += 2

            # Location match
            if user_profile['job_location'] != 'Any' and pd.notna(job.get('job_location')):
                if user_profile['job_location'].lower() in job.get('job_location', '').lower():
                    score += 1

            # Keyword match
            job_text = f"{job.get('job_title', '')} {job.get('company', '')}".lower()
            keyword_matches = sum(1 for kw in user_profile['keywords'] if kw in job_text)
            score += keyword_matches

            if score >= 3:
                relevant_jobs.append(idx)

        return relevant_jobs[:20]

    def evaluate_precision_recall(self, recommender, k=10):
        """Evaluate precision and recall at k"""
        precisions = []
        recalls = []

        for user in self.test_users:
            user_id = user['user_id']
            user_truth = self.test_interactions[
                self.test_interactions['user_id'] == user_id
            ]['job_idx'].tolist()

            if not user_truth:
                continue

            try:
                recommendations = recommender.recommend(user, n_recommendations=k)
                if recommendations is None or len(recommendations) == 0:
                    precisions.append(0)
                    recalls.append(0)
                    continue

                recommended_jobs = recommendations['job_idx'].tolist()
                relevant_recommended = set(recommended_jobs) & set(user_truth)

                precision = len(relevant_recommended) / len(recommended_jobs) if recommended_jobs else 0
                recall = len(relevant_recommended) / len(user_truth) if user_truth else 0

                precisions.append(precision)
                recalls.append(recall)

            except Exception as e:
                st.warning(f"Error evaluating user {user_id}: {str(e)}")
                precisions.append(0)
                recalls.append(0)

        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        return avg_precision, avg_recall

    def evaluate_diversity(self, recommender, k=10):
        """Evaluate diversity as ratio of unique recommended items"""
        all_recommendations = []
        for user in self.test_users[:20]:
            try:
                recommendations = recommender.recommend(user, n_recommendations=k)
                if recommendations is not None and len(recommendations) > 0:
                    all_recommendations.extend(recommendations['job_idx'].tolist())
            except Exception:
                continue

        if not all_recommendations:
            return 0

        return len(set(all_recommendations)) / len(all_recommendations)

    def evaluate_coverage(self, recommender, k=10):
        """Evaluate catalog coverage"""
        recommended_items = set()
        for user in self.test_users[:20]:
            try:
                recommendations = recommender.recommend(user, n_recommendations=k)
                if recommendations is not None and len(recommendations) > 0:
                    recommended_items.update(recommendations['job_idx'].tolist())
            except Exception:
                continue

        total_items = len(self.data)
        return len(recommended_items) / total_items if total_items > 0 else 0

    def evaluate_novelty(self, recommender, k=10):
        """Evaluate novelty based on item popularity"""
        item_popularity = {}
        if hasattr(self, 'test_interactions') and self.test_interactions is not None:
            popularity_counts = self.test_interactions['job_idx'].value_counts()
            total_interactions = len(self.test_interactions)
            for job_idx in range(len(self.data)):
                item_popularity[job_idx] = popularity_counts.get(job_idx, 0) / total_interactions
        else:
            for job_idx in range(len(self.data)):
                item_popularity[job_idx] = 1.0 / len(self.data)

        novelty_scores = []
        for user in self.test_users[:20]:
            try:
                recommendations = recommender.recommend(user, n_recommendations=k)
                if recommendations is not None and len(recommendations) > 0:
                    user_novelty = 0
                    for job_idx in recommendations['job_idx']:
                        popularity = item_popularity.get(job_idx, 1.0 / len(self.data))
                        user_novelty += -np.log2(popularity) if popularity > 0 else 0
                    novelty_scores.append(user_novelty / len(recommendations))
            except Exception:
                continue

        return np.mean(novelty_scores) if novelty_scores else 0

    def evaluate_serendipity(self, recommender, k=10):
        """Evaluate serendipity as unexpected but relevant recommendations"""
        serendipity_scores = []
        for user in self.test_users[:10]:
            try:
                recommendations = recommender.recommend(user, n_recommendations=k)
                if recommendations is None or len(recommendations) == 0:
                    continue

                user_vector = self._create_user_profile_vector(user)
                serendipity_score = 0

                for job_idx in recommendations['job_idx']:
                    job = self.data.iloc[job_idx]
                    job_vector = self._create_job_vector(job)

                    similarity = cosine_similarity([user_vector], [job_vector])[0][0]
                    unexpectedness = 1 - similarity
                    relevance = self._calculate_simple_relevance(job, user)
                    serendipity_score += unexpectedness * relevance

                serendipity_scores.append(serendipity_score / len(recommendations))

            except Exception:
                continue

        return np.mean(serendipity_scores) if serendipity_scores else 0

    def _create_user_profile_vector(self, user):
        """Vectorize user profile: job_level, job_type, location, keywords"""
        vector = []

        level_mapping = {lvl: i+1 for i, lvl in enumerate(self.data['job_level'].dropna().unique())}
        vector.append(level_mapping.get(user.get('preferred_job_level'), 0))

        type_mapping = {t: i+1 for i, t in enumerate(self.data['job_type'].dropna().unique())}
        vector.append(type_mapping.get(user.get('preferred_job_type'), 0))

        # Location: hash to numeric
        vector.append(hash(user.get('job_location', 'Any')) % 100 / 100)

        # Keyword count
        vector.append(len(user.get('keywords', [])))
        return vector

    def _create_job_vector(self, job):
        """Vectorize job: level, type, location, title/company keyword count"""
        vector = []

        level_mapping = {lvl: i+1 for i, lvl in enumerate(self.data['job_level'].dropna().unique())}
        vector.append(level_mapping.get(job.get('job_level'), 0))

        type_mapping = {t: i+1 for i, t in enumerate(self.data['job_type'].dropna().unique())}
        vector.append(type_mapping.get(job.get('job_type'), 0))

        vector.append(hash(job.get('job_location', 'Any')) % 100 / 100)

        keyword_count = len(str(job.get('job_title', '')).split()) + len(str(job.get('company', '')).split())
        vector.append(keyword_count)
        return vector

    def _calculate_simple_relevance(self, job, user):
        """Simplified relevance for serendipity calculation"""
        relevance = 0
        if job.get('job_level') == user.get('preferred_job_level'):
            relevance += 0.4
        if job.get('job_type') == user.get('preferred_job_type'):
            relevance += 0.3
        if user.get('job_location', 'Any') != 'Any' and pd.notna(job.get('job_location')):
            if user.get('job_location').lower() in job.get('job_location', '').lower():
                relevance += 0.2

        job_text = f"{job.get('job_title', '')} {job.get('company', '')}".lower()
        keyword_matches = sum(1 for kw in user.get('keywords', []) if kw in job_text)
        relevance += min(0.1, 0.05 * keyword_matches)

        return min(1.0, relevance)

    # Evaluation methods for all systems and report generation remain same, no changes needed.
