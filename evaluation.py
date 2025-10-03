import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ Cached Evaluator ------------------
@st.cache_data(show_spinner=False)
def get_evaluator(data, n_test_users=5, sample_size=200):
    return Evaluator(data, n_test_users=n_test_users, sample_size=sample_size)

class Evaluator:
    """Fast evaluator with cached recommendations and optional sampling."""

    def __init__(self, data, n_test_users=5, sample_size=200):
        self.data = data.reset_index(drop=True)
        self.n_test_users = n_test_users
        self.data_sample = self.data.sample(min(sample_size, len(self.data)), random_state=42)
        self.test_users = []
        self.test_interactions = pd.DataFrame()
        self._rec_cache = {}
        self._prepare_test_users()
        self._generate_ground_truth()

    # ----------------- Test Users -----------------
    def _prepare_test_users(self):
        np.random.seed(42)
        all_keywords = self._extract_keywords()
        job_levels = self.data['job_level'].dropna().unique().tolist()
        job_types = self.data['job_type'].dropna().unique().tolist()
        locations = self.data['job_location'].dropna().unique().tolist()

        for user_id in range(self.n_test_users):
            n_skills = np.random.randint(1, min(5, len(all_keywords))) if all_keywords else 0
            self.test_users.append({
                'user_id': user_id,
                'job_level': np.random.choice(job_levels),
                'job_type': np.random.choice(job_types),
                'location': np.random.choice(locations),
                'skills': list(np.random.choice(all_keywords, n_skills, replace=False)) if n_skills > 0 else []
            })

    def _extract_keywords(self):
        keywords = []
        for col in ['job_title', 'company']:
            if col in self.data.columns:
                keywords.extend(self.data[col].dropna().str.lower().str.split().explode().tolist())
        return list(set(keywords))

    # ----------------- Ground Truth -----------------
    def _generate_ground_truth(self):
        records = []
        for user in self.test_users:
            relevant_jobs = self._find_relevant_jobs(user)
            for idx in relevant_jobs:
                records.append({'user_id': user['user_id'], 'job_idx': idx, 'relevant': 1})
        self.test_interactions = pd.DataFrame(records)

    def _find_relevant_jobs(self, user):
        relevant = []
        for idx, job in self.data_sample.iterrows():
            score = 0
            if job.get('job_level') == user.get('job_level'): score += 2
            if job.get('job_type') == user.get('job_type'): score += 2
            if user.get('location') and pd.notna(job.get('job_location')):
                if user['location'].lower() in job.get('job_location','').lower():
                    score += 1
            if user.get('skills'):
                job_text = f"{job.get('job_title','')} {job.get('company','')}".lower()
                score += sum(1 for kw in user['skills'] if kw.lower() in job_text)
            if score >= 1:  # ensure at least one match
                relevant.append(idx)
        return relevant[:20]

    # ----------------- Cached Recommendations -----------------
    def _get_recs(self, recommender, user, k=10):
        uid = user['user_id']
        if uid in self._rec_cache:
            return self._rec_cache[uid]
        recs = recommender.recommend(user, n_recommendations=k)
        self._rec_cache[uid] = recs
        return recs

    # ----------------- Metrics -----------------
    def evaluate_precision_recall(self, recommender, k=10):
        precisions, recalls = [], []
        for user in self.test_users:
            user_truth = self.test_interactions[self.test_interactions['user_id']==user['user_id']]['job_idx'].tolist()
            if not user_truth:
                continue
            recs = self._get_recs(recommender, user, k)
            if recs is None or recs.empty:
                precisions.append(0)
                recalls.append(0)
                continue
            recommended_jobs = recs['job_idx'].tolist()
            relevant_recommended = set(recommended_jobs) & set(user_truth)
            precisions.append(len(relevant_recommended)/len(recommended_jobs) if recommended_jobs else 0)
            recalls.append(len(relevant_recommended)/len(user_truth) if user_truth else 0)
        return np.mean(precisions) if precisions else 0, np.mean(recalls) if recalls else 0

    def evaluate_diversity(self, recommender, k=10):
        all_recs = []
        for user in self.test_users:
            recs = self._get_recs(recommender, user, k)
            if recs is not None:
                all_recs.extend(recs['job_idx'].tolist())
        return len(set(all_recs))/len(all_recs) if all_recs else 0

    def evaluate_coverage(self, recommender, k=10):
        items = set()
        for user in self.test_users:
            recs = self._get_recs(recommender, user, k)
            if recs is not None:
                items.update(recs['job_idx'].tolist())
        return len(items)/len(self.data) if len(self.data) > 0 else 0

    # ----------------- Evaluate all systems -----------------
    def evaluate_all_systems(self, cb_filter, cf_filter, kb_filter, hybrid, k=10):
        results = {}
        systems = {'Content-Based': cb_filter, 'Collaborative': cf_filter,
                   'Knowledge-Based': kb_filter, 'Hybrid': hybrid}
        for name, rec in systems.items():
            precision, recall = self.evaluate_precision_recall(rec, k)
            f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
            diversity = self.evaluate_diversity(rec, k)
            coverage = self.evaluate_coverage(rec, k)
            results[name] = {'precision': precision, 'recall': recall, 'f1_score': f1,
                             'diversity': diversity, 'coverage': coverage}
        return results

# ------------------ Streamlit Integration ------------------
def run_evaluation(data, cb_filter, cf_filter, kb_filter, hybrid):
    with st.spinner("Evaluating recommendation systems... ⏳"):
        evaluator = get_evaluator(data)
        results = evaluator.evaluate_all_systems(cb_filter, cf_filter, kb_filter, hybrid, k=10)
    st.success("✅ Evaluation Complete!")
    st.dataframe(pd.DataFrame(results).T)
