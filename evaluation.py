import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# ------------------ Cached Evaluator ------------------
@st.cache_data(show_spinner=False)
def get_evaluator(data, n_test_users=5, sample_size=200):
    return Evaluator(data, n_test_users=n_test_users, sample_size=sample_size)

class Evaluator:
    """Robust evaluator for content, collaborative, knowledge-based, and hybrid recommenders."""

    def __init__(self, data, n_test_users=5, sample_size=200):
        self.data = data.reset_index(drop=True)
        self.n_test_users = n_test_users
        self.sample_size = min(sample_size, len(self.data))
        self.data_sample = self.data.sample(self.sample_size, random_state=42) if self.sample_size > 0 else self.data
        self.test_users = []
        self.test_interactions = pd.DataFrame()
        self._rec_cache = {}
        self.location_col = 'job_location' if 'job_location' in self.data.columns else 'location' if 'location' in self.data.columns else None
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
        self.city_abbrev = {
            'NYC': 'New York', 'LA': 'Los Angeles', 'SF': 'San Francisco', 'CHI': 'Chicago',
            'BOS': 'Boston', 'DC': 'Washington', 'ATL': 'Atlanta', 'MIA': 'Miami',
            'SEA': 'Seattle', 'DEN': 'Denver', 'PHX': 'Phoenix', 'HOU': 'Houston',
            'DAL': 'Dallas', 'PHL': 'Philadelphia', 'DET': 'Detroit', 'MIN': 'Minneapolis',
            'STL': 'St. Louis', 'CLT': 'Charlotte', 'LAS': 'Las Vegas', 'ORL': 'Orlando'
        }
        if self.sample_size == 0:
            st.error("Input dataset is empty; cannot initialize evaluator")
            return
        self._prepare_test_users()
        self._generate_ground_truth()

    def _normalize_location(self, loc):
        """Normalize location abbreviations"""
        if pd.isna(loc) or not loc:
            return ""
        loc = str(loc).strip().upper()
        if loc in self.state_abbrev:
            return self.state_abbrev[loc].lower()
        if loc in self.city_abbrev:
            return self.city_abbrev[loc].lower()
        return loc.lower()

    def _match_locations(self, query, job_location, threshold=40):
        """Fuzzy match locations with state-level fallback"""
        if not query or not job_location:
            return False
        query_norm = self._normalize_location(query)
        job_norm = self._normalize_location(job_location)
        score = difflib.SequenceMatcher(None, query_norm, job_norm).ratio() * 100
        state_query = self.state_abbrev.get(query_norm.upper(), query_norm).lower()
        state_job = self.state_abbrev.get(job_norm.upper(), job_norm).lower()
        state_score = difflib.SequenceMatcher(None, state_query, state_job).ratio() * 100
        return score >= threshold or state_score >= 70

    # ----------------- Test Users -----------------
    def _prepare_test_users(self):
        """Generate synthetic test users"""
        np.random.seed(42)
        all_keywords = self._extract_keywords()
        job_levels = self.data['job_level'].dropna().unique().tolist() or ['Any']
        job_types = self.data['job_type'].dropna().unique().tolist() or ['Any']
        locations = self.data[self.location_col].dropna().unique().tolist() if self.location_col else ['Any']

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
        """Extract keywords from job_title, company, and job_skills"""
        keywords = []
        for col in ['job_title', 'company']:
            if col in self.data.columns:
                keywords.extend(self.data[col].dropna().str.lower().str.split().explode().tolist())
        if 'job_skills' in self.data.columns:
            for skills in self.data['job_skills'].dropna():
                if isinstance(skills, list):
                    keywords.extend([str(s).strip().lower() for s in skills if str(s).strip()])
                else:
                    keywords.extend([s.strip().lower() for s in str(skills).split(',') if s.strip()])
        return list(set(keywords))

    # ----------------- Ground Truth -----------------
    def _generate_ground_truth(self):
        """Generate synthetic ground truth interactions"""
        records = []
        for user in self.test_users:
            relevant_jobs = self._find_relevant_jobs(user)
            if not relevant_jobs:
                st.warning(f"No relevant jobs found for user {user['user_id']} with profile: {user}")
            for idx in relevant_jobs:
                records.append({'user_id': user['user_id'], 'job_idx': idx, 'relevant': 1})
        self.test_interactions = pd.DataFrame(records)
        if self.test_interactions.empty:
            st.error("No ground truth interactions generated; evaluation may be unreliable")

    def _find_relevant_jobs(self, user):
        """Find relevant jobs for a user based on profile"""
        relevant = []
        for idx, job in self.data_sample.iterrows():
            score = 0
            if job.get('job_level') == user.get('job_level'):
                score += 2
            if job.get('job_type') == user.get('job_type'):
                score += 2
            if user.get('location') and pd.notna(job.get(self.location_col)) and self._match_locations(user['location'], job.get(self.location_col)):
                score += 3  # Increased weight for location
            if user.get('skills'):
                job_text = f"{job.get('job_title', '')} {job.get('company', '')}".lower()
                job_skills = job.get('job_skills', []) if isinstance(job.get('job_skills'), list) else str(job.get('job_skills', '')).split(',')
                job_skills = [str(s).strip().lower() for s in job_skills if str(s).strip()]
                score += sum(1 for kw in user['skills'] if kw.lower() in job_text or kw.lower() in job_skills)
            if score >= 1:
                relevant.append(idx)
        return relevant[:20]

    # ----------------- Cached Recommendations -----------------
    def _get_recs(self, recommender, user, k=10):
        """Get cached recommendations for a user"""
        uid = user['user_id']
        if uid in self._rec_cache:
            return self._rec_cache[uid]
        try:
            recs = recommender.recommend(user, n_recommendations=k)
            if recs is None or recs.empty:
                st.warning(f"Recommender {recommender.__class__.__name__} returned no recommendations for user {uid}: {user}")
                recs = pd.DataFrame({'job_idx': [], 'score': []})
            elif 'job_idx' not in recs.columns:
                recs = recs.reset_index().rename(columns={'index': 'job_idx'})
            self._rec_cache[uid] = recs
            return recs
        except Exception as e:
            st.error(f"Error in {recommender.__class__.__name__} for user {uid}: {str(e)}")
            return pd.DataFrame({'job_idx': [], 'score': []})

    # ----------------- Metrics -----------------
    def evaluate_precision_recall(self, recommender, k=10):
        """Calculate precision and recall for a recommender"""
        precisions, recalls = [], []
        for user in self.test_users:
            user_truth = self.test_interactions[self.test_interactions['user_id'] == user['user_id']]['job_idx'].tolist()
            if not user_truth:
                precisions.append(0)
                recalls.append(0)
                continue
            recs = self._get_recs(recommender, user, k)
            recommended_jobs = recs['job_idx'].tolist() if not recs.empty else []
            relevant_recommended = set(recommended_jobs) & set(user_truth)
            precisions.append(len(relevant_recommended) / len(recommended_jobs) if recommended_jobs else 0)
            recalls.append(len(relevant_recommended) / len(user_truth) if user_truth else 0)
        precision = np.mean(precisions) if precisions else 0
        recall = np.mean(recalls) if recalls else 0
        return precision, recall

    def evaluate_diversity(self, recommender, k=10):
        """Calculate diversity as unique items recommended"""
        all_recs = []
        for user in self.test_users:
            recs = self._get_recs(recommender, user, k)
            if not recs.empty:
                all_recs.extend(recs['job_idx'].tolist())
        return len(set(all_recs)) / len(all_recs) if all_recs else 0

    def evaluate_coverage(self, recommender, k=10):
        """Calculate coverage as fraction of items recommended"""
        items = set()
        for user in self.test_users:
            recs = self._get_recs(recommender, user, k)
            if not recs.empty:
                items.update(recs['job_idx'].tolist())
        return len(items) / len(self.data) if len(self.data) > 0 else 0

    # ----------------- Evaluate all systems -----------------
    def evaluate_all_systems(self, cb_filter, cf_filter, kb_filter, hybrid, k=10):
        """Evaluate all recommendation systems"""
        results = {}
        systems = {
            'Content-Based': cb_filter,
            'Collaborative': cf_filter,
            'Knowledge-Based': kb_filter,
            'Hybrid': hybrid
        }
        for name, rec in systems.items():
            try:
                precision, recall = self.evaluate_precision_recall(rec, k)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                diversity = self.evaluate_diversity(rec, k)
                coverage = self.evaluate_coverage(rec, k)
                results[name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'diversity': diversity,
                    'coverage': coverage
                }
            except Exception as e:
                st.error(f"Evaluation failed for {name}: {str(e)}")
                results[name] = {
                    'precision': 0,
                    'recall': 0,
                    'f1_score': 0,
                    'diversity': 0,
                    'coverage': 0
                }
        return results

    # ----------------- Streamlit Integration ------------------
def run_evaluation(data, cb_filter, cf_filter, kb_filter, hybrid):
    """Run evaluation and display results in Streamlit"""
    try:
        with st.spinner("Evaluating recommendation systems... ⏳"):
            evaluator = get_evaluator(data)
            results = evaluator.evaluate_all_systems(cb_filter, cf_filter, kb_filter, hybrid, k=10)
        st.success("✅ Evaluation Complete!")
        st.subheader("Evaluation Results")
        st.dataframe(pd.DataFrame(results).T.round(4))
        if all(all(v == 0 for v in metrics.values()) for metrics in results.values()):
            st.warning("All metrics are zero; recommenders may be returning empty results. Check dataset or user profiles.")
    except Exception as e:
        st.error(f"Evaluation failed: {str(e)}")
        st.info("Please ensure recommenders are properly initialized and dataset contains valid data.")