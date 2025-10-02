import pandas as pd
import numpy as np
import streamlit as st
import random

class HybridRecommender:
    """Hybrid recommender system combining content-based, collaborative, and knowledge-based filtering"""

    def __init__(self, content_filter, collaborative_filter, knowledge_filter):
        self.content_filter = content_filter
        self.collaborative_filter = collaborative_filter
        self.knowledge_filter = knowledge_filter

        self.default_weights = {
            'content': 0.4,
            'collaborative': 0.3,
            'knowledge': 0.3
        }

    def recommend(self, user_profile, n_recommendations=10, weights=None):
        if weights is None:
            weights = self.default_weights

        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        else:
            weights = self.default_weights

        st.info("Generating hybrid recommendations...")

        recommendations = {}

        # --- Content-based ---
        if weights.get('content', 0) > 0:
            try:
                cb_recs = self.content_filter.recommend(user_profile, n_recommendations*3)
                if cb_recs is not None and not cb_recs.empty:
                    cb_recs = cb_recs.reset_index(drop=True)
                    if 'job_idx' not in cb_recs.columns:
                        cb_recs['job_idx'] = cb_recs.index
                    recommendations['content'] = cb_recs
                    st.success(f"Content-based: {len(cb_recs)} recommendations")
            except Exception as e:
                st.warning(f"Content-based filtering failed: {e}")

        # --- Collaborative ---
        if weights.get('collaborative', 0) > 0:
            try:
                cf_recs = self.collaborative_filter.recommend(user_profile, n_recommendations*3)
                if cf_recs is not None and not cf_recs.empty:
                    cf_recs = cf_recs.reset_index(drop=True)
                    if 'job_idx' not in cf_recs.columns:
                        cf_recs['job_idx'] = cf_recs.index
                    recommendations['collaborative'] = cf_recs
                    st.success(f"Collaborative filtering: {len(cf_recs)} recommendations")
            except Exception as e:
                st.warning(f"Collaborative filtering failed: {e}")

        # --- Knowledge-based ---
        if weights.get('knowledge', 0) > 0:
            try:
                kb_recs = self.knowledge_filter.recommend(user_profile, n_recommendations*3)
                if kb_recs is not None and not kb_recs.empty:
                    kb_recs = kb_recs.reset_index(drop=True)
                    if 'job_idx' not in kb_recs.columns:
                        kb_recs['job_idx'] = kb_recs.index
                    recommendations['knowledge'] = kb_recs
                    st.success(f"Knowledge-based: {len(kb_recs)} recommendations")
            except Exception as e:
                st.warning(f"Knowledge-based filtering failed: {e}")

        if not recommendations:
            st.error("All recommendation systems failed")
            return None

        return self._combine_recommendations(recommendations, weights, n_recommendations)

    def _combine_recommendations(self, recommendations, weights, n_recommendations):
        """Combine recommendations from all systems using weighted averaging with randomness for diversity"""
        all_jobs = set()
        for recs in recommendations.values():
            all_jobs.update(recs['job_idx'].tolist())

        hybrid_scores = {}
        for job_idx in all_jobs:
            total_score = 0
            total_weight = 0

            for system, recs in recommendations.items():
                w = weights.get(system, 0)
                if w <= 0: 
                    continue
                match = recs[recs['job_idx'] == job_idx]
                if match.empty: 
                    continue
                score_col = {'content':'similarity_score', 'collaborative':'cf_score', 'knowledge':'kb_score'}[system]
                score = match[score_col].iloc[0]
                total_score += self._normalize(score, recs[score_col]) * w
                total_weight += w

            # Add small randomness to break ties and improve diversity
            hybrid_scores[job_idx] = (total_score/total_weight if total_weight>0 else random.random()*1e-6) + random.uniform(0, 1e-3)

        # Sort and pick top N
        sorted_jobs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        top_jobs = sorted_jobs[:n_recommendations]

        return pd.DataFrame({
            'job_idx':[j[0] for j in top_jobs],
            'hybrid_score':[j[1] for j in top_jobs]
        })

    def _normalize(self, score, score_series):
        min_s, max_s = score_series.min(), score_series.max()
        if max_s == min_s:
            return 0.5
        return (score - min_s) / (max_s - min_s)
