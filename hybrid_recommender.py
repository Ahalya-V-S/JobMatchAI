import pandas as pd
import numpy as np
import streamlit as st

class HybridRecommender:
    """Hybrid recommender system combining content-based, collaborative, and knowledge-based filtering"""
    
    def __init__(self, content_filter, collaborative_filter, knowledge_filter):
        self.content_filter = content_filter
        self.collaborative_filter = collaborative_filter
        self.knowledge_filter = knowledge_filter
        
        # Default weights for different approaches
        self.default_weights = {
            'content': 0.4,
            'collaborative': 0.3,
            'knowledge': 0.3
        }
    
    def recommend(self, user_profile, n_recommendations=10, weights=None):
        """Generate hybrid recommendations by combining all three approaches"""
        
        if weights is None:
            weights = self.default_weights
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        else:
            weights = self.default_weights
        
        st.info("Generating hybrid recommendations...")
        
        # Get recommendations from each system
        recommendations = {}
        
        # Content-based recommendations
        if weights.get('content', 0) > 0:
            try:
                cb_recs = self.content_filter.recommend(user_profile, n_recommendations * 3)
                if cb_recs is not None and len(cb_recs) > 0:
                    recommendations['content'] = cb_recs
                    st.success(f"Content-based: {len(cb_recs)} recommendations")
                else:
                    st.warning("Content-based filtering returned no recommendations")
            except Exception as e:
                st.error(f"Content-based filtering failed: {str(e)}")
        
        # Collaborative filtering recommendations
        if weights.get('collaborative', 0) > 0:
            try:
                cf_recs = self.collaborative_filter.recommend(user_profile, n_recommendations * 3)
                if cf_recs is not None and len(cf_recs) > 0:
                    recommendations['collaborative'] = cf_recs
                    st.success(f"Collaborative filtering: {len(cf_recs)} recommendations")
                else:
                    st.warning("Collaborative filtering returned no recommendations")
            except Exception as e:
                st.error(f"Collaborative filtering failed: {str(e)}")
        
        # Knowledge-based recommendations
        if weights.get('knowledge', 0) > 0:
            try:
                kb_recs = self.knowledge_filter.recommend(user_profile, n_recommendations * 3)
                if kb_recs is not None and len(kb_recs) > 0:
                    recommendations['knowledge'] = kb_recs
                    st.success(f"Knowledge-based: {len(kb_recs)} recommendations")
                else:
                    st.warning("Knowledge-based filtering returned no recommendations")
            except Exception as e:
                st.error(f"Knowledge-based filtering failed: {str(e)}")
        
        if not recommendations:
            st.error("All recommendation systems failed to generate recommendations")
            return None
        
        # Combine recommendations using weighted approach
        final_recommendations = self._combine_recommendations(
            recommendations, weights, n_recommendations
        )
        
        return final_recommendations
    
    def _combine_recommendations(self, recommendations, weights, n_recommendations):
        """Combine recommendations from different systems using weighted averaging"""
        
        # Collect all unique job indices
        all_jobs = set()
        for system_recs in recommendations.values():
            all_jobs.update(system_recs['job_idx'].tolist())
        
        # Calculate hybrid scores for each job
        hybrid_scores = {}
        
        for job_idx in all_jobs:
            total_score = 0
            total_weight = 0
            
            # Content-based score
            if 'content' in recommendations and weights.get('content', 0) > 0:
                cb_recs = recommendations['content']
                cb_match = cb_recs[cb_recs['job_idx'] == job_idx]
                if not cb_match.empty:
                    cb_score = cb_match['similarity_score'].iloc[0]
                    total_score += cb_score * weights['content']
                    total_weight += weights['content']
            
            # Collaborative filtering score
            if 'collaborative' in recommendations and weights.get('collaborative', 0) > 0:
                cf_recs = recommendations['collaborative']
                cf_match = cf_recs[cf_recs['job_idx'] == job_idx]
                if not cf_match.empty:
                    cf_score = cf_match['cf_score'].iloc[0]
                    # Normalize CF score to 0-1 range
                    cf_score_norm = self._normalize_score(cf_score, cf_recs['cf_score'])
                    total_score += cf_score_norm * weights['collaborative']
                    total_weight += weights['collaborative']
            
            # Knowledge-based score
            if 'knowledge' in recommendations and weights.get('knowledge', 0) > 0:
                kb_recs = recommendations['knowledge']
                kb_match = kb_recs[kb_recs['job_idx'] == job_idx]
                if not kb_match.empty:
                    kb_score = kb_match['kb_score'].iloc[0]
                    total_score += kb_score * weights['knowledge']
                    total_weight += weights['knowledge']
            
            # Calculate final hybrid score
            if total_weight > 0:
                hybrid_scores[job_idx] = total_score / total_weight
            else:
                hybrid_scores[job_idx] = 0
        
        # Sort by hybrid score and return top recommendations
        sorted_jobs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        top_jobs = sorted_jobs[:n_recommendations]
        
        # Create final recommendations DataFrame
        final_recs = pd.DataFrame({
            'job_idx': [job[0] for job in top_jobs],
            'hybrid_score': [job[1] for job in top_jobs]
        })
        
        return final_recs
    
    def _normalize_score(self, score, score_series):
        """Normalize a score to 0-1 range based on the score series statistics"""
        min_score = score_series.min()
        max_score = score_series.max()
        
        if max_score == min_score:
            return 0.5  # If all scores are the same
        
        return (score - min_score) / (max_score - min_score)
    
    def recommend_with_switching(self, user_profile, n_recommendations=10):
        """Alternative hybrid approach using switching strategy"""
        
        # Determine which system to use based on user profile characteristics
        primary_system = self._select_primary_system(user_profile)
        
        st.info(f"Using {primary_system} as primary recommendation system")
        
        # Get recommendations from primary system
        if primary_system == 'content':
            primary_recs = self.content_filter.recommend(user_profile, n_recommendations)
        elif primary_system == 'collaborative':
            primary_recs = self.collaborative_filter.recommend(user_profile, n_recommendations)
        else:  # knowledge
            primary_recs = self.knowledge_filter.recommend(user_profile, n_recommendations)
        
        # If primary system fails or returns insufficient results, use fallback
        if primary_recs is None or len(primary_recs) < n_recommendations:
            st.warning(f"{primary_system} system insufficient, using fallback systems")
            
            # Try other systems as fallback
            fallback_systems = ['content', 'collaborative', 'knowledge']
            fallback_systems.remove(primary_system)
            
            for fallback in fallback_systems:
                try:
                    if fallback == 'content':
                        fallback_recs = self.content_filter.recommend(user_profile, n_recommendations)
                    elif fallback == 'collaborative':
                        fallback_recs = self.collaborative_filter.recommend(user_profile, n_recommendations)
                    else:
                        fallback_recs = self.knowledge_filter.recommend(user_profile, n_recommendations)
                    
                    if fallback_recs is not None and len(fallback_recs) > 0:
                        if primary_recs is not None:
                            # Combine primary and fallback recommendations
                            combined_recs = self._merge_recommendations(primary_recs, fallback_recs, n_recommendations)
                            return combined_recs
                        else:
                            return fallback_recs
                except Exception as e:
                    st.warning(f"Fallback system {fallback} failed: {str(e)}")
                    continue
        
        return primary_recs
    
    def _select_primary_system(self, user_profile):
        """Select the most appropriate primary system based on user profile"""
        
        # If user has many specific constraints, use knowledge-based
        constraint_count = 0
        if user_profile.get('location') and user_profile['location'] != 'Any':
            constraint_count += 1
        if user_profile.get('min_salary'):
            constraint_count += 1
        if user_profile.get('max_salary'):
            constraint_count += 1
        if user_profile.get('job_type') and user_profile['job_type'] != 'Any':
            constraint_count += 1
        if user_profile.get('experience_level'):
            constraint_count += 1
        
        if constraint_count >= 3:
            return 'knowledge'
        
        # If user has detailed skills, use content-based
        if user_profile.get('skills') and len(user_profile['skills']) >= 3:
            return 'content'
        
        # Otherwise, use collaborative filtering
        return 'collaborative'
    
    def _merge_recommendations(self, primary_recs, fallback_recs, n_recommendations):
        """Merge recommendations from primary and fallback systems"""
        
        # Get unique job indices from both recommendations
        primary_jobs = set(primary_recs['job_idx'].tolist())
        fallback_jobs = set(fallback_recs['job_idx'].tolist())
        
        # Start with primary recommendations
        merged_jobs = []
        
        # Add primary recommendations first
        for idx, row in primary_recs.iterrows():
            merged_jobs.append({
                'job_idx': row['job_idx'],
                'score': row.iloc[1],  # Second column should be the score
                'source': 'primary'
            })
        
        # Add fallback recommendations that aren't already included
        for idx, row in fallback_recs.iterrows():
            if row['job_idx'] not in primary_jobs and len(merged_jobs) < n_recommendations:
                merged_jobs.append({
                    'job_idx': row['job_idx'],
                    'score': row.iloc[1],  # Second column should be the score
                    'source': 'fallback'
                })
        
        # Create final DataFrame
        merged_df = pd.DataFrame(merged_jobs[:n_recommendations])
        
        # Rename score column appropriately
        if not merged_df.empty:
            merged_df = merged_df[['job_idx', 'score']].rename(columns={'score': 'hybrid_score'})
        
        return merged_df
    
    def recommend_with_mixed_strategy(self, user_profile, n_recommendations=10):
        """Mixed strategy: combine different approaches for different types of recommendations"""
        
        # Allocate recommendations across different systems
        n_content = max(1, int(n_recommendations * 0.4))
        n_collaborative = max(1, int(n_recommendations * 0.3))
        n_knowledge = max(1, int(n_recommendations * 0.3))
        
        all_recommendations = []
        
        # Get content-based recommendations
        try:
            cb_recs = self.content_filter.recommend(user_profile, n_content * 2)
            if cb_recs is not None and len(cb_recs) > 0:
                for idx, row in cb_recs.head(n_content).iterrows():
                    all_recommendations.append({
                        'job_idx': row['job_idx'],
                        'score': row['similarity_score'],
                        'system': 'content'
                    })
        except Exception as e:
            st.warning(f"Content-based recommendations failed: {str(e)}")
        
        # Get collaborative filtering recommendations
        try:
            cf_recs = self.collaborative_filter.recommend(user_profile, n_collaborative * 2)
            if cf_recs is not None and len(cf_recs) > 0:
                # Avoid duplicates
                existing_jobs = {rec['job_idx'] for rec in all_recommendations}
                for idx, row in cf_recs.iterrows():
                    if row['job_idx'] not in existing_jobs and len([r for r in all_recommendations if r['system'] == 'collaborative']) < n_collaborative:
                        cf_score_norm = self._normalize_score(row['cf_score'], cf_recs['cf_score'])
                        all_recommendations.append({
                            'job_idx': row['job_idx'],
                            'score': cf_score_norm,
                            'system': 'collaborative'
                        })
                        existing_jobs.add(row['job_idx'])
        except Exception as e:
            st.warning(f"Collaborative filtering recommendations failed: {str(e)}")
        
        # Get knowledge-based recommendations
        try:
            kb_recs = self.knowledge_filter.recommend(user_profile, n_knowledge * 2)
            if kb_recs is not None and len(kb_recs) > 0:
                # Avoid duplicates
                existing_jobs = {rec['job_idx'] for rec in all_recommendations}
                for idx, row in kb_recs.iterrows():
                    if row['job_idx'] not in existing_jobs and len([r for r in all_recommendations if r['system'] == 'knowledge']) < n_knowledge:
                        all_recommendations.append({
                            'job_idx': row['job_idx'],
                            'score': row['kb_score'],
                            'system': 'knowledge'
                        })
                        existing_jobs.add(row['job_idx'])
        except Exception as e:
            st.warning(f"Knowledge-based recommendations failed: {str(e)}")
        
        # Sort by score and return top recommendations
        all_recommendations.sort(key=lambda x: x['score'], reverse=True)
        final_recs = all_recommendations[:n_recommendations]
        
        # Create DataFrame
        final_df = pd.DataFrame({
            'job_idx': [rec['job_idx'] for rec in final_recs],
            'hybrid_score': [rec['score'] for rec in final_recs],
            'primary_system': [rec['system'] for rec in final_recs]
        })
        
        return final_df
    
    def explain_hybrid_recommendation(self, job_idx, user_profile, weights=None):
        """Provide explanation for a hybrid recommendation"""
        
        if weights is None:
            weights = self.default_weights
        
        explanations = {
            'job_idx': job_idx,
            'components': {}
        }
        
        # Get job data
        if hasattr(self.content_filter, 'data'):
            job_data = self.content_filter.data.iloc[job_idx]
        elif hasattr(self.collaborative_filter, 'data'):
            job_data = self.collaborative_filter.data.iloc[job_idx]
        elif hasattr(self.knowledge_filter, 'data'):
            job_data = self.knowledge_filter.data.iloc[job_idx]
        else:
            return None
        
        # Content-based explanation
        if weights.get('content', 0) > 0:
            # Get similar jobs and features that contributed
            explanations['components']['content'] = {
                'weight': weights['content'],
                'reasoning': 'Recommended based on job description and requirements similarity to your profile'
            }
        
        # Collaborative explanation
        if weights.get('collaborative', 0) > 0:
            explanations['components']['collaborative'] = {
                'weight': weights['collaborative'],
                'reasoning': 'Recommended because users with similar preferences also showed interest in this job'
            }
        
        # Knowledge-based explanation
        if weights.get('knowledge', 0) > 0:
            kb_explanation = self.knowledge_filter.explain_recommendation(job_data, user_profile)
            explanations['components']['knowledge'] = {
                'weight': weights['knowledge'],
                'reasoning': 'Recommended based on how well it matches your specified constraints',
                'detailed_factors': kb_explanation
            }
        
        return explanations
    
    def get_system_statistics(self):
        """Get statistics about the performance and coverage of each system"""
        stats = {
            'content_based': {
                'available': self.content_filter is not None and hasattr(self.content_filter, 'tfidf_matrix'),
                'features': 'TF-IDF vectorization of job descriptions and skills'
            },
            'collaborative': {
                'available': self.collaborative_filter is not None and hasattr(self.collaborative_filter, 'svd_model'),
                'features': 'Matrix factorization on user-job interactions'
            },
            'knowledge_based': {
                'available': self.knowledge_filter is not None and hasattr(self.knowledge_filter, 'rules'),
                'features': f'{len(self.knowledge_filter.rules) if hasattr(self.knowledge_filter, "rules") else 0} domain-specific rules'
            }
        }
        
        return stats
