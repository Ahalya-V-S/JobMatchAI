import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class Evaluator:
    """Evaluation metrics for recommendation systems"""
    
    def __init__(self, data):
        self.data = data
        self.test_users = None
        self.test_interactions = None
        self._prepare_test_data()
    
    def _prepare_test_data(self):
        """Prepare test data for evaluation"""
        np.random.seed(42)
        
        # Create synthetic test users with known preferences
        n_test_users = 50
        test_users = []
        
        for user_id in range(n_test_users):
            # Create diverse user profiles for testing
            experience_levels = ['Entry level', 'Associate', 'Mid-Senior level', 'Director', 'Executive']
            job_categories = ['Engineering', 'Management', 'Sales', 'Marketing', 'Data', 'Design', 'HR', 'Finance']
            
            user_profile = {
                'user_id': user_id,
                'experience_level': np.random.choice(experience_levels),
                'preferred_category': np.random.choice(job_categories),
                'min_salary': np.random.randint(40000, 80000),
                'max_salary': np.random.randint(100000, 200000),
                'skills': self._generate_random_skills(),
                'location': self._get_random_location()
            }
            test_users.append(user_profile)
        
        self.test_users = test_users
        
        # Generate ground truth interactions
        self._generate_ground_truth()
    
    def _generate_random_skills(self):
        """Generate random skills from the dataset"""
        all_skills = []
        if 'skills' in self.data.columns:
            for skills_str in self.data['skills'].dropna().sample(min(100, len(self.data))):
                if isinstance(skills_str, str):
                    skills = [skill.strip() for skill in skills_str.split(',')]
                    all_skills.extend(skills)
        
        unique_skills = list(set(all_skills))
        if len(unique_skills) > 0:
            n_skills = np.random.randint(2, min(6, len(unique_skills)))
            return np.random.choice(unique_skills, size=n_skills, replace=False).tolist()
        return []
    
    def _get_random_location(self):
        """Get random location from the dataset"""
        if 'location' in self.data.columns:
            locations = self.data['location'].dropna().unique()
            if len(locations) > 0:
                return np.random.choice(locations)
        return 'Any'
    
    def _generate_ground_truth(self):
        """Generate ground truth for evaluation"""
        ground_truth = []
        
        for user in self.test_users:
            # Find jobs that should be relevant to this user
            relevant_jobs = self._find_relevant_jobs(user)
            
            # Create interactions with relevant jobs
            for job_idx in relevant_jobs:
                ground_truth.append({
                    'user_id': user['user_id'],
                    'job_idx': job_idx,
                    'relevant': 1
                })
        
        self.test_interactions = pd.DataFrame(ground_truth)
    
    def _find_relevant_jobs(self, user_profile):
        """Find jobs that should be relevant to a user based on their profile"""
        relevant_jobs = []
        
        for idx in range(len(self.data)):
            job = self.data.iloc[idx]
            relevance_score = 0
            
            # Experience level match
            if 'experience_level' in self.data.columns:
                if job.get('experience_level') == user_profile['experience_level']:
                    relevance_score += 3
                elif self._is_experience_compatible(job.get('experience_level'), user_profile['experience_level']):
                    relevance_score += 1
            
            # Job category match
            if 'job_category' in self.data.columns:
                if job.get('job_category') == user_profile['preferred_category']:
                    relevance_score += 3
            
            # Salary match
            if 'salary_min' in self.data.columns and 'salary_max' in self.data.columns:
                job_min = job.get('salary_min')
                job_max = job.get('salary_max')
                if pd.notna(job_min) and pd.notna(job_max):
                    if (job_min <= user_profile['max_salary'] and 
                        job_max >= user_profile['min_salary']):
                        relevance_score += 2
            
            # Skills match
            if 'skills' in self.data.columns and user_profile.get('skills'):
                job_skills_str = job.get('skills', '')
                if isinstance(job_skills_str, str):
                    job_skills = set(skill.strip().lower() for skill in job_skills_str.split(','))
                    user_skills = set(skill.lower() for skill in user_profile['skills'])
                    if job_skills & user_skills:  # Intersection
                        relevance_score += len(job_skills & user_skills)
            
            # Location match
            if 'location' in self.data.columns:
                if (user_profile['location'] != 'Any' and 
                    user_profile['location'].lower() in str(job.get('location', '')).lower()):
                    relevance_score += 1
            
            # If relevance score is high enough, consider it relevant
            if relevance_score >= 4:
                relevant_jobs.append(idx)
        
        # Limit to top relevant jobs to avoid too many positives
        return relevant_jobs[:20]
    
    def _is_experience_compatible(self, job_exp, user_exp):
        """Check if experience levels are compatible"""
        exp_hierarchy = {
            'Entry level': 1,
            'Associate': 2,
            'Mid-Senior level': 3,
            'Director': 4,
            'Executive': 5
        }
        
        job_level = exp_hierarchy.get(job_exp, 0)
        user_level = exp_hierarchy.get(user_exp, 0)
        
        return abs(job_level - user_level) <= 1
    
    def evaluate_precision_recall(self, recommender, k=10):
        """Evaluate precision and recall at k"""
        precisions = []
        recalls = []
        
        for user in self.test_users:
            user_id = user['user_id']
            
            # Get ground truth for this user
            user_truth = self.test_interactions[
                self.test_interactions['user_id'] == user_id
            ]['job_idx'].tolist()
            
            if not user_truth:
                continue
            
            # Get recommendations
            try:
                recommendations = recommender.recommend(user, n_recommendations=k)
                if recommendations is None or len(recommendations) == 0:
                    precisions.append(0)
                    recalls.append(0)
                    continue
                
                recommended_jobs = recommendations['job_idx'].tolist()
                
                # Calculate precision and recall
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
        """Evaluate diversity of recommendations"""
        all_recommendations = []
        
        for user in self.test_users[:20]:  # Limit for performance
            try:
                recommendations = recommender.recommend(user, n_recommendations=k)
                if recommendations is not None and len(recommendations) > 0:
                    all_recommendations.extend(recommendations['job_idx'].tolist())
            except Exception:
                continue
        
        if not all_recommendations:
            return 0
        
        # Calculate diversity as the ratio of unique recommendations to total recommendations
        unique_recommendations = len(set(all_recommendations))
        total_recommendations = len(all_recommendations)
        
        diversity = unique_recommendations / total_recommendations if total_recommendations > 0 else 0
        
        return diversity
    
    def evaluate_coverage(self, recommender, k=10):
        """Evaluate catalog coverage"""
        recommended_items = set()
        
        for user in self.test_users[:20]:  # Limit for performance
            try:
                recommendations = recommender.recommend(user, n_recommendations=k)
                if recommendations is not None and len(recommendations) > 0:
                    recommended_items.update(recommendations['job_idx'].tolist())
            except Exception:
                continue
        
        total_items = len(self.data)
        coverage = len(recommended_items) / total_items if total_items > 0 else 0
        
        return coverage
    
    def evaluate_novelty(self, recommender, k=10):
        """Evaluate novelty of recommendations (how uncommon the recommended items are)"""
        # Calculate item popularity
        item_popularity = {}
        
        # Use synthetic interactions or job view counts if available
        if hasattr(self, 'test_interactions') and self.test_interactions is not None:
            popularity_counts = self.test_interactions['job_idx'].value_counts()
            total_interactions = len(self.test_interactions)
            
            for job_idx in range(len(self.data)):
                item_popularity[job_idx] = popularity_counts.get(job_idx, 0) / total_interactions
        else:
            # Fallback: assume uniform popularity
            for job_idx in range(len(self.data)):
                item_popularity[job_idx] = 1.0 / len(self.data)
        
        novelty_scores = []
        
        for user in self.test_users[:20]:  # Limit for performance
            try:
                recommendations = recommender.recommend(user, n_recommendations=k)
                if recommendations is not None and len(recommendations) > 0:
                    user_novelty = 0
                    for job_idx in recommendations['job_idx']:
                        # Novelty is the negative log of popularity
                        popularity = item_popularity.get(job_idx, 1.0 / len(self.data))
                        novelty = -np.log2(popularity) if popularity > 0 else 0
                        user_novelty += novelty
                    
                    avg_novelty = user_novelty / len(recommendations)
                    novelty_scores.append(avg_novelty)
            except Exception:
                continue
        
        return np.mean(novelty_scores) if novelty_scores else 0
    
    def evaluate_serendipity(self, recommender, k=10):
        """Evaluate serendipity (unexpected but relevant recommendations)"""
        serendipity_scores = []
        
        for user in self.test_users[:10]:  # Limit for performance
            try:
                recommendations = recommender.recommend(user, n_recommendations=k)
                if recommendations is None or len(recommendations) == 0:
                    continue
                
                user_profile_vector = self._create_user_profile_vector(user)
                serendipity_score = 0
                
                for job_idx in recommendations['job_idx']:
                    job = self.data.iloc[job_idx]
                    job_vector = self._create_job_vector(job)
                    
                    # Calculate unexpectedness (low similarity to user profile)
                    similarity = cosine_similarity([user_profile_vector], [job_vector])[0][0]
                    unexpectedness = 1 - similarity
                    
                    # Check if the job is actually relevant (simplified relevance check)
                    relevance = self._calculate_simple_relevance(job, user)
                    
                    # Serendipity = unexpectedness * relevance
                    serendipity_score += unexpectedness * relevance
                
                avg_serendipity = serendipity_score / len(recommendations)
                serendipity_scores.append(avg_serendipity)
                
            except Exception:
                continue
        
        return np.mean(serendipity_scores) if serendipity_scores else 0
    
    def _create_user_profile_vector(self, user):
        """Create a simple numerical vector representation of user profile"""
        vector = []
        
        # Experience level (0-5)
        exp_mapping = {'Entry level': 1, 'Associate': 2, 'Mid-Senior level': 3, 'Director': 4, 'Executive': 5}
        vector.append(exp_mapping.get(user.get('experience_level'), 0))
        
        # Salary preferences (normalized)
        vector.append(user.get('min_salary', 50000) / 100000)
        vector.append(user.get('max_salary', 150000) / 100000)
        
        # Number of skills
        vector.append(len(user.get('skills', [])))
        
        # Category preference (simplified encoding)
        categories = ['Engineering', 'Management', 'Sales', 'Marketing', 'Data', 'Design', 'HR', 'Finance']
        category_vector = [1 if cat == user.get('preferred_category') else 0 for cat in categories]
        vector.extend(category_vector)
        
        return vector
    
    def _create_job_vector(self, job):
        """Create a simple numerical vector representation of job"""
        vector = []
        
        # Experience level
        exp_mapping = {'Entry level': 1, 'Associate': 2, 'Mid-Senior level': 3, 'Director': 4, 'Executive': 5}
        vector.append(exp_mapping.get(job.get('experience_level'), 0))
        
        # Salary (normalized)
        salary_min = job.get('salary_min', 50000) if pd.notna(job.get('salary_min')) else 50000
        salary_max = job.get('salary_max', 150000) if pd.notna(job.get('salary_max')) else 150000
        vector.append(salary_min / 100000)
        vector.append(salary_max / 100000)
        
        # Skills count
        skills_str = job.get('skills', '')
        skills_count = len(skills_str.split(',')) if isinstance(skills_str, str) and skills_str else 0
        vector.append(skills_count)
        
        # Job category
        categories = ['Engineering', 'Management', 'Sales', 'Marketing', 'Data', 'Design', 'HR', 'Finance']
        job_category = job.get('job_category', 'Other')
        category_vector = [1 if cat == job_category else 0 for cat in categories]
        vector.extend(category_vector)
        
        return vector
    
    def _calculate_simple_relevance(self, job, user):
        """Calculate simple relevance score between job and user"""
        relevance = 0
        
        # Experience match
        if job.get('experience_level') == user.get('experience_level'):
            relevance += 0.3
        
        # Category match
        if job.get('job_category') == user.get('preferred_category'):
            relevance += 0.3
        
        # Salary compatibility
        job_min = job.get('salary_min')
        job_max = job.get('salary_max')
        if (pd.notna(job_min) and pd.notna(job_max) and
            job_min <= user.get('max_salary', float('inf')) and
            job_max >= user.get('min_salary', 0)):
            relevance += 0.2
        
        # Skills overlap
        if 'skills' in job and user.get('skills'):
            job_skills = set(skill.strip().lower() for skill in str(job['skills']).split(','))
            user_skills = set(skill.lower() for skill in user['skills'])
            overlap = len(job_skills & user_skills)
            relevance += min(0.2, overlap * 0.05)
        
        return min(1.0, relevance)
    
    def evaluate_all_systems(self, content_filter, collaborative_filter, knowledge_filter, hybrid_recommender):
        """Evaluate all recommendation systems"""
        st.info("Starting comprehensive evaluation...")
        
        systems = {
            'Content-Based': content_filter,
            'Collaborative': collaborative_filter, 
            'Knowledge-Based': knowledge_filter,
            'Hybrid': hybrid_recommender
        }
        
        results = {}
        
        for system_name, system in systems.items():
            st.info(f"Evaluating {system_name} system...")
            
            try:
                # Precision and Recall
                precision, recall = self.evaluate_precision_recall(system, k=10)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # Diversity
                diversity = self.evaluate_diversity(system, k=10)
                
                # Coverage
                coverage = self.evaluate_coverage(system, k=10)
                
                # Novelty
                novelty = self.evaluate_novelty(system, k=10)
                
                # Serendipity
                serendipity = self.evaluate_serendipity(system, k=10)
                
                results[system_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'diversity': diversity,
                    'coverage': coverage,
                    'novelty': novelty,
                    'serendipity': serendipity
                }
                
                st.success(f"{system_name} evaluation completed")
                
            except Exception as e:
                st.error(f"Error evaluating {system_name}: {str(e)}")
                results[system_name] = {
                    'precision': 0,
                    'recall': 0,
                    'f1_score': 0,
                    'diversity': 0,
                    'coverage': 0,
                    'novelty': 0,
                    'serendipity': 0
                }
        
        return results
    
    def create_evaluation_report(self, results):
        """Create a comprehensive evaluation report"""
        report = {
            'summary': {},
            'detailed_results': results,
            'recommendations': []
        }
        
        # Calculate summary statistics
        metrics = ['precision', 'recall', 'f1_score', 'diversity', 'coverage', 'novelty', 'serendipity']
        
        for metric in metrics:
            metric_values = [results[system].get(metric, 0) for system in results.keys()]
            report['summary'][metric] = {
                'best_system': max(results.keys(), key=lambda x: results[x].get(metric, 0)),
                'best_score': max(metric_values),
                'average_score': np.mean(metric_values),
                'worst_score': min(metric_values)
            }
        
        # Generate recommendations
        best_precision = report['summary']['precision']['best_system']
        best_diversity = report['summary']['diversity']['best_system']
        best_coverage = report['summary']['coverage']['best_system']
        
        if best_precision == 'Hybrid':
            report['recommendations'].append("The hybrid system shows the best precision, indicating effective combination of approaches.")
        
        if best_diversity != best_precision:
            report['recommendations'].append(f"Consider using {best_diversity} system when diversity is more important than precision.")
        
        if report['summary']['coverage']['best_score'] < 0.1:
            report['recommendations'].append("Low coverage across all systems suggests need for more diverse recommendation strategies.")
        
        return report
    
    def compare_systems_statistical(self, results):
        """Perform statistical comparison of systems"""
        comparison = {}
        
        systems = list(results.keys())
        metrics = ['precision', 'recall', 'f1_score', 'diversity', 'coverage']
        
        for metric in metrics:
            metric_scores = {system: results[system].get(metric, 0) for system in systems}
            
            # Find best and worst performing systems
            best_system = max(metric_scores, key=metric_scores.get)
            worst_system = min(metric_scores, key=metric_scores.get)
            
            # Calculate relative improvement
            best_score = metric_scores[best_system]
            worst_score = metric_scores[worst_system]
            
            improvement = ((best_score - worst_score) / worst_score * 100) if worst_score > 0 else 0
            
            comparison[metric] = {
                'best_system': best_system,
                'worst_system': worst_system,
                'improvement_percentage': improvement,
                'scores': metric_scores
            }
        
        return comparison
