import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

class KnowledgeBasedFilter:
    """Knowledge-based filtering using user constraints and domain rules"""
    
    def __init__(self):
        self.data = None
        self.scaler = MinMaxScaler()
        self.rules = []
    
    def fit(self, data):
        """Initialize the knowledge-based filter with data and rules"""
        st.info("Initializing Knowledge-Based Filter...")
        
        self.data = data.copy()
        self._define_recommendation_rules()
        
        st.success("Knowledge-Based Filter initialization completed!")
    
    def _define_recommendation_rules(self):
        """Define domain-specific recommendation rules"""
        
        # Rule 1: Experience level compatibility
        self.rules.append({
            'name': 'experience_compatibility',
            'weight': 0.25,
            'function': self._check_experience_compatibility
        })
        
        # Rule 2: Salary range fit
        self.rules.append({
            'name': 'salary_fit',
            'weight': 0.20,
            'function': self._check_salary_fit
        })
        
        # Rule 3: Skills match
        self.rules.append({
            'name': 'skills_match',
            'weight': 0.25,
            'function': self._check_skills_match
        })
        
        # Rule 4: Location preference
        self.rules.append({
            'name': 'location_preference',
            'weight': 0.15,
            'function': self._check_location_preference
        })
        
        # Rule 5: Job type compatibility
        self.rules.append({
            'name': 'job_type_compatibility',
            'weight': 0.10,
            'function': self._check_job_type_compatibility
        })
        
        # Rule 6: Company size preference
        self.rules.append({
            'name': 'company_size_preference',
            'weight': 0.05,
            'function': self._check_company_size_preference
        })
        
        st.info(f"Defined {len(self.rules)} recommendation rules")
    
    def recommend(self, user_profile, n_recommendations=10):
        """Generate knowledge-based recommendations"""
        if self.data is None:
            return None
        
        # Apply hard constraints first
        candidate_jobs = self._apply_hard_constraints(user_profile)
        
        if candidate_jobs.empty:
            st.warning("No jobs match the hard constraints. Relaxing constraints...")
            candidate_jobs = self._apply_soft_constraints(user_profile)
        
        if candidate_jobs.empty:
            return None
        
        # Score remaining jobs using rules
        job_scores = []
        
        for job_idx in candidate_jobs.index:
            job = candidate_jobs.loc[job_idx]
            total_score = 0
            
            for rule in self.rules:
                rule_score = rule['function'](job, user_profile)
                weighted_score = rule_score * rule['weight']
                total_score += weighted_score
            
            job_scores.append({
                'job_idx': job_idx,
                'kb_score': total_score
            })
        
        # Sort by score and return top recommendations
        job_scores.sort(key=lambda x: x['kb_score'], reverse=True)
        top_jobs = job_scores[:n_recommendations]
        
        return pd.DataFrame(top_jobs)
    
    def _apply_hard_constraints(self, user_profile):
        """Apply mandatory constraints that cannot be violated"""
        filtered_data = self.data.copy()
        
        # Salary constraints (hard constraint)
        if user_profile.get('min_salary'):
            # Job must pay at least the minimum salary requirement
            salary_mask = (
                (filtered_data['salary_max'] >= user_profile['min_salary']) |
                filtered_data['salary_max'].isna()
            )
            filtered_data = filtered_data[salary_mask]
        
        if user_profile.get('max_salary'):
            # Job's minimum salary shouldn't exceed user's maximum
            salary_mask = (
                (filtered_data['salary_min'] <= user_profile['max_salary']) |
                filtered_data['salary_min'].isna()
            )
            filtered_data = filtered_data[salary_mask]
        
        # Location constraints (if specified as hard requirement)
        if user_profile.get('location') and user_profile['location'] != 'Any':
            if not user_profile.get('remote_ok', True):  # If user doesn't want remote
                location_mask = filtered_data['location'].str.contains(
                    user_profile['location'], case=False, na=False
                )
                filtered_data = filtered_data[location_mask]
        
        # Job type constraints
        if user_profile.get('job_type') and user_profile['job_type'] != 'Any':
            job_type_mask = filtered_data['employment_type'] == user_profile['job_type']
            filtered_data = filtered_data[job_type_mask]
        
        return filtered_data
    
    def _apply_soft_constraints(self, user_profile):
        """Apply relaxed constraints when hard constraints are too restrictive"""
        filtered_data = self.data.copy()
        
        # Relax salary constraints (allow 20% deviation)
        if user_profile.get('min_salary'):
            relaxed_min = user_profile['min_salary'] * 0.8
            salary_mask = (
                (filtered_data['salary_max'] >= relaxed_min) |
                filtered_data['salary_max'].isna()
            )
            filtered_data = filtered_data[salary_mask]
        
        # Keep other constraints as soft preferences
        return filtered_data
    
    def _check_experience_compatibility(self, job, user_profile):
        """Check if job experience level matches user's experience"""
        user_exp = user_profile.get('experience_level')
        job_exp = job.get('experience_level')
        
        if not user_exp or not job_exp:
            return 0.5  # Neutral score if information is missing
        
        # Define experience level hierarchy
        exp_levels = {
            'Entry level': 1,
            'Associate': 2,
            'Mid-Senior level': 3,
            'Director': 4,
            'Executive': 5
        }
        
        user_level = exp_levels.get(user_exp, 0)
        job_level = exp_levels.get(job_exp, 0)
        
        if user_level == 0 or job_level == 0:
            return 0.5
        
        # Perfect match
        if user_level == job_level:
            return 1.0
        
        # One level difference (acceptable)
        if abs(user_level - job_level) == 1:
            return 0.8
        
        # Two levels difference (possible but less ideal)
        if abs(user_level - job_level) == 2:
            return 0.4
        
        # More than two levels difference (poor match)
        return 0.1
    
    def _check_salary_fit(self, job, user_profile):
        """Check how well job salary fits user expectations"""
        user_min = user_profile.get('min_salary')
        user_max = user_profile.get('max_salary')
        job_min = job.get('salary_min')
        job_max = job.get('salary_max')
        
        # If salary information is missing, return neutral score
        if pd.isna(job_min) and pd.isna(job_max):
            return 0.5
        
        if not user_min and not user_max:
            return 0.5
        
        # Calculate job salary range midpoint
        if pd.notna(job_min) and pd.notna(job_max):
            job_salary = (job_min + job_max) / 2
        elif pd.notna(job_min):
            job_salary = job_min
        elif pd.notna(job_max):
            job_salary = job_max
        else:
            return 0.5
        
        # Calculate user salary range midpoint
        if user_min and user_max:
            user_salary = (user_min + user_max) / 2
        elif user_min:
            user_salary = user_min
        elif user_max:
            user_salary = user_max
        else:
            return 0.5
        
        # Check if job salary is within acceptable range
        if user_min and user_max:
            if user_min <= job_salary <= user_max:
                return 1.0
            elif job_salary < user_min:
                # Salary too low
                ratio = job_salary / user_min
                return max(0, ratio)
            else:
                # Salary higher than expected (usually good)
                return min(1.0, 1.0 + (job_salary - user_max) / user_max * 0.5)
        
        return 0.5
    
    def _check_skills_match(self, job, user_profile):
        """Check how well job required skills match user skills"""
        user_skills = set(skill.lower() for skill in user_profile.get('skills', []))
        job_skills_str = job.get('skills', '')
        
        if not user_skills or not job_skills_str:
            return 0.3  # Low score if skills information is missing
        
        # Extract job skills
        job_skills = set()
        if isinstance(job_skills_str, str) and job_skills_str:
            job_skills = set(skill.strip().lower() for skill in job_skills_str.split(','))
        
        if not job_skills:
            return 0.3
        
        # Calculate skill match percentage
        matching_skills = user_skills & job_skills
        
        if not matching_skills:
            return 0.1  # Very low score if no skills match
        
        # Score based on percentage of user skills that match
        user_match_ratio = len(matching_skills) / len(user_skills)
        
        # Score based on percentage of job requirements met
        job_match_ratio = len(matching_skills) / len(job_skills)
        
        # Combine both ratios (weighted towards user skills coverage)
        combined_score = 0.7 * user_match_ratio + 0.3 * job_match_ratio
        
        return min(1.0, combined_score)
    
    def _check_location_preference(self, job, user_profile):
        """Check location preference match"""
        user_location = user_profile.get('location')
        job_location = job.get('location', '')
        
        if not user_location or user_location == 'Any':
            return 0.7  # Neutral-positive score if no preference
        
        if not job_location:
            return 0.3  # Low score if job location is unknown
        
        # Check for exact location match
        if user_location.lower() in job_location.lower():
            return 1.0
        
        # Check for remote work compatibility
        if job.get('is_remote', False):
            return 0.9  # High score for remote work
        
        # Extract city/state for partial matches
        user_parts = user_location.split(',')
        job_parts = job_location.split(',')
        
        # Check for partial matches (city or state)
        for user_part in user_parts:
            user_part = user_part.strip().lower()
            for job_part in job_parts:
                job_part = job_part.strip().lower()
                if user_part in job_part or job_part in user_part:
                    return 0.7
        
        return 0.2  # Low score for location mismatch
    
    def _check_job_type_compatibility(self, job, user_profile):
        """Check job type (employment type) compatibility"""
        user_job_type = user_profile.get('job_type')
        job_type = job.get('employment_type')
        
        if not user_job_type or user_job_type == 'Any':
            return 0.7  # Neutral-positive if no preference
        
        if not job_type:
            return 0.5  # Neutral if job type is unknown
        
        # Exact match
        if user_job_type == job_type:
            return 1.0
        
        # Define compatibility matrix
        compatibility = {
            'Full-time': {'Part-time': 0.3, 'Contract': 0.4, 'Temporary': 0.2, 'Internship': 0.1},
            'Part-time': {'Full-time': 0.5, 'Contract': 0.6, 'Temporary': 0.7, 'Internship': 0.3},
            'Contract': {'Full-time': 0.6, 'Part-time': 0.7, 'Temporary': 0.8, 'Internship': 0.2},
            'Temporary': {'Full-time': 0.3, 'Part-time': 0.8, 'Contract': 0.9, 'Internship': 0.4},
            'Internship': {'Full-time': 0.7, 'Part-time': 0.5, 'Contract': 0.3, 'Temporary': 0.2}
        }
        
        return compatibility.get(user_job_type, {}).get(job_type, 0.3)
    
    def _check_company_size_preference(self, job, user_profile):
        """Check company size preference"""
        user_pref = user_profile.get('company_size')
        job_company_size = job.get('company_size')
        
        if not user_pref or user_pref == 'Any':
            return 0.7  # Neutral-positive if no preference
        
        if not job_company_size or job_company_size == 'Unknown':
            return 0.5  # Neutral if company size is unknown
        
        # Exact match
        if user_pref == job_company_size:
            return 1.0
        
        # Define size compatibility (adjacent sizes are somewhat compatible)
        size_order = ['Startup (1-50)', 'Small (51-200)', 'Medium (201-1000)', 'Large (1000+)']
        
        try:
            user_idx = size_order.index(user_pref)
            job_idx = size_order.index(job_company_size)
            
            diff = abs(user_idx - job_idx)
            if diff == 1:
                return 0.6  # Adjacent sizes
            elif diff == 2:
                return 0.3  # Two sizes apart
            else:
                return 0.1  # Very different sizes
        except ValueError:
            return 0.3  # Unknown size category
    
    def explain_recommendation(self, job, user_profile):
        """Provide explanation for why a job was recommended"""
        explanations = []
        
        for rule in self.rules:
            score = rule['function'](job, user_profile)
            weight = rule['weight']
            weighted_score = score * weight
            
            if weighted_score > 0.1:  # Only explain significant factors
                rule_name = rule['name'].replace('_', ' ').title()
                explanations.append({
                    'factor': rule_name,
                    'score': score,
                    'weight': weight,
                    'contribution': weighted_score
                })
        
        # Sort by contribution
        explanations.sort(key=lambda x: x['contribution'], reverse=True)
        
        return explanations
    
    def get_constraint_satisfaction(self, user_profile):
        """Get statistics on how well the dataset satisfies user constraints"""
        if self.data is None:
            return None
        
        stats = {}
        
        # Experience level distribution
        if user_profile.get('experience_level'):
            exp_matches = self.data['experience_level'] == user_profile['experience_level']
            stats['experience_match_rate'] = exp_matches.mean()
        
        # Salary range satisfaction
        if user_profile.get('min_salary') or user_profile.get('max_salary'):
            salary_satisfies = pd.Series([True] * len(self.data))
            
            if user_profile.get('min_salary'):
                salary_satisfies &= (
                    (self.data['salary_max'] >= user_profile['min_salary']) |
                    self.data['salary_max'].isna()
                )
            
            if user_profile.get('max_salary'):
                salary_satisfies &= (
                    (self.data['salary_min'] <= user_profile['max_salary']) |
                    self.data['salary_min'].isna()
                )
            
            stats['salary_satisfaction_rate'] = salary_satisfies.mean()
        
        # Location match rate
        if user_profile.get('location') and user_profile['location'] != 'Any':
            location_matches = self.data['location'].str.contains(
                user_profile['location'], case=False, na=False
            )
            stats['location_match_rate'] = location_matches.mean()
        
        return stats
