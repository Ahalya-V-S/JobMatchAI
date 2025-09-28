import pandas as pd
import numpy as np
import re
import string
import streamlit as st
from typing import List, Dict, Any, Optional, Union
import hashlib
from datetime import datetime
import json

class DataValidator:
    """Utility class for data validation"""
    
    @staticmethod
    def validate_user_profile(profile: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate user profile data"""
        errors = []
        
        # Check required fields
        if not profile:
            errors.append("User profile cannot be empty")
            return False, errors
        
        # Validate salary range
        min_salary = profile.get('min_salary')
        max_salary = profile.get('max_salary')
        
        if min_salary is not None and min_salary < 0:
            errors.append("Minimum salary cannot be negative")
        
        if max_salary is not None and max_salary < 0:
            errors.append("Maximum salary cannot be negative")
        
        if (min_salary is not None and max_salary is not None and 
            min_salary > max_salary):
            errors.append("Minimum salary cannot be greater than maximum salary")
        
        # Validate skills
        skills = profile.get('skills', [])
        if not isinstance(skills, list):
            errors.append("Skills must be provided as a list")
        else:
            for skill in skills:
                if not isinstance(skill, str) or not skill.strip():
                    errors.append("All skills must be non-empty strings")
                    break
        
        # Validate experience level
        valid_experience_levels = [
            'Entry level', 'Associate', 'Mid-Senior level', 'Director', 'Executive'
        ]
        experience = profile.get('experience_level')
        if experience and experience not in valid_experience_levels:
            errors.append(f"Experience level must be one of: {', '.join(valid_experience_levels)}")
        
        # Validate job type
        valid_job_types = [
            'Full-time', 'Part-time', 'Contract', 'Temporary', 'Internship', 'Any'
        ]
        job_type = profile.get('job_type')
        if job_type and job_type not in valid_job_types:
            errors.append(f"Job type must be one of: {', '.join(valid_job_types)}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_dataset(data: pd.DataFrame) -> tuple[bool, List[str]]:
        """Validate dataset integrity"""
        errors = []
        
        if data is None or data.empty:
            errors.append("Dataset is empty or None")
            return False, errors
        
        # Check for required columns
        required_columns = ['title', 'company']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Check for completely empty columns
        empty_columns = [col for col in data.columns if data[col].isna().all()]
        if empty_columns:
            errors.append(f"Completely empty columns found: {', '.join(empty_columns)}")
        
        # Validate salary data if present
        if 'salary_min' in data.columns and 'salary_max' in data.columns:
            invalid_salary_rows = (
                (data['salary_min'] > data['salary_max']) & 
                data['salary_min'].notna() & 
                data['salary_max'].notna()
            ).sum()
            
            if invalid_salary_rows > 0:
                errors.append(f"Found {invalid_salary_rows} rows where salary_min > salary_max")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_recommendations(recommendations: pd.DataFrame, 
                               dataset_size: int) -> tuple[bool, List[str]]:
        """Validate recommendation results"""
        errors = []
        
        if recommendations is None or recommendations.empty:
            errors.append("Recommendations are empty")
            return False, errors
        
        # Check for required columns
        if 'job_idx' not in recommendations.columns:
            errors.append("Recommendations must contain 'job_idx' column")
        
        # Check for valid job indices
        if 'job_idx' in recommendations.columns:
            invalid_indices = (
                (recommendations['job_idx'] < 0) | 
                (recommendations['job_idx'] >= dataset_size)
            ).sum()
            
            if invalid_indices > 0:
                errors.append(f"Found {invalid_indices} invalid job indices")
        
        # Check for duplicates
        if 'job_idx' in recommendations.columns:
            duplicates = recommendations['job_idx'].duplicated().sum()
            if duplicates > 0:
                errors.append(f"Found {duplicates} duplicate job recommendations")
        
        return len(errors) == 0, errors

class TextProcessor:
    """Utility class for text processing"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\-\.,;:!?]', '', text)
        
        return text
    
    @staticmethod
    def extract_skills(skills_text: str) -> List[str]:
        """Extract and clean skills from text"""
        if pd.isna(skills_text) or not isinstance(skills_text, str):
            return []
        
        # Remove brackets and quotes
        cleaned = re.sub(r'[\[\]"\']', '', skills_text)
        
        # Split by comma and clean each skill
        skills = [skill.strip().title() for skill in cleaned.split(',')]
        
        # Filter out empty skills and common non-skills
        non_skills = {'', 'None', 'N/A', 'Not Specified', 'Various'}
        skills = [skill for skill in skills if skill and skill not in non_skills]
        
        return skills
    
    @staticmethod
    def normalize_location(location: str) -> str:
        """Normalize location text"""
        if pd.isna(location) or not isinstance(location, str):
            return ""
        
        # Clean and standardize
        location = location.strip()
        
        # Handle remote work indicators
        remote_patterns = [
            r'\b(remote|work from home|wfh|telecommute)\b',
            r'\b(anywhere|global|worldwide)\b'
        ]
        
        for pattern in remote_patterns:
            if re.search(pattern, location, re.IGNORECASE):
                return "Remote"
        
        # Remove extra punctuation and standardize format
        location = re.sub(r'[^\w\s\-,.]', '', location)
        location = re.sub(r'\s+', ' ', location)
        
        return location.title()
    
    @staticmethod
    def extract_salary_from_text(text: str) -> tuple[Optional[float], Optional[float]]:
        """Extract salary range from text"""
        if pd.isna(text) or not isinstance(text, str):
            return None, None
        
        # Common salary patterns
        patterns = [
            r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)[^\d]*?(?:to|-)[\s]*\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d{1,3}(?:,\d{3})*)[kK]?[\s]*?(?:to|-)[\s]*?(\d{1,3}(?:,\d{3})*)[kK]?',
            r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if len(matches[0]) == 2:  # Range found
                    try:
                        min_sal = float(matches[0][0].replace(',', ''))
                        max_sal = float(matches[0][1].replace(',', ''))
                        
                        # Handle 'k' notation
                        if 'k' in text.lower():
                            min_sal *= 1000
                            max_sal *= 1000
                        
                        return min(min_sal, max_sal), max(min_sal, max_sal)
                    except ValueError:
                        continue
                else:  # Single salary found
                    try:
                        salary = float(matches[0].replace(',', ''))
                        if 'k' in text.lower():
                            salary *= 1000
                        return salary, salary
                    except ValueError:
                        continue
        
        return None, None
    
    @staticmethod
    def create_text_fingerprint(text: str) -> str:
        """Create a fingerprint for text deduplication"""
        if not text:
            return ""
        
        # Normalize text
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        normalized = re.sub(r'\s+', ' ', normalized.strip())
        
        # Create hash
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

class RecommendationUtils:
    """Utilities for recommendation systems"""
    
    @staticmethod
    def normalize_scores(scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    @staticmethod
    def combine_scores(score_lists: List[List[float]], 
                      weights: List[float]) -> List[float]:
        """Combine multiple score lists with weights"""
        if not score_lists or not weights:
            return []
        
        if len(score_lists) != len(weights):
            raise ValueError("Number of score lists must match number of weights")
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0 / len(weights)] * len(weights)
        else:
            weights = [w / total_weight for w in weights]
        
        # Combine scores
        combined = []
        max_length = max(len(scores) for scores in score_lists)
        
        for i in range(max_length):
            weighted_sum = 0
            total_weight = 0
            
            for j, scores in enumerate(score_lists):
                if i < len(scores):
                    weighted_sum += scores[i] * weights[j]
                    total_weight += weights[j]
            
            if total_weight > 0:
                combined.append(weighted_sum / total_weight)
            else:
                combined.append(0)
        
        return combined
    
    @staticmethod
    def diversify_recommendations(recommendations: pd.DataFrame, 
                                data: pd.DataFrame,
                                diversity_factor: float = 0.5) -> pd.DataFrame:
        """Add diversity to recommendations using maximal marginal relevance"""
        if recommendations.empty or diversity_factor <= 0:
            return recommendations
        
        # Get the score column (assume it's the second column)
        score_col = recommendations.columns[1]
        
        # Start with the highest scoring item
        diversified = [recommendations.iloc[0]]
        remaining = recommendations.iloc[1:].copy()
        
        while len(diversified) < len(recommendations) and not remaining.empty:
            best_mmr_score = -1
            best_idx = 0
            
            for idx, candidate in remaining.iterrows():
                candidate_job = data.iloc[candidate['job_idx']]
                
                # Calculate relevance (original score)
                relevance = candidate[score_col]
                
                # Calculate diversity (minimum similarity to selected items)
                max_similarity = 0
                for selected in diversified:
                    selected_job = data.iloc[selected['job_idx']]
                    similarity = RecommendationUtils._calculate_job_similarity(
                        candidate_job, selected_job
                    )
                    max_similarity = max(max_similarity, similarity)
                
                # MMR score
                mmr_score = (1 - diversity_factor) * relevance - diversity_factor * max_similarity
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx
            
            # Add best MMR item
            diversified.append(remaining.loc[best_idx])
            remaining = remaining.drop(best_idx)
        
        return pd.DataFrame(diversified).reset_index(drop=True)
    
    @staticmethod
    def _calculate_job_similarity(job1: pd.Series, job2: pd.Series) -> float:
        """Calculate similarity between two jobs"""
        similarity = 0
        
        # Company similarity
        if (pd.notna(job1.get('company')) and pd.notna(job2.get('company')) and
            job1['company'] == job2['company']):
            similarity += 0.3
        
        # Location similarity
        if (pd.notna(job1.get('location')) and pd.notna(job2.get('location')) and
            job1['location'] == job2['location']):
            similarity += 0.2
        
        # Experience level similarity
        if (pd.notna(job1.get('experience_level')) and pd.notna(job2.get('experience_level')) and
            job1['experience_level'] == job2['experience_level']):
            similarity += 0.2
        
        # Job category similarity
        if (pd.notna(job1.get('job_category')) and pd.notna(job2.get('job_category')) and
            job1['job_category'] == job2['job_category']):
            similarity += 0.3
        
        return min(1.0, similarity)
    
    @staticmethod
    def filter_recommendations_by_constraints(recommendations: pd.DataFrame,
                                            data: pd.DataFrame,
                                            constraints: Dict[str, Any]) -> pd.DataFrame:
        """Filter recommendations based on hard constraints"""
        if recommendations.empty:
            return recommendations
        
        filtered_indices = []
        
        for _, rec in recommendations.iterrows():
            job = data.iloc[rec['job_idx']]
            
            # Check each constraint
            valid = True
            
            # Salary constraints
            if constraints.get('min_salary'):
                if (pd.notna(job.get('salary_max')) and 
                    job['salary_max'] < constraints['min_salary']):
                    valid = False
            
            if constraints.get('max_salary'):
                if (pd.notna(job.get('salary_min')) and 
                    job['salary_min'] > constraints['max_salary']):
                    valid = False
            
            # Location constraints
            if constraints.get('location') and constraints['location'] != 'Any':
                if (pd.notna(job.get('location')) and 
                    constraints['location'].lower() not in job['location'].lower()):
                    valid = False
            
            # Job type constraints
            if constraints.get('job_type') and constraints['job_type'] != 'Any':
                if job.get('employment_type') != constraints['job_type']:
                    valid = False
            
            if valid:
                filtered_indices.append(rec.name)
        
        return recommendations.loc[filtered_indices].reset_index(drop=True)

class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = datetime.now()
    
    def end_timer(self, operation: str) -> float:
        """End timing and return duration in seconds"""
        if operation not in self.start_times:
            return 0
        
        duration = (datetime.now() - self.start_times[operation]).total_seconds()
        
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append(duration)
        del self.start_times[operation]
        
        return duration
    
    def get_average_time(self, operation: str) -> float:
        """Get average time for an operation"""
        if operation not in self.metrics or not self.metrics[operation]:
            return 0
        
        return np.mean(self.metrics[operation])
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary for all operations"""
        summary = {}
        
        for operation, times in self.metrics.items():
            if times:
                summary[operation] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'average_time': np.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'std_time': np.std(times)
                }
        
        return summary

class ConfigManager:
    """Configuration management utilities"""
    
    @staticmethod
    def get_default_weights() -> Dict[str, float]:
        """Get default weights for hybrid recommendation"""
        return {
            'content': 0.4,
            'collaborative': 0.3,
            'knowledge': 0.3
        }
    
    @staticmethod
    def validate_weights(weights: Dict[str, float]) -> tuple[bool, str]:
        """Validate recommendation weights"""
        required_keys = {'content', 'collaborative', 'knowledge'}
        
        if not all(key in weights for key in required_keys):
            return False, f"Missing required weight keys: {required_keys - weights.keys()}"
        
        if any(weight < 0 for weight in weights.values()):
            return False, "All weights must be non-negative"
        
        total_weight = sum(weights.values())
        if total_weight == 0:
            return False, "At least one weight must be positive"
        
        return True, ""
    
    @staticmethod
    def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1"""
        total = sum(weights.values())
        if total == 0:
            return ConfigManager.get_default_weights()
        
        return {key: value / total for key, value in weights.items()}

# Performance monitoring instance
performance_monitor = PerformanceMonitor()

# Utility functions for common operations
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers"""
    return numerator / denominator if denominator != 0 else default

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def format_salary(salary: float) -> str:
    """Format salary for display"""
    if pd.isna(salary):
        return "Not specified"
    
    if salary >= 1000000:
        return f"${salary/1000000:.1f}M"
    elif salary >= 1000:
        return f"${salary/1000:.0f}K"
    else:
        return f"${salary:.0f}"

def calculate_percentage(part: float, total: float) -> float:
    """Calculate percentage safely"""
    return safe_divide(part * 100, total, 0.0)

def chunks(lst: List, chunk_size: int):
    """Split list into chunks of specified size"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def create_cache_key(*args) -> str:
    """Create a cache key from arguments"""
    key_parts = []
    for arg in args:
        if isinstance(arg, dict):
            key_parts.append(json.dumps(arg, sort_keys=True))
        elif isinstance(arg, (list, tuple)):
            key_parts.append(str(sorted(arg) if all(isinstance(x, (str, int, float)) for x in arg) else arg))
        else:
            key_parts.append(str(arg))
    
    return hashlib.md5('|'.join(key_parts).encode()).hexdigest()
