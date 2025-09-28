import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from collections import Counter
import re

class Analytics:
    """Analytics and visualization for LinkedIn jobs dataset"""
    
    def __init__(self, data):
        self.data = data
    
    def get_experience_distribution(self):
        """Get distribution of jobs by experience level"""
        if 'experience_level' not in self.data.columns:
            return pd.Series()
        
        exp_dist = self.data['experience_level'].value_counts()
        return exp_dist
    
    def get_top_companies(self, n=15):
        """Get top companies by number of job postings"""
        if 'company' not in self.data.columns:
            return pd.Series()
        
        company_counts = self.data['company'].value_counts().head(n)
        return company_counts
    
    def get_top_locations(self, n=15):
        """Get top locations by number of job postings"""
        if 'location' not in self.data.columns:
            return pd.Series()
        
        location_counts = self.data['location'].value_counts().head(n)
        return location_counts
    
    def get_salary_statistics(self):
        """Get salary statistics"""
        salary_stats = {}
        
        if 'salary_min' in self.data.columns and 'salary_max' in self.data.columns:
            # Calculate average salary
            salary_avg = (self.data['salary_min'] + self.data['salary_max']) / 2
            salary_avg = salary_avg.dropna()
            
            if len(salary_avg) > 0:
                salary_stats = {
                    'mean': salary_avg.mean(),
                    'median': salary_avg.median(),
                    'min': salary_avg.min(),
                    'max': salary_avg.max(),
                    'std': salary_avg.std(),
                    'count': len(salary_avg)
                }
        
        return salary_stats
    
    def get_top_skills(self, n=20):
        """Get most in-demand skills"""
        if 'skills' not in self.data.columns:
            return pd.Series()
        
        all_skills = []
        
        for skills_str in self.data['skills'].dropna():
            if isinstance(skills_str, str) and skills_str.strip():
                # Clean and split skills
                skills = [skill.strip().title() for skill in skills_str.split(',')]
                all_skills.extend(skills)
        
        if not all_skills:
            return pd.Series()
        
        # Count skills and return top N
        skill_counts = pd.Series(Counter(all_skills))
        return skill_counts.head(n)
    
    def get_job_type_distribution(self):
        """Get distribution of employment types"""
        if 'employment_type' not in self.data.columns:
            return pd.Series()
        
        return self.data['employment_type'].value_counts()
    
    def get_remote_job_statistics(self):
        """Get statistics about remote jobs"""
        stats = {}
        
        if 'is_remote' in self.data.columns:
            remote_count = self.data['is_remote'].sum()
            total_count = len(self.data)
            
            stats['remote_jobs'] = remote_count
            stats['total_jobs'] = total_count
            stats['remote_percentage'] = (remote_count / total_count) * 100 if total_count > 0 else 0
        else:
            # Check location field for remote indicators
            remote_keywords = ['remote', 'work from home', 'wfh', 'telecommute']
            remote_pattern = '|'.join(remote_keywords)
            
            if 'location' in self.data.columns:
                remote_mask = self.data['location'].str.contains(remote_pattern, case=False, na=False)
                remote_count = remote_mask.sum()
                total_count = len(self.data)
                
                stats['remote_jobs'] = remote_count
                stats['total_jobs'] = total_count
                stats['remote_percentage'] = (remote_count / total_count) * 100 if total_count > 0 else 0
        
        return stats
    
    def get_salary_by_experience(self):
        """Get salary statistics by experience level"""
        if not {'salary_min', 'salary_max', 'experience_level'}.issubset(self.data.columns):
            return pd.DataFrame()
        
        # Calculate average salary
        salary_data = self.data.copy()
        salary_data['salary_avg'] = (salary_data['salary_min'] + salary_data['salary_max']) / 2
        
        # Group by experience level
        salary_by_exp = salary_data.groupby('experience_level')['salary_avg'].agg([
            'mean', 'median', 'count', 'std'
        ]).round(0)
        
        return salary_by_exp
    
    def get_salary_by_location(self, top_n=10):
        """Get salary statistics by location"""
        if not {'salary_min', 'salary_max', 'location'}.issubset(self.data.columns):
            return pd.DataFrame()
        
        # Calculate average salary
        salary_data = self.data.copy()
        salary_data['salary_avg'] = (salary_data['salary_min'] + salary_data['salary_max']) / 2
        
        # Get top locations by job count
        top_locations = self.data['location'].value_counts().head(top_n).index
        
        # Filter for top locations and group
        location_salary = salary_data[salary_data['location'].isin(top_locations)]
        salary_by_location = location_salary.groupby('location')['salary_avg'].agg([
            'mean', 'median', 'count'
        ]).round(0)
        
        return salary_by_location.sort_values('mean', ascending=False)
    
    def get_skills_by_job_category(self):
        """Get top skills by job category"""
        if not {'skills', 'job_category'}.issubset(self.data.columns):
            return {}
        
        skills_by_category = {}
        
        for category in self.data['job_category'].unique():
            if pd.isna(category):
                continue
                
            category_data = self.data[self.data['job_category'] == category]
            
            # Extract skills for this category
            category_skills = []
            for skills_str in category_data['skills'].dropna():
                if isinstance(skills_str, str) and skills_str.strip():
                    skills = [skill.strip().title() for skill in skills_str.split(',')]
                    category_skills.extend(skills)
            
            if category_skills:
                skill_counts = pd.Series(Counter(category_skills))
                skills_by_category[category] = skill_counts.head(10)
        
        return skills_by_category
    
    def create_salary_distribution_plot(self):
        """Create salary distribution visualization"""
        if not {'salary_min', 'salary_max'}.issubset(self.data.columns):
            return None
        
        salary_data = self.data.copy()
        salary_data['salary_avg'] = (salary_data['salary_min'] + salary_data['salary_max']) / 2
        salary_clean = salary_data['salary_avg'].dropna()
        
        if len(salary_clean) == 0:
            return None
        
        fig = px.histogram(
            x=salary_clean,
            nbins=50,
            title="Salary Distribution",
            labels={'x': 'Average Salary ($)', 'y': 'Number of Jobs'}
        )
        
        fig.update_layout(
            xaxis_title="Average Salary ($)",
            yaxis_title="Number of Jobs",
            bargap=0.1
        )
        
        return fig
    
    def create_top_companies_plot(self, n=15):
        """Create top companies visualization"""
        top_companies = self.get_top_companies(n)
        
        if top_companies.empty:
            return None
        
        fig = px.bar(
            x=top_companies.values,
            y=top_companies.index,
            orientation='h',
            title=f"Top {n} Companies by Job Postings",
            labels={'x': 'Number of Jobs', 'y': 'Company'}
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=600
        )
        
        return fig
    
    def create_experience_level_plot(self):
        """Create experience level distribution plot"""
        exp_dist = self.get_experience_distribution()
        
        if exp_dist.empty:
            return None
        
        fig = px.pie(
            values=exp_dist.values,
            names=exp_dist.index,
            title="Job Distribution by Experience Level"
        )
        
        return fig
    
    def create_skills_heatmap(self):
        """Create skills demand heatmap by job category"""
        skills_by_category = self.get_skills_by_job_category()
        
        if not skills_by_category:
            return None
        
        # Prepare data for heatmap
        all_skills = set()
        for skills in skills_by_category.values():
            all_skills.update(skills.index[:5])  # Top 5 skills per category
        
        all_skills = sorted(list(all_skills))
        categories = list(skills_by_category.keys())
        
        # Create matrix
        heatmap_data = []
        for category in categories:
            row = []
            for skill in all_skills:
                count = skills_by_category[category].get(skill, 0)
                row.append(count)
            heatmap_data.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=all_skills,
            y=categories,
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title="Skills Demand by Job Category",
            xaxis_title="Skills",
            yaxis_title="Job Categories",
            height=600
        )
        
        return fig
    
    def create_salary_by_experience_plot(self):
        """Create salary comparison by experience level"""
        salary_by_exp = self.get_salary_by_experience()
        
        if salary_by_exp.empty:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=salary_by_exp.index,
            y=salary_by_exp['mean'],
            name='Average Salary',
            error_y=dict(type='data', array=salary_by_exp['std'])
        ))
        
        fig.add_trace(go.Scatter(
            x=salary_by_exp.index,
            y=salary_by_exp['median'],
            mode='markers+lines',
            name='Median Salary',
            marker=dict(color='red', size=8)
        ))
        
        fig.update_layout(
            title="Salary by Experience Level",
            xaxis_title="Experience Level",
            yaxis_title="Salary ($)",
            height=500
        )
        
        return fig
    
    def create_location_salary_plot(self):
        """Create salary comparison by location"""
        salary_by_location = self.get_salary_by_location()
        
        if salary_by_location.empty:
            return None
        
        fig = px.bar(
            x=salary_by_location['mean'],
            y=salary_by_location.index,
            orientation='h',
            title="Average Salary by Location (Top 10)",
            labels={'x': 'Average Salary ($)', 'y': 'Location'}
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=500
        )
        
        return fig
    
    def create_job_trends_plot(self):
        """Create job posting trends if date information is available"""
        # This would require date information in the dataset
        # For now, return a placeholder or analyze other trends
        
        if 'posted_date' in self.data.columns:
            # Actual trend analysis if date column exists
            date_data = pd.to_datetime(self.data['posted_date'], errors='coerce')
            if not date_data.isna().all():
                daily_posts = date_data.dt.date.value_counts().sort_index()
                
                fig = px.line(
                    x=daily_posts.index,
                    y=daily_posts.values,
                    title="Job Posting Trends Over Time",
                    labels={'x': 'Date', 'y': 'Number of Job Posts'}
                )
                
                return fig
        
        # Alternative: analyze by job category distribution
        if 'job_category' in self.data.columns:
            category_dist = self.data['job_category'].value_counts()
            
            fig = px.bar(
                x=category_dist.values,
                y=category_dist.index,
                orientation='h',
                title="Job Distribution by Category",
                labels={'x': 'Number of Jobs', 'y': 'Job Category'}
            )
            
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=500
            )
            
            return fig
        
        return None
    
    def get_dataset_summary(self):
        """Get comprehensive dataset summary"""
        summary = {
            'total_jobs': len(self.data),
            'total_companies': self.data['company'].nunique() if 'company' in self.data.columns else 0,
            'total_locations': self.data['location'].nunique() if 'location' in self.data.columns else 0,
            'missing_values': self.data.isnull().sum().sum(),
            'data_completeness': ((self.data.size - self.data.isnull().sum().sum()) / self.data.size) * 100,
            'salary_info_coverage': 0,
            'skills_info_coverage': 0
        }
        
        # Calculate salary information coverage
        if 'salary_min' in self.data.columns or 'salary_max' in self.data.columns:
            salary_available = 0
            if 'salary_min' in self.data.columns:
                salary_available += self.data['salary_min'].notna().sum()
            if 'salary_max' in self.data.columns:
                salary_available += self.data['salary_max'].notna().sum()
            
            summary['salary_info_coverage'] = (salary_available / (len(self.data) * 2)) * 100
        
        # Calculate skills information coverage
        if 'skills' in self.data.columns:
            skills_available = self.data['skills'].notna().sum()
            summary['skills_info_coverage'] = (skills_available / len(self.data)) * 100
        
        return summary
    
    def create_comprehensive_dashboard(self):
        """Create a comprehensive analytics dashboard"""
        plots = {}
        
        # Create all available plots
        plots['salary_distribution'] = self.create_salary_distribution_plot()
        plots['top_companies'] = self.create_top_companies_plot()
        plots['experience_levels'] = self.create_experience_level_plot()
        plots['salary_by_experience'] = self.create_salary_by_experience_plot()
        plots['location_salary'] = self.create_location_salary_plot()
        plots['job_trends'] = self.create_job_trends_plot()
        plots['skills_heatmap'] = self.create_skills_heatmap()
        
        # Filter out None plots
        available_plots = {k: v for k, v in plots.items() if v is not None}
        
        return available_plots
