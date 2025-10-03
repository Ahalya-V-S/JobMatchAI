import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

class Analytics:
    """Analytics and visualization for LinkedIn jobs dataset"""

    def __init__(self, data):
        self.data = data.copy()
        # Rename columns to match existing methods
        self.data.rename(columns={
            'job_level': 'experience_level',
            'job_skills': 'skills',
            'job_type': 'employment_type',
            'last_processed_time': 'last_processed_time'
        }, inplace=True)

    def get_experience_distribution(self):
        if 'experience_level' not in self.data.columns:
            return pd.Series()
        return self.data['experience_level'].value_counts()

    def get_top_companies(self, n=15):
        if 'company' not in self.data.columns:
            return pd.Series()
        return self.data['company'].value_counts().head(n)

    def get_top_locations(self, n=15):
        if 'job_location' not in self.data.columns:
            return pd.Series()
        return self.data['job_location'].value_counts().head(n)

    def get_top_skills(self, n=20):
        if 'skills' not in self.data.columns:
            return pd.Series()
        all_skills = []
        for skills_str in self.data['skills'].dropna():
            if isinstance(skills_str, str) and skills_str.strip():
                skills = [skill.strip().title() for skill in skills_str.split(',')]
                all_skills.extend(skills)
        return pd.Series(Counter(all_skills)).head(n)

    def get_job_type_distribution(self):
        if 'employment_type' not in self.data.columns:
            return pd.Series()
        return self.data['employment_type'].value_counts()

    def get_remote_job_statistics(self):
        stats = {}
        if 'job_location' in self.data.columns:
            remote_keywords = ['remote', 'work from home', 'wfh', 'telecommute']
            remote_pattern = '|'.join(remote_keywords)
            remote_mask = self.data['job_location'].str.contains(remote_pattern, case=False, na=False)
            remote_count = remote_mask.sum()
            total_count = len(self.data)
            stats['remote_jobs'] = remote_count
            stats['total_jobs'] = total_count
            stats['remote_percentage'] = (remote_count / total_count) * 100 if total_count > 0 else 0
        return stats

    def get_dataset_summary(self):
        summary = {
            'total_jobs': len(self.data),
            'total_companies': self.data['company'].nunique() if 'company' in self.data.columns else 0,
            'total_locations': self.data['job_location'].nunique() if 'job_location' in self.data.columns else 0,
            'missing_values': self.data.isnull().sum().sum(),
            'data_completeness': ((self.data.size - self.data.isnull().sum().sum()) / self.data.size) * 100,
            'skills_info_coverage': (self.data['skills'].notna().sum() / len(self.data) * 100) if 'skills' in self.data.columns else 0
        }
        return summary

    # ----------------- Plotting Methods -----------------

    def create_top_companies_plot(self, n=15):
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
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
        return fig

    def create_top_locations_plot(self, n=15):
        top_locations = self.get_top_locations(n)
        if top_locations.empty:
            return None
        fig = px.bar(
            x=top_locations.values,
            y=top_locations.index,
            orientation='h',
            title=f"Top {n} Job Locations",
            labels={'x': 'Number of Jobs', 'y': 'Location'}
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
        return fig

    def create_experience_level_plot(self):
        exp_dist = self.get_experience_distribution()
        if exp_dist.empty:
            return None
        fig = px.pie(values=exp_dist.values, names=exp_dist.index, title="Job Distribution by Experience Level")
        return fig

    def create_top_skills_plot(self, n=20):
        top_skills = self.get_top_skills(n)
        if top_skills.empty:
            return None
        df_skills = top_skills.reset_index()
        df_skills.columns = ['skill', 'count']
        fig = px.bar(
            df_skills,
            x='count',
            y='skill',
            orientation='h',
            title=f"Top {n} Skills in Job Market"
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
        return fig

    def create_job_type_plot(self):
        job_types = self.get_job_type_distribution()
        if job_types.empty:
            return None
        fig = px.pie(values=job_types.values, names=job_types.index, title="Job Type Distribution")
        return fig

    def create_job_posting_trends_plot(self):
        if 'last_processed_time' not in self.data.columns:
            return None

        # Convert to datetime (handles timezone and microseconds)
        df = self.data.copy()
        df['last_processed_time'] = pd.to_datetime(df['last_processed_time'], errors='coerce')

        # Drop invalid dates
        df = df.dropna(subset=['last_processed_time'])
        if df.empty:
            return None

        # Group by week
        df['week'] = df['last_processed_time'].dt.to_period('W').apply(lambda r: r.start_time)

        trends = df.groupby('week').size().reset_index(name='count')

        # Plot
        fig = px.line(trends, x='week', y='count', title="Job Posting Trends Over Time (Weekly)")
        fig.update_layout(xaxis_title="Week", yaxis_title="Number of Jobs", height=500)

        return fig
