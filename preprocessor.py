import pandas as pd
import numpy as np
import re
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

class DataPreprocessor:
    """Handles data preprocessing for the LinkedIn jobs dataset"""
    
    def __init__(self):
        self.label_encoders = {}
        self.stemmer = PorterStemmer()
        self._download_nltk_data()
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            st.warning(f"Could not download NLTK data: {str(e)}")
            self.stop_words = set()
    
    def preprocess(self, data):
        """Main preprocessing pipeline"""
        st.info("Starting data preprocessing...")
        
        # Create a copy to avoid modifying original data
        processed_data = data.copy()
        
        # 1. Handle missing values
        processed_data = self._handle_missing_values(processed_data)
        
        # 2. Clean and standardize text fields
        processed_data = self._clean_text_fields(processed_data)
        
        # 3. Extract and process salary information
        processed_data = self._process_salary_data(processed_data)
        
        # 4. Standardize categorical fields
        processed_data = self._standardize_categorical_fields(processed_data)
        
        # 5. Process skills data
        processed_data = self._process_skills_data(processed_data)
        
        # 6. Create derived features
        processed_data = self._create_derived_features(processed_data)
        
        # 7. Generate synthetic user interactions for collaborative filtering
        processed_data = self._generate_user_interactions(processed_data)
        
        st.success("Data preprocessing completed successfully!")
        
        return processed_data
    
    def _handle_missing_values(self, data):
        """Handle missing values in the dataset"""
        st.info("Handling missing values...")
        
        # Fill missing text fields with empty strings
        text_columns = ['title', 'company', 'location', 'description', 'skills']
        for col in text_columns:
            if col in data.columns:
                data[col] = data[col].fillna('')
        
        # Fill missing categorical fields with 'Unknown'
        categorical_columns = ['employment_type', 'experience_level', 'company_size']
        for col in categorical_columns:
            if col in data.columns:
                data[col] = data[col].fillna('Unknown')
        
        # Handle numerical fields
        numerical_columns = ['salary_min', 'salary_max']
        for col in numerical_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        return data
    
    def _clean_text_fields(self, data):
        """Clean and standardize text fields"""
        st.info("Cleaning text fields...")
        
        text_columns = ['title', 'company', 'location', 'description']
        
        for col in text_columns:
            if col in data.columns:
                # Remove extra whitespace and standardize
                data[col] = data[col].astype(str).str.strip()
                data[col] = data[col].str.replace(r'\s+', ' ', regex=True)
                
                # Remove special characters from company and location
                if col in ['company', 'location']:
                    data[col] = data[col].str.replace(r'[^\w\s\-\.,]', '', regex=True)
        
        return data
    
    def _process_salary_data(self, data):
        """Extract and process salary information"""
        st.info("Processing salary data...")
        
        # If salary columns don't exist, try to extract from description or other fields
        if 'salary_min' not in data.columns or 'salary_max' not in data.columns:
            data['salary_min'] = np.nan
            data['salary_max'] = np.nan
            
            # Try to extract salary from description
            if 'description' in data.columns:
                salary_pattern = r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
                for idx, desc in data['description'].items():
                    if pd.notna(desc):
                        salaries = re.findall(salary_pattern, str(desc))
                        if len(salaries) >= 2:
                            try:
                                sal1 = float(salaries[0].replace(',', ''))
                                sal2 = float(salaries[1].replace(',', ''))
                                data.loc[idx, 'salary_min'] = min(sal1, sal2)
                                data.loc[idx, 'salary_max'] = max(sal1, sal2)
                            except:
                                continue
                        elif len(salaries) == 1:
                            try:
                                sal = float(salaries[0].replace(',', ''))
                                data.loc[idx, 'salary_min'] = sal
                                data.loc[idx, 'salary_max'] = sal
                            except:
                                continue
        
        # Clean existing salary data
        for col in ['salary_min', 'salary_max']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                # Remove unrealistic salaries
                data.loc[data[col] < 10000, col] = np.nan
                data.loc[data[col] > 1000000, col] = np.nan
        
        # Ensure salary_min <= salary_max
        if 'salary_min' in data.columns and 'salary_max' in data.columns:
            mask = (data['salary_min'] > data['salary_max']) & data['salary_min'].notna() & data['salary_max'].notna()
            data.loc[mask, ['salary_min', 'salary_max']] = data.loc[mask, ['salary_max', 'salary_min']].values
        
        return data
    
    def _standardize_categorical_fields(self, data):
        """Standardize categorical fields"""
        st.info("Standardizing categorical fields...")
        
        # Standardize experience levels
        if 'experience_level' in data.columns:
            experience_mapping = {
                'entry': 'Entry level',
                'entry level': 'Entry level',
                'junior': 'Entry level',
                'associate': 'Associate',
                'mid': 'Mid-Senior level',
                'mid-level': 'Mid-Senior level',
                'mid-senior': 'Mid-Senior level',
                'mid-senior level': 'Mid-Senior level',
                'senior': 'Mid-Senior level',
                'director': 'Director',
                'executive': 'Executive',
                'c-level': 'Executive'
            }
            
            data['experience_level'] = data['experience_level'].str.lower().map(experience_mapping).fillna(data['experience_level'])
        
        # Standardize employment types
        if 'employment_type' in data.columns:
            employment_mapping = {
                'full time': 'Full-time',
                'full-time': 'Full-time',
                'fulltime': 'Full-time',
                'part time': 'Part-time',
                'part-time': 'Part-time',
                'parttime': 'Part-time',
                'contract': 'Contract',
                'contractor': 'Contract',
                'temporary': 'Temporary',
                'temp': 'Temporary',
                'internship': 'Internship',
                'intern': 'Internship'
            }
            
            data['employment_type'] = data['employment_type'].str.lower().map(employment_mapping).fillna(data['employment_type'])
        
        return data
    
    def _process_skills_data(self, data):
        """Process and clean skills data"""
        st.info("Processing skills data...")
        
        if 'skills' in data.columns:
            # Clean skills field
            data['skills'] = data['skills'].astype(str)
            data['skills'] = data['skills'].str.replace(r'[\[\]"\']', '', regex=True)
            data['skills'] = data['skills'].str.replace(r'\s*,\s*', ', ', regex=True)
            
            # Extract individual skills for analysis
            all_skills = []
            for skills_str in data['skills']:
                if skills_str and skills_str != 'nan':
                    skills = [skill.strip().title() for skill in skills_str.split(',')]
                    all_skills.extend(skills)
            
            # Create skills vocabulary
            unique_skills = list(set(all_skills))
            data['skills_list'] = data['skills'].apply(
                lambda x: [skill.strip().title() for skill in str(x).split(',') if skill.strip() and skill.strip() != 'nan'] if x and x != 'nan' else []
            )
            
            # Count number of skills required
            data['skills_count'] = data['skills_list'].apply(len)
        
        return data
    
    def _create_derived_features(self, data):
        """Create derived features for better recommendations"""
        st.info("Creating derived features...")
        
        # Create salary range feature
        if 'salary_min' in data.columns and 'salary_max' in data.columns:
            data['salary_range'] = data['salary_max'] - data['salary_min']
            data['salary_avg'] = (data['salary_min'] + data['salary_max']) / 2
            
            # Categorize salary levels
            salary_bins = [0, 50000, 75000, 100000, 150000, float('inf')]
            salary_labels = ['Low', 'Medium-Low', 'Medium', 'High', 'Very High']
            data['salary_category'] = pd.cut(data['salary_avg'], bins=salary_bins, labels=salary_labels, include_lowest=True)
        
        # Create company size categories
        if 'company_size' not in data.columns:
            # If company size is not available, create a placeholder
            data['company_size'] = 'Unknown'
        
        # Create location features
        if 'location' in data.columns:
            # Extract state/country information
            data['location_clean'] = data['location'].str.split(',').str[-1].str.strip()
            
            # Mark remote jobs
            data['is_remote'] = data['location'].str.contains('remote|Remote|REMOTE', na=False)
        
        # Create job title categories
        if 'title' in data.columns:
            title_keywords = {
                'Engineering': ['engineer', 'developer', 'programmer', 'software', 'technical'],
                'Management': ['manager', 'director', 'lead', 'head', 'chief', 'vp', 'vice president'],
                'Sales': ['sales', 'account', 'business development', 'revenue'],
                'Marketing': ['marketing', 'brand', 'digital', 'content', 'social media'],
                'Data': ['data', 'analyst', 'scientist', 'analytics', 'business intelligence'],
                'Design': ['design', 'designer', 'ui', 'ux', 'creative', 'graphic'],
                'HR': ['human resources', 'hr', 'recruiter', 'talent', 'people'],
                'Finance': ['finance', 'financial', 'accounting', 'accountant', 'controller']
            }
            
            data['job_category'] = 'Other'
            for category, keywords in title_keywords.items():
                pattern = '|'.join(keywords)
                mask = data['title'].str.contains(pattern, case=False, na=False)
                data.loc[mask, 'job_category'] = category
        
        return data
    
    def _generate_user_interactions(self, data):
        """Generate synthetic user interactions for collaborative filtering"""
        st.info("Generating user interaction data...")
        
        # Create synthetic users based on job characteristics
        np.random.seed(42)
        n_users = min(1000, len(data) // 10)  # Create reasonable number of users
        n_jobs = len(data)
        
        # Generate user profiles
        users = []
        for user_id in range(n_users):
            # Random user preferences
            preferred_location = np.random.choice(data['location'].dropna().unique()) if 'location' in data.columns else None
            preferred_experience = np.random.choice(['Entry level', 'Associate', 'Mid-Senior level', 'Director'])
            preferred_category = np.random.choice(['Engineering', 'Management', 'Sales', 'Marketing', 'Data', 'Design', 'HR', 'Finance', 'Other'])
            
            users.append({
                'user_id': user_id,
                'preferred_location': preferred_location,
                'preferred_experience': preferred_experience,
                'preferred_category': preferred_category
            })
        
        # Generate interactions (ratings) based on user preferences
        interactions = []
        for user in users:
            # Each user interacts with 5-20 jobs
            n_interactions = np.random.randint(5, 21)
            
            # Higher probability of interacting with jobs matching preferences
            job_indices = np.random.choice(n_jobs, size=n_interactions, replace=False)
            
            for job_idx in job_indices:
                job = data.iloc[job_idx]
                
                # Base rating
                rating = np.random.uniform(1, 5)
                
                # Adjust rating based on preferences
                if 'location' in data.columns and user['preferred_location'] in str(job['location']):
                    rating += 1
                
                if 'experience_level' in data.columns and user['preferred_experience'] == job['experience_level']:
                    rating += 0.5
                
                if 'job_category' in data.columns and user['preferred_category'] == job['job_category']:
                    rating += 0.5
                
                # Normalize rating to 1-5 scale
                rating = min(5, max(1, rating))
                
                interactions.append({
                    'user_id': user['user_id'],
                    'job_id': job_idx,
                    'rating': rating
                })
        
        # Create interaction matrix
        interaction_df = pd.DataFrame(interactions)
        data['user_interactions'] = [interaction_df] * len(data)  # Store interactions with data
        
        return data
    
    def clean_text_for_similarity(self, text):
        """Clean text for similarity calculations"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize and remove stopwords
        try:
            tokens = word_tokenize(text)
            tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words and len(token) > 2]
            return ' '.join(tokens)
        except:
            # Fallback if NLTK fails
            tokens = text.split()
            return ' '.join([token for token in tokens if len(token) > 2])
