import os
import pandas as pd
import streamlit as st
import kagglehub
import shutil

class DataLoader:
    """Handles downloading, caching, and loading LinkedIn jobs datasets from Kaggle"""


    def __init__(self, cache_dir="data_cache"):
        self.cache_dir = cache_dir
        self.dataset_path = None
        self.data = None
        os.makedirs(self.cache_dir, exist_ok=True)

    def download_dataset(self):
        """Download the LinkedIn jobs dataset from Kaggle if not cached"""
        try:
            merged_csv = os.path.join(self.cache_dir, "merged_dataset.csv")
            if os.path.exists(merged_csv):
                st.info("Using cached merged CSV")
                self.dataset_path = merged_csv
                return merged_csv

            st.info("Downloading dataset from Kaggle...")
            path = kagglehub.dataset_download("asaniczka/1-3m-linkedin-jobs-and-skills-2024")

            # Find CSV files in the downloaded folder
            csv_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")]
            if not csv_files:
                st.error("No CSV files found in downloaded dataset.")
                return None

            # Load all CSVs into a dictionary
            dfs = {os.path.basename(f): pd.read_csv(f) for f in csv_files}

            # Identify the main dataframe (the one with most columns)
            main_name = max(dfs, key=lambda x: dfs[x].shape[1])
            merged_df = dfs[main_name]

            # Merge all others on 'job_link'
            for name, df in dfs.items():
                if name == main_name:
                    continue
                if "job_link" in df.columns and "job_link" in merged_df.columns:
                    merged_df = merged_df.merge(df, on="job_link", how="left")

            # Save merged dataset
            merged_df.to_csv(merged_csv, index=False)
            st.success(f"Merged dataset saved: {merged_csv}")
            self.dataset_path = merged_csv
            return merged_csv

        except Exception as e:
            if "404" in str(e):
                st.error("Failed to download dataset: 404 Not Found. Check Kaggle access or dataset link.")
            else:
                st.error(f"Failed to download dataset: {str(e)}")
            return None

    def load_data(self):
        """Load the LinkedIn jobs dataset"""
        try:
            # Download or use cached dataset
            if not self.dataset_path:
                self.dataset_path = self.download_dataset()

            if not self.dataset_path:
                return None

            # Load CSV
            try:
                self.data = pd.read_csv(self.dataset_path, nrows=500000, low_memory=False)
                st.success(f"Loaded dataset with {len(self.data):,} rows")
                st.info(f"Dataset shape: {self.data.shape}")
                st.info(f"Columns: {', '.join(self.data.columns.tolist())}")
                return self.data

            except pd.errors.EmptyDataError:
                st.error("The dataset file is empty")
                return None
            except pd.errors.ParserError as e:
                st.error(f"Error parsing CSV file: {str(e)}")
                return None
            except MemoryError:
                st.warning("Dataset is too large, loading a sample...")
                self.data = pd.read_csv(self.dataset_path, nrows=50000, low_memory=False)
                st.success(f"Loaded sample of {len(self.data):,} rows")
                return self.data
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
                return None

        except Exception as e:
            st.error(f"Error in load_data: {str(e)}")
            return None

    def save_csv(self, file_name="linkedin_jobs.csv"):
        """Save the current dataframe to CSV locally"""
        if self.data is None:
            st.warning("No data to save")
            return None

        save_path = os.path.join(self.cache_dir, file_name)
        try:
            self.data.to_csv(save_path, index=False)
            st.success(f"Dataset saved to {save_path}")
            return save_path
        except Exception as e:
            st.error(f"Failed to save CSV: {str(e)}")
            return None

    def get_data_info(self):
        """Get basic information about the loaded data"""
        if self.data is None:
            return None
        return {
            'shape': self.data.shape,
            'columns': self.data.columns.tolist(),
            'dtypes': self.data.dtypes.to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'missing_values': self.data.isnull().sum().to_dict()
        }

    def load_additional_files(self):
        """Load additional CSV files if they exist (skills, companies, etc.)"""
        additional_data = {}

        if not self.dataset_path:
            return additional_data

        try:
            # Look for additional CSV files
            for root, dirs, files in os.walk(os.path.dirname(self.dataset_path)):
                for file in files:
                    if file.endswith('.csv') and os.path.join(root, file) != self.dataset_path:
                        file_path = os.path.join(root, file)
                        file_name = os.path.splitext(file)[0]

                        try:
                            additional_data[file_name] = pd.read_csv(file_path, low_memory=False)
                            st.info(f"Loaded additional file: {file_name} with {len(additional_data[file_name])} rows")
                        except Exception as e:
                            st.warning(f"Could not load {file_name}: {str(e)}")
            return additional_data
        except Exception as e:
            st.warning(f"Error loading additional files: {str(e)}")
            return additional_data
