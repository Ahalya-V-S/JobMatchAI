
import os
import pandas as pd
import streamlit as st
import kagglehub


class DataLoader:
        """Handles loading and initial processing of the LinkedIn jobs dataset"""

        def __init__(self):
                self.dataset_path = None
                self.data = None

        def download_dataset(self):
                """Download the LinkedIn jobs dataset from Kaggle"""
                try:
                        # Download latest version
                        path = kagglehub.dataset_download("asaniczka/1-3m-linkedin-jobs-and-skills-2024")
                        self.dataset_path = path
                        st.success(f"Dataset downloaded successfully to: {path}")
                        return path
                except Exception as e:
                        # Check for 404 error in the exception message
                        if "404" in str(e):
                                st.error("Failed to download dataset: 404 Not Found. The dataset may be private, removed, or the URL is incorrect. Please check your Kaggle access and the dataset link.")
                        else:
                                st.error(f"Failed to download dataset: {str(e)}")
                        return None

        def load_data(self):
                """Load the LinkedIn jobs dataset"""
                try:
                        # Download dataset if not already downloaded
                        if not self.dataset_path:
                                self.dataset_path = self.download_dataset()

                        if not self.dataset_path:
                                return None

                        # Find CSV files in the dataset directory
                        csv_files = []
                        for root, dirs, files in os.walk(self.dataset_path):
                                for file in files:
                                        if file.endswith('.csv'):
                                                csv_files.append(os.path.join(root, file))

                        if not csv_files:
                                st.error("No CSV files found in the dataset directory")
                                return None

                        # Load the main jobs dataset (usually the largest file)
                        main_file = max(csv_files, key=os.path.getsize)
                        st.info(f"Loading data from: {os.path.basename(main_file)}")

                        # Load with error handling for large files
                        try:
                                self.data = pd.read_csv(main_file, low_memory=False)
                                st.success(f"Successfully loaded {len(self.data):,} job records")

                                # Display basic info about the dataset
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
                                # Load only a sample if memory is insufficient
                                self.data = pd.read_csv(main_file, nrows=50000, low_memory=False)
                                st.success(f"Loaded sample of {len(self.data):,} job records")
                                return self.data
                        except Exception as e:
                                st.error(f"Error loading dataset: {str(e)}")
                                return None

                except Exception as e:
                        st.error(f"Error in load_data: {str(e)}")
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
                        for root, dirs, files in os.walk(self.dataset_path):
                                for file in files:
                                        if file.endswith('.csv'):
                                                file_path = os.path.join(root, file)
                                                file_name = os.path.splitext(file)[0]

                                                # Skip the main jobs file
                                                main_file = max([os.path.join(root, f) for f in files if f.endswith('.csv')], key=os.path.getsize)
                                                if file_path != main_file:
                                                        try:
                                                                additional_data[file_name] = pd.read_csv(file_path, low_memory=False)
                                                                st.info(f"Loaded additional file: {file_name} with {len(additional_data[file_name])} records")
                                                        except Exception as e:
                                                                st.warning(f"Could not load {file_name}: {str(e)}")
                        return additional_data
                except Exception as e:
                        st.warning(f"Error loading additional files: {str(e)}")
                        return additional_data