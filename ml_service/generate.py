# import os
# import logging
# import numpy as np
# import pandas as pd
# import torch
# from datetime import datetime
# from typing import Dict, Any
# from config import OUTPUT_DIR, SEQ_LENGTH, LATENT_DIM, TABULAR_FEATURES, COND_FEATURES

# logger = logging.getLogger(__name__)

# class DiabetesDataGenerator:
#     def __init__(self):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     def generate_synthetic_data(self, num_samples: int, diabetes_ratio: float = 0.5, hypertension_ratio: float = 0.7) -> Dict[str, Any]:
#         """Generate synthetic diabetes data."""
#         logger.info(f"Generating {num_samples} synthetic diabetes samples...")
        
#         # Generate synthetic tabular data
#         tabular_data = self._generate_tabular_data(num_samples, diabetes_ratio, hypertension_ratio)
        
#         # Generate synthetic time series data
#         timeseries_data = self._generate_timeseries_data(tabular_data)
        
#         # Save to CSV files
#         ts_file, tab_file = self._save_to_csv(timeseries_data, tabular_data, num_samples)
        
#         # Create preview
#         preview = self._create_preview(timeseries_data, tabular_data)
        
#         return {
#             'timeseries_file': ts_file,
#             'tabular_file': tab_file,
#             'preview': preview
#         }

#     def _generate_tabular_data(self, num_samples: int, diabetes_ratio: float, hypertension_ratio: float) -> pd.DataFrame:
#         """Generate synthetic tabular patient data with proper hypertension values."""
#         data = []
        
#         for i in range(num_samples):
#             # Determine diabetes status
#             is_diabetic = np.random.random() < diabetes_ratio
            
#             # Generate correlated features
#             if is_diabetic:
#                 age = np.random.randint(45, 56)
#                 bmi = np.random.normal(29.0, 4.0)
#                 average_rbs = np.random.normal(280, 50)
#                 hba1c = np.random.normal(6.8, 0.3)
#                 hr = np.random.randint(78, 95)
#             else:
#                 age = np.random.randint(30, 45)
#                 bmi = np.random.normal(24.0, 3.0)
#                 average_rbs = np.random.normal(140, 25)
#                 hba1c = np.random.normal(5.8, 0.2)
#                 hr = np.random.randint(70, 85)
            
#             # Constrain values
#             bmi = max(18.5, min(40.0, bmi))
#             average_rbs = max(100.0, min(400.0, average_rbs))
#             hba1c = max(5.72, min(7.0, hba1c))
            
#             # Generate other features
#             rr = np.random.randint(12, 19)
#             spo2 = np.random.uniform(95.0, 100.0)
            
#             # Generate Blood Pressure with realistic correlations
#             is_hypertensive = np.random.random() < hypertension_ratio
            
#             if is_hypertensive or (is_diabetic and np.random.random() < 0.8):  # Diabetics more likely to have hypertension
#                 # Generate hypertensive BP values
#                 systolic = np.random.randint(130, 181)  # 130-180
#                 diastolic = np.random.randint(85, 121)   # 85-120
#                 bp_status = 1
#             else:
#                 # Generate normal BP values
#                 systolic = np.random.randint(100, 121)   # 100-120
#                 diastolic = np.random.randint(70, 81)    # 70-80
#                 bp_status = 0
            
#             # Ensure diastolic is reasonable relative to systolic
#             if diastolic > systolic - 20:
#                 diastolic = max(70, systolic - 20)
            
#             # Format hypertension as "systolic/diastolic"
#             hypertension = f"{systolic}/{diastolic}"
            
#             data.append({
#                 'patient_id': f'P{i+1:05d}',
#                 'age': age,
#                 'bmi': round(bmi, 1),
#                 'average_rbs': round(average_rbs, 1),
#                 'hba1c': round(hba1c, 2),
#                 'hypertension': hypertension,  # NOW INCLUDED!
#                 'respiratory_rate': rr,
#                 'heart_rate': hr,
#                 'spo2': round(spo2, 1),
#                 'diabetes': 1 if is_diabetic else 0,
#                 'bp_status': bp_status
#             })
        
#         return pd.DataFrame(data)

#     def _generate_timeseries_data(self, tabular_df: pd.DataFrame) -> pd.DataFrame:
#         """Generate synthetic time series RBS data."""
#         timeseries_data = []
        
#         base_date = datetime.now().date()
        
#         for _, patient in tabular_df.iterrows():
#             avg_rbs = patient['average_rbs']
#             is_diabetic = patient['diabetes'] == 1
            
#             # Generate hourly RBS pattern
#             for hour_offset in range(SEQ_LENGTH):
#                 current_hour = 6 + hour_offset
#                 timestamp = datetime.combine(base_date, datetime.min.time().replace(hour=current_hour))
                
#                 # Add diurnal variation
#                 if hour_offset <= 2:  # Morning (6-8 AM): Dawn phenomenon
#                     rbs_multiplier = np.random.normal(1.05, 0.1)
#                 elif hour_offset in [3, 4]:  # Post-breakfast (9-10 AM)
#                     rbs_multiplier = np.random.normal(1.25, 0.15)
#                 elif hour_offset in [6, 7]:  # Post-lunch (12-1 PM)
#                     rbs_multiplier = np.random.normal(1.20, 0.12)
#                 elif hour_offset in [8, 9]:  # Post-meal continuation (2-3 PM)
#                     rbs_multiplier = np.random.normal(1.10, 0.10)
#                 else:  # Other times: baseline
#                     rbs_multiplier = np.random.normal(0.95, 0.08)
                
#                 rbs_value = avg_rbs * rbs_multiplier
#                 rbs_value += np.random.normal(0, 15 if is_diabetic else 10)
#                 rbs_value = max(100.0, min(400.0, rbs_value))
                
#                 timeseries_data.append({
#                     'patient_id': patient['patient_id'],
#                     'timestamp': timestamp,
#                     'rbs_value': round(rbs_value, 1)
#                 })
        
#         return pd.DataFrame(timeseries_data)

#     def _save_to_csv(self, timeseries_df: pd.DataFrame, tabular_df: pd.DataFrame, num_samples: int) -> tuple:
#         """Save data to CSV files."""
#         os.makedirs(OUTPUT_DIR, exist_ok=True)
        
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
#         ts_file = os.path.join(OUTPUT_DIR, f"synthetic_timeseries_{timestamp}_{num_samples}.csv")
#         tab_file = os.path.join(OUTPUT_DIR, f"synthetic_tabular_{timestamp}_{num_samples}.csv")
        
#         timeseries_df.to_csv(ts_file, index=False)
#         tabular_df.to_csv(tab_file, index=False)
        
#         logger.info(f"Synthetic data saved to {ts_file} and {tab_file}")
#         logger.info(f"Tabular data columns: {list(tabular_df.columns)}")
#         logger.info(f"Sample hypertension values: {tabular_df['hypertension'].head().tolist()}")
        
#         return ts_file, tab_file

#     def _create_preview(self, timeseries_df: pd.DataFrame, tabular_df: pd.DataFrame) -> Dict[str, Any]:
#         """Create a preview of generated data."""
#         preview = {
#             'num_generated': len(tabular_df),
#             'timeseries_shape': list(timeseries_df.shape),
#             'tabular_shape': list(tabular_df.shape),
#             'tabular_columns': list(tabular_df.columns),
#             'sample_hypertension_values': tabular_df['hypertension'].head(5).tolist(),
#             'sample_patients': []
#         }
        
#         # Add sample data for first 3 patients
#         for i in range(min(3, len(tabular_df))):
#             patient_id = tabular_df.iloc[i]['patient_id']
#             patient_ts = timeseries_df[timeseries_df['patient_id'] == patient_id]
            
#             sample = {
#                 'patient_id': patient_id,
#                 'tabular_data': tabular_df.iloc[i].to_dict(),
#                 'timeseries_sample': patient_ts.head(5).to_dict('records')
#             }
#             preview['sample_patients'].append(sample)
        
#         return preview
import os
import logging
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, Any

from gan_trainer import GANTrainer
from config import OUTPUT_DIR, SEQ_LENGTH, LATENT_DIM, TABULAR_FEATURES, COND_FEATURES

logger = logging.getLogger(__name__)

class DiabetesDataGenerator:
    """GAN-based generator for synthetic diabetes data."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gan_trainer = GANTrainer()
        
        # Try to load pretrained models
        self.models_loaded = self.gan_trainer.load_models()
        
        if self.models_loaded:
            logger.info("Using pretrained GAN models for generation")
        else:
            logger.warning("GAN models not found. Will use statistical generation as fallback.")

    def generate_synthetic_data(self, num_samples: int, diabetes_ratio: float = 0.5, 
                                hypertension_ratio: float = 0.7) -> Dict[str, Any]:
        """Generate synthetic diabetes data using GAN or statistical fallback."""
        logger.info(f"Generating {num_samples} synthetic diabetes samples...")
        
        if self.models_loaded:
            # Use GAN models
            tabular_data, timeseries_data = self._generate_with_gan(
                num_samples, diabetes_ratio, hypertension_ratio
            )
        else:
            # Fallback to statistical generation
            logger.warning("Using statistical fallback for data generation")
            tabular_data = self._generate_tabular_data_statistical(
                num_samples, diabetes_ratio, hypertension_ratio
            )
            timeseries_data = self._generate_timeseries_data_statistical(tabular_data)
        
        # Save to CSV files
        ts_file, tab_file = self._save_to_csv(timeseries_data, tabular_data, num_samples)
        
        # Create preview
        preview = self._create_preview(timeseries_data, tabular_data)
        
        return {
            'timeseries_file': ts_file,
            'tabular_file': tab_file,
            'preview': preview
        }

    def _generate_with_gan(self, num_samples: int, diabetes_ratio: float, 
                          hypertension_ratio: float) -> tuple:
        """Generate data using trained GAN models."""
        logger.info("Generating data with GAN models...")
        
        self.gan_trainer.tab_gen.eval()
        self.gan_trainer.ts_gen.eval()
        self.gan_trainer.cross_modal.eval()
        
        all_tabular_data = []
        all_timeseries_data = []
        
        with torch.no_grad():
            # Generate in batches
            batch_size = 100
            num_batches = (num_samples + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
                
                # Generate condition features
                conditions = self._generate_conditions(current_batch_size, diabetes_ratio)
                conditions_tensor = torch.FloatTensor(conditions).to(self.device)
                
                # Generate latent noise
                z = torch.randn(current_batch_size, LATENT_DIM, device=self.device)
                
                # Generate tabular data
                fake_tabular = self.gan_trainer.tab_gen(z, conditions_tensor)
                
                # Generate time series data
                fake_timeseries = self.gan_trainer.ts_gen(z, conditions_tensor)
                
                # Convert to numpy and denormalize
                fake_tabular_np = fake_tabular.cpu().numpy()
                fake_timeseries_np = fake_timeseries.cpu().numpy()
                
                # Process each sample
                for i in range(current_batch_size):
                    patient_id = f'P{batch_idx * batch_size + i + 1:05d}'
                    
                    # Denormalize tabular data
                    tab_sample = self._denormalize_tabular(fake_tabular_np[i], conditions[i])
                    tab_sample['patient_id'] = patient_id
                    
                    # Determine outcomes
                    tab_sample['diabetes'] = 1 if conditions[i][0] > 0.5 else 0  # Based on age/conditions
                    if tab_sample['average_rbs'] > 200 and tab_sample['hba1c'] > 6.7:
                        tab_sample['diabetes'] = 1
                    elif tab_sample['average_rbs'] < 200 and tab_sample['hba1c'] < 5.7:
                        tab_sample['diabetes'] = 0
                    
                    tab_sample['bp_status'] = 1 if hypertension_ratio > np.random.random() else 0
                    
                    all_tabular_data.append(tab_sample)
                    
                    # Denormalize time series data
                    ts_samples = self._denormalize_timeseries(
                        fake_timeseries_np[i], patient_id, tab_sample['average_rbs']
                    )
                    all_timeseries_data.extend(ts_samples)
        
        tabular_df = pd.DataFrame(all_tabular_data)
        timeseries_df = pd.DataFrame(all_timeseries_data)
        
        logger.info(f"Generated {len(tabular_df)} patients using GAN")
        
        return tabular_df, timeseries_df

    def _generate_conditions(self, num_samples: int, diabetes_ratio: float):
        """Generate condition features for conditional GAN."""
        conditions = []
        for _ in range(num_samples):
            is_diabetic = np.random.random() < diabetes_ratio
            
            if is_diabetic:
                age = np.random.randint(45, 56) / 55.0  # Normalize
                bmi = np.random.normal(29.0, 4.0) / 45.0
                hba1c = np.random.normal(6.8, 0.3) / 7.0
            else:
                age = np.random.randint(30, 45) / 55.0
                bmi = np.random.normal(24.0, 3.0) / 45.0
                hba1c = np.random.normal(5.8, 0.2) / 7.0
            
            conditions.append([age, bmi, hba1c])
        
        return conditions

    def _denormalize_tabular(self, normalized_data, condition):
        """Denormalize tabular data from [0,1] to real ranges."""
        # Extract features (excluding hypertension which is split into systolic/diastolic)
        age = condition[0] * 55.0
        bmi = condition[1] * 45.0
        hba1c = condition[2] * 7.0
        
        # Denormalize other features
        average_rbs = normalized_data[0] * 300.0 + 100.0  # 100-400 range
        respiratory_rate = int(normalized_data[1] * 6 + 12)  # 12-18 range
        heart_rate = int(normalized_data[2] * 30 + 70)  # 70-100 range
        spo2 = normalized_data[3] * 5.0 + 95.0  # 95-100 range
        
        # BP values (normalized_data[4] and [5] are systolic/diastolic)
        systolic = int(normalized_data[4] * 80 + 100)  # 100-180 range
        diastolic = int(normalized_data[5] * 50 + 70)  # 70-120 range
        
        # Ensure diastolic is reasonable relative to systolic
        if diastolic > systolic - 20:
            diastolic = max(70, systolic - 20)
        
        hypertension = f"{systolic}/{diastolic}"
        
        return {
            'age': int(age),
            'bmi': round(float(bmi), 1),
            'average_rbs': round(float(average_rbs), 1),
            'hba1c': round(float(hba1c), 2),
            'hypertension': hypertension,
            'respiratory_rate': respiratory_rate,
            'heart_rate': heart_rate,
            'spo2': round(float(spo2), 1)
        }

    def _denormalize_timeseries(self, normalized_sequence, patient_id, avg_rbs):
        """Denormalize time series data."""
        base_date = datetime.now().date()
        timeseries_samples = []
        
        for hour_offset in range(SEQ_LENGTH):
            current_hour = 6 + hour_offset
            timestamp = datetime.combine(base_date, datetime.min.time().replace(hour=current_hour))
            
            # Denormalize RBS value
            rbs_value = float(normalized_sequence[hour_offset][0]) * 300.0 + 100.0
            
            # Add some variation around the average
            rbs_value = rbs_value * 0.7 + avg_rbs * 0.3
            rbs_value = max(100.0, min(400.0, rbs_value))
            
            timeseries_samples.append({
                'patient_id': patient_id,
                'timestamp': timestamp,
                'rbs_value': round(rbs_value, 1)
            })
        
        return timeseries_samples

    def _generate_tabular_data_statistical(self, num_samples: int, diabetes_ratio: float, 
                                          hypertension_ratio: float) -> pd.DataFrame:
        """Statistical fallback for tabular data generation."""
        data = []
        
        for i in range(num_samples):
            is_diabetic = np.random.random() < diabetes_ratio
            
            if is_diabetic:
                age = np.random.randint(45, 56)
                bmi = np.random.normal(29.0, 4.0)
                average_rbs = np.random.normal(280, 50)
                hba1c = np.random.normal(6.8, 0.3)
                hr = np.random.randint(78, 95)
            else:
                age = np.random.randint(30, 45)
                bmi = np.random.normal(24.0, 3.0)
                average_rbs = np.random.normal(140, 25)
                hba1c = np.random.normal(5.8, 0.2)
                hr = np.random.randint(70, 85)
            
            bmi = max(18.5, min(40.0, bmi))
            average_rbs = max(100.0, min(400.0, average_rbs))
            hba1c = max(5.72, min(7.0, hba1c))
            
            rr = np.random.randint(12, 19)
            spo2 = np.random.uniform(95.0, 100.0)
            
            is_hypertensive = np.random.random() < hypertension_ratio
            
            if is_hypertensive or (is_diabetic and np.random.random() < 0.8):
                systolic = np.random.randint(130, 181)
                diastolic = np.random.randint(85, 121)
                bp_status = 1
            else:
                systolic = np.random.randint(100, 121)
                diastolic = np.random.randint(70, 81)
                bp_status = 0
            
            if diastolic > systolic - 20:
                diastolic = max(70, systolic - 20)
            
            hypertension = f"{systolic}/{diastolic}"
            
            data.append({
                'patient_id': f'P{i+1:05d}',
                'age': age,
                'bmi': round(bmi, 1),
                'average_rbs': round(average_rbs, 1),
                'hba1c': round(hba1c, 2),
                'hypertension': hypertension,
                'respiratory_rate': rr,
                'heart_rate': hr,
                'spo2': round(spo2, 1),
                'diabetes': 1 if is_diabetic else 0,
                'bp_status': bp_status
            })
        
        return pd.DataFrame(data)

    def _generate_timeseries_data_statistical(self, tabular_df: pd.DataFrame) -> pd.DataFrame:
        """Statistical fallback for time series generation."""
        timeseries_data = []
        base_date = datetime.now().date()
        
        for _, patient in tabular_df.iterrows():
            avg_rbs = patient['average_rbs']
            is_diabetic = patient['diabetes'] == 1
            
            for hour_offset in range(SEQ_LENGTH):
                current_hour = 6 + hour_offset
                timestamp = datetime.combine(base_date, datetime.min.time().replace(hour=current_hour))
                
                if hour_offset <= 2:
                    rbs_multiplier = np.random.normal(1.05, 0.1)
                elif hour_offset in [3, 4]:
                    rbs_multiplier = np.random.normal(1.25, 0.15)
                elif hour_offset in [6, 7]:
                    rbs_multiplier = np.random.normal(1.20, 0.12)
                elif hour_offset in [8, 9]:
                    rbs_multiplier = np.random.normal(1.10, 0.10)
                else:
                    rbs_multiplier = np.random.normal(0.95, 0.08)
                
                rbs_value = avg_rbs * rbs_multiplier
                rbs_value += np.random.normal(0, 15 if is_diabetic else 10)
                rbs_value = max(100.0, min(400.0, rbs_value))
                
                timeseries_data.append({
                    'patient_id': patient['patient_id'],
                    'timestamp': timestamp,
                    'rbs_value': round(rbs_value, 1)
                })
        
        return pd.DataFrame(timeseries_data)

    def _save_to_csv(self, timeseries_df: pd.DataFrame, tabular_df: pd.DataFrame, 
                     num_samples: int) -> tuple:
        """Save data to CSV files."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        generation_method = "GAN" if self.models_loaded else "Statistical"
        
        ts_file = os.path.join(OUTPUT_DIR, f"synthetic_timeseries_{generation_method}_{timestamp}_{num_samples}.csv")
        tab_file = os.path.join(OUTPUT_DIR, f"synthetic_tabular_{generation_method}_{timestamp}_{num_samples}.csv")
        
        timeseries_df.to_csv(ts_file, index=False)
        tabular_df.to_csv(tab_file, index=False)
        
        logger.info(f"Synthetic data saved using {generation_method} method")
        logger.info(f"Files: {ts_file}, {tab_file}")
        
        return ts_file, tab_file

    def _create_preview(self, timeseries_df: pd.DataFrame, tabular_df: pd.DataFrame) -> Dict[str, Any]:
        """Create a preview of generated data."""
        preview = {
            'generation_method': 'GAN' if self.models_loaded else 'Statistical',
            'num_generated': len(tabular_df),
            'timeseries_shape': list(timeseries_df.shape),
            'tabular_shape': list(tabular_df.shape),
            'tabular_columns': list(tabular_df.columns),
            'sample_hypertension_values': tabular_df['hypertension'].head(5).tolist(),
            'sample_patients': []
        }
        
        for i in range(min(3, len(tabular_df))):
            patient_id = tabular_df.iloc[i]['patient_id']
            patient_ts = timeseries_df[timeseries_df['patient_id'] == patient_id]
            
            sample = {
                'patient_id': patient_id,
                'tabular_data': tabular_df.iloc[i].to_dict(),
                'timeseries_sample': patient_ts.head(5).to_dict('records')
            }
            preview['sample_patients'].append(sample)
        
        return preview
