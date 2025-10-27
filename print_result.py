import os
files = os.listdir('results')

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

for file in files:
    if file.endswith('_predictions.csv'):
        print(f"example - {file}\n")

        # Parse filename for display
        base_name = file.replace('_predictions.csv', '')
        parts = base_name.split('_')

        try:
            # Example filename structure:
            # globem_compass_4shot_generalized_random_seed42_gpt_5_cot_0_20251027_102147_predictions.csv
            # parts: ['globem', 'compass', '4shot', 'generalized', 'random', 'seed42', 'gpt', '5', 'cot', '0', '20251027', '102147']
            shot_type = parts[2] # e.g., '4shot', 'zeroshot'
            strategy = parts[3] # e.g., 'generalized', 'hybrid'
            seed = parts[5] # e.g., 'seed42'
            model_type = parts[6].upper() # e.g., 'GPT'
            model_version = parts[7] # e.g., '5'
            cot_value = parts[8] # e.g., '0'

            # Format components for printing
            sensor_type = parts[1].capitalize() # e.g., 'compass', 'structured'
            formatted_shot_type = shot_type.replace('shot', '-shot').capitalize()
            formatted_strategy = strategy.capitalize()
            formatted_seed = seed.replace('seed', 'Seed ').capitalize() # 'seed42' -> 'Seed 42'
            formatted_model = f"{model_type}-{model_version}"

            print_header = (
                f"{sensor_type} - {formatted_shot_type} - {formatted_strategy} - Random - "
                f"{formatted_seed} - {formatted_model} - {cot_value.upper()}"
            )
            print(print_header + "\n")

            # Load data
            file_path = os.path.join('results', file)
            df = pd.read_csv(file_path)
            efficiency_file_path = file_path.replace('_predictions.csv', '_efficiency.csv')
            efficiency_df = pd.read_csv(efficiency_file_path)
            
            import json # Required for loading metadata.json
            import numpy as np # Required for calculating mean of timings

            # Extract the experiment name (e.g., 'globem_compass_4shot_generalized_random_seed42')
            # from the base_name (e.g., 'globem_compass_4shot_generalized_random_seed42_gpt_5_cot_0_20251027_102147')
            # The rule is to parse up to 'seed{number or None}'.
            
            extracted_experiment_name_parts = []
            found_seed_part = False
            for part in parts:
                extracted_experiment_name_parts.append(part)
                if part.startswith('seed'):
                    found_seed_part = True
                    break
            
            if found_seed_part:
                extracted_experiment_name = '_'.join(extracted_experiment_name_parts)
            else:
                # Fallback if 'seed' part is not found, assume the experiment name ends before the model type
                # (e.g., 'gpt', which is parts[6])
                try:
                    extracted_experiment_name = '_'.join(parts[:parts.index(model_type.lower())])
                except ValueError:
                    extracted_experiment_name = base_name # Last resort, use the whole base_name

            # Construct the path to the metadata.json file for prompt generation timings
            prompt_file_path = os.path.join('saved_prompts', extracted_experiment_name, 'metadata.json')
            
            # Load the prompt metadata and prepare the dictionary for print_prompt_generation_time
            prompts_dict = {} # Initialize in case of error
            try:
                with open(prompt_file_path, 'r') as f:
                    metadata = json.load(f)
                
                # Calculate average timings from the lists in metadata.json
                # The print_prompt_generation_time function expects single float values.
                step_timings = metadata.get('step_timings', {})
                
                prompts_dict = {
                    "data_sampling": np.mean(step_timings.get('data_sampling', [0.0])),
                    "data_sampling_std": np.std(step_timings.get('data_sampling', [0.0])),
                    "icl_selection": np.mean(step_timings.get('icl_selection', [0.0])),
                    "icl_selection_std": np.std(step_timings.get('icl_selection', [0.0])),
                    "prompt_building": np.mean(step_timings.get('prompt_building', [0.0])),
                    "prompt_building_std": np.std(step_timings.get('prompt_building', [0.0])),
                }
            except:
                print(f"An unexpected error occurred while loading prompt metadata from '{prompt_file_path}': {e}. Skipping prompt generation time display.")
                # Set default values to 0.0 for any other error
                prompts_dict = {
                    "data_sampling": 0.0,
                    "data_sampling_std": 0.0,
                    "icl_selection": 0.0,
                    "icl_selection_std": 0.0,
                    "prompt_building": 0.0,
                    "prompt_building_std": 0.0,
                }

            # Define a helper function to calculate and print metrics
            def print_metrics(label_name, y_real, y_pred):
                accuracy = accuracy_score(y_real, y_pred)
                # Use average='macro' as per instruction for F1, Precision, Recall
                f1 = f1_score(y_real, y_pred, average='macro', zero_division=0)
                precision = precision_score(y_real, y_pred, average='macro', zero_division=0)
                recall = recall_score(y_real, y_pred, average='macro', zero_division=0)

                print(label_name)
                print(f"Accuracy: {accuracy:.4f}")
                print(f"F1-Score (macro): {f1:.4f}")
                print(f"Precision (macro): {precision:.4f}")
                print(f"Recall (macro): {recall:.4f}\n")

            def print_resource_usage(df):
                print("Resource Usage:")
                print(f"Latency (s): {df.loc[df['Metric'] == 'Avg Latency (s)', 'Value'].values[0]:.2f} ± {df.loc[df['Metric'] == 'Std Latency (s)', 'Value'].values[0]:.2f} seconds, Total: {df.loc[df['Metric'] == 'Total Latency (s)', 'Value'].values[0]:.2f} seconds")
                print(f"Cost (per sample: ${df.loc[df['Metric'] == 'Cost per Sample ($)', 'Value'].values[0]:.4f} ± ${df.loc[df['Metric'] == 'Std Cost per Sample ($)', 'Value'].values[0]:.4f}, Total: ${df.loc[df['Metric'] == 'Total Cost ($)', 'Value'].values[0]:.4f}")
                print(f"Prompt Tokens: {df.loc[df['Metric'] == 'Prompt Tokens (Avg)', 'Value'].values[0]:,.0f} ± {df.loc[df['Metric'] == 'Prompt Tokens (Std)', 'Value'].values[0]:,.0f}, Total: {df.loc[df['Metric'] == 'Prompt Tokens (Total)', 'Value'].values[0]:,.0f}")
                print(f"Completion Tokens: {df.loc[df['Metric'] == 'Completion Tokens (Avg)', 'Value'].values[0]:,.0f} ± {df.loc[df['Metric'] == 'Completion Tokens (Std)', 'Value'].values[0]:,.0f}, Total: {df.loc[df['Metric'] == 'Completion Tokens (Total)', 'Value'].values[0]:,.0f}")
                print(f"Tokens per Request: {df.loc[df['Metric'] == 'Avg Tokens per Request', 'Value'].values[0]:,.0f} ± {df.loc[df['Metric'] == 'Std Tokens per Request', 'Value'].values[0]:,.0f}")
                print(f"Throughput: Tokens/Second: {df.loc[df['Metric'] == 'Tokens/Second', 'Value'].values[0]:.2f}, Samples/Minute: {df.loc[df['Metric'] == 'Samples/Minute', 'Value'].values[0]:.2f}")
                print(f"Total Tokens: {df.loc[df['Metric'] == 'Total Tokens', 'Value'].values[0]:,.0f}")


            def print_prompt_generation_time(prompts_dict):
                print("\nPrompt Generation Time:")
                print(f"Data Sampling: {prompts_dict['data_sampling']:.4f}s ± {prompts_dict['data_sampling_std']:.4f}s")
                print(f"ICL Selection: {prompts_dict['icl_selection']:.4f}s ± {prompts_dict['icl_selection_std']:.4f}s")
                print(f"Prompt Building: {prompts_dict['prompt_building']:.4f}s ± {prompts_dict['prompt_building_std']:.4f}s")
                
                # Calculate and print metrics for Depression
            print_metrics("Depression", df['y_dep_real'], df['y_dep_pred'])

            # Calculate and print metrics for Anxiety
            print_metrics("Anxiety", df['y_anx_real'], df['y_anx_pred'])

            # Print resource usage
            print_resource_usage(efficiency_df)
            print_prompt_generation_time(prompts_dict)

        except IndexError:
            print(f"Warning: Could not parse filename '{file}' due to unexpected format. Skipping.\n")
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found. Skipping.\n")
        except KeyError as e:
            print(f"Error: Missing expected column '{e}' in '{file}'. Skipping.\n")
        except Exception as e:
            print(f"An unexpected error occurred while processing '{file}': {e}. Skipping.\n")


