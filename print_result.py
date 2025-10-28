import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json  # Required for loading metadata.json
import numpy as np  # Required for calculating mean of timings

# List to store results for all files
all_results = []

# Define helper functions outside the loop for efficiency
def calculate_metrics(y_real, y_pred):
    """Calculates accuracy, F1, precision, and recall scores."""
    accuracy = accuracy_score(y_real, y_pred)
    f1 = f1_score(y_real, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_real, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_real, y_pred, average='macro', zero_division=0)
    return accuracy, f1, precision, recall

def print_metrics(label_name, y_real, y_pred):
    """Prints classification metrics for a given label."""
    accuracy, f1, precision, recall = calculate_metrics(y_real, y_pred)
    print(label_name)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (macro): {f1:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}\n")

def get_resource_usage_value(df, metric_name):
    """Helper to safely get a value from efficiency_df, returning 0.0 if not found."""
    if metric_name in df['Metric'].values:
        return df.loc[df['Metric'] == metric_name, 'Value'].values[0]
    return 0.0

def print_resource_usage(df):
    """Prints resource usage metrics."""
    print("Resource Usage:")
    avg_latency = get_resource_usage_value(df, 'Avg Latency (s)')
    std_latency = get_resource_usage_value(df, 'Std Latency (s)')
    total_latency = get_resource_usage_value(df, 'Total Latency (s)')
    print(f"Latency (s): {avg_latency:.2f} ± {std_latency:.2f} seconds, Total: {total_latency:.2f} seconds")

    cost_per_sample = get_resource_usage_value(df, 'Cost per Sample ($)')
    std_cost_per_sample = get_resource_usage_value(df, 'Std Cost per Sample ($)')
    total_cost = get_resource_usage_value(df, 'Total Cost ($)')
    print(f"Cost (per sample: ${cost_per_sample:.4f} ± ${std_cost_per_sample:.4f}, Total: ${total_cost:.4f}")

    prompt_tokens_avg = get_resource_usage_value(df, 'Prompt Tokens (Avg)')
    prompt_tokens_std = get_resource_usage_value(df, 'Prompt Tokens (Std)')
    prompt_tokens_total = get_resource_usage_value(df, 'Prompt Tokens (Total)')
    print(f"Prompt Tokens: {prompt_tokens_avg:,.0f} ± {prompt_tokens_std:,.0f}, Total: {prompt_tokens_total:,.0f}")

    completion_tokens_avg = get_resource_usage_value(df, 'Completion Tokens (Avg)')
    completion_tokens_std = get_resource_usage_value(df, 'Completion Tokens (Std)')
    completion_tokens_total = get_resource_usage_value(df, 'Completion Tokens (Total)')
    print(f"Completion Tokens: {completion_tokens_avg:,.0f} ± {completion_tokens_std:,.0f}, Total: {completion_tokens_total:,.0f}")

    avg_tokens_per_request = get_resource_usage_value(df, 'Avg Tokens per Request')
    std_tokens_per_request = get_resource_usage_value(df, 'Std Tokens per Request')
    print(f"Tokens per Request: {avg_tokens_per_request:,.0f} ± {std_tokens_per_request:,.0f}")

    tokens_per_second = get_resource_usage_value(df, 'Tokens/Second')
    samples_per_minute = get_resource_usage_value(df, 'Samples/Minute')
    print(f"Throughput: Tokens/Second: {tokens_per_second:.2f}, Samples/Minute: {samples_per_minute:.2f}")

    total_tokens = get_resource_usage_value(df, 'Total Tokens')
    print(f"Total Tokens: {total_tokens:,.0f}")

def print_prompt_generation_time(prompts_dict):
    """Prints prompt generation timing metrics."""
    print("\nPrompt Generation Time:")
    print(f"Data Loading: {prompts_dict['loading']:.4f}s ± {prompts_dict['loading_std']:.4f}s")
    print(f"Test Sampling: {prompts_dict['test_sampling']:.4f}s ± {prompts_dict['test_sampling_std']:.4f}s")
    print(f"Feature Engineering: {prompts_dict['feature_engineering']:.4f}s ± {prompts_dict['feature_engineering_std']:.4f}s")
    print(f"ICL Selection: {prompts_dict['icl_selection']:.4f}s ± {prompts_dict['icl_selection_std']:.4f}s")
    print(f"Prompt Building: {prompts_dict['prompt_building']:.4f}s ± {prompts_dict['prompt_building_std']:.4f}s")


files = os.listdir('results')

for file in files:
    if file.endswith('_predictions.csv'):
        print(f"example - {file}\n")

        # Parse filename for display
        base_name = file.replace('_predictions.csv', '')
        parts = base_name.split('_')

        current_result = {} # Dictionary to store results for the current file

        try:
            # Example filename structure:
            # globem_compass_4shot_generalized_random_seed42_gpt_5_cot_0_20251027_102147_predictions.csv
            # parts: ['globem', 'compass', '4shot', 'generalized', 'random', 'seed42', 'gpt', '5', 'cot', '0', '20251027', '102147']
            dataset_name = parts[0].upper() # e.g., 'GLOBEM'
            sensor_type = parts[1].capitalize() # e.g., 'Compass', 'Structured'
            shot_type = parts[2] # e.g., '4shot', 'zeroshot'
            strategy = parts[3] # e.g., 'generalized', 'hybrid'
            seed = parts[5] # e.g., 'seed42'
            model_type = parts[6].upper() # e.g., 'GPT'
            model_version = parts[7] # e.g., '5'
            reasoning_method = parts[4] # e.g., '0'

            # Format components for printing and CSV
            formatted_shot_type = shot_type.replace('shot', '-shot').capitalize()
            formatted_strategy = strategy.capitalize()
            formatted_seed = seed.replace('seed', 'Seed ').capitalize() # 'seed42' -> 'Seed 42'
            formatted_model = f"{model_type}-{model_version}"
            # reasoning_method = 'CoT' if cot_value == '1' else 'DO'
            icl_strategy = 'ZeroShot' if shot_type == 'zeroshot' else formatted_strategy

            print_header = (
                f"{sensor_type} - {formatted_shot_type} - {formatted_strategy} - Random - "
                f"{formatted_seed} - {formatted_model} - {reasoning_method}"
            )
            print(print_header + "\n")

            # Load data
            file_path = os.path.join('results', file)
            df = pd.read_csv(file_path)
            efficiency_file_path = file_path.replace('_predictions.csv', '_efficiency.csv')
            efficiency_df = pd.read_csv(efficiency_file_path)
            
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
            prompts_dict = {
                "loading": 0.0, "loading_std": 0.0,
                "test_sampling": 0.0, "test_sampling_std": 0.0,
                "feature_engineering": 0.0, "feature_engineering_std": 0.0,
                "icl_selection": 0.0, "icl_selection_std": 0.0,
                "prompt_building": 0.0, "prompt_building_std": 0.0,
            } # Initialize with default values
            try:
                with open(prompt_file_path, 'r') as f:
                    metadata = json.load(f)
                
                # Calculate average timings from the lists in metadata.json
                step_timings = metadata.get('step_timings', {})
                
                for key in prompts_dict.keys():
                    if key.endswith('_std'):
                        base_key = key.replace('_std', '')
                        prompts_dict[key] = np.std(step_timings.get(base_key, [0.0]))
                    else:
                        prompts_dict[key] = np.mean(step_timings.get(key, [0.0]))
            except FileNotFoundError:
                print(f"Warning: Prompt metadata file '{prompt_file_path}' not found. Skipping prompt generation time display for this file.")
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from '{prompt_file_path}'. Skipping prompt generation time display for this file.")
            except Exception as e:
                print(f"An unexpected error occurred while loading prompt metadata from '{prompt_file_path}': {e}. Skipping prompt generation time display for this file.")

            # Calculate metrics for Depression
            dep_accuracy, dep_f1, dep_precision, dep_recall = calculate_metrics(df['y_dep_real'], df['y_dep_pred'])
            print_metrics("Depression", df['y_dep_real'], df['y_dep_pred'])

            # Calculate metrics for Anxiety
            anx_accuracy, anx_f1, anx_precision, anx_recall = calculate_metrics(df['y_anx_real'], df['y_anx_pred'])
            print_metrics("Anxiety", df['y_anx_real'], df['y_anx_pred'])

            # Print resource usage
            print_resource_usage(efficiency_df)
            print_prompt_generation_time(prompts_dict)

            # Populate current_result dictionary for CSV
            current_result['Dataset'] = dataset_name
            current_result['Model'] = formatted_model
            current_result['Features'] = sensor_type
            current_result['ICL Strategy'] = icl_strategy
            current_result['Reasoning'] = reasoning_method

            current_result['Accuracy (Depression)'] = np.round(dep_accuracy, 4)
            current_result['Macro F1 (Depression)'] = np.round(dep_f1, 4)
            current_result['Precision (Depression)'] = np.round(dep_precision, 4)
            current_result['Recall (Depression)'] = np.round(dep_recall, 4)

            current_result['Accuracy (Anxiety)'] = np.round(anx_accuracy, 4)
            current_result['Macro F1 (Anxiety)'] = np.round(anx_f1, 4)
            current_result['Precision (Anxiety)'] = np.round(anx_precision, 4)
            current_result['Recall (Anxiety)'] = np.round(anx_recall, 4)

            # Resource Usage
            avg_latency = get_resource_usage_value(efficiency_df, 'Avg Latency (s)')
            std_latency = get_resource_usage_value(efficiency_df, 'Std Latency (s)')
            current_result['Latency (sec)'] = f"{avg_latency:.2f}±{std_latency:.2f}"

            cost_per_sample = get_resource_usage_value(efficiency_df, 'Cost per Sample ($)')
            std_cost_per_sample = get_resource_usage_value(efficiency_df, 'Std Cost per Sample ($)')
            current_result['Cost ($)'] = f"{cost_per_sample:.4f}±{std_cost_per_sample:.4f}"

            current_result['Input Token (Avg)'] = np.round(get_resource_usage_value(efficiency_df, 'Prompt Tokens (Avg)'), 4)
            current_result['Output Token (Avg)'] = np.round(get_resource_usage_value(efficiency_df, 'Completion Tokens (Avg)'), 4)  
            current_result['Throughput (Tokens/Sec)'] = np.round(get_resource_usage_value(efficiency_df, 'Tokens/Second'), 4)
            current_result['Throughput (Samples/Minute)'] = np.round(get_resource_usage_value(efficiency_df, 'Samples/Minute'), 4)  
            current_result['Total Token'] = np.round(get_resource_usage_value(efficiency_df, 'Total Tokens'), 0)
            current_result['Total Latency (sec) - Experiment'] = np.round(get_resource_usage_value(efficiency_df, 'Total Latency (s)'), 4)
            current_result['Total Cost ($) - Experiment'] = np.round(get_resource_usage_value(efficiency_df, 'Total Cost ($)'), 4)
            current_result['Total Input Token - Experiment'] = np.round(get_resource_usage_value(efficiency_df, 'Prompt Tokens (Total)'), 0)
            current_result['Total Output Token - Experiment'] = np.round(get_resource_usage_value(efficiency_df, 'Completion Tokens (Total)'), 0)

            # Prompt Generation Time
            current_result['Total Data Prep Time (sec)'] = np.round(prompts_dict['loading'] + prompts_dict['test_sampling'] + prompts_dict['feature_engineering'], 4)
            current_result['Mean ICL Sampling (sec)'] = np.round(prompts_dict['icl_selection'], 4)

            all_results.append(current_result)

        except IndexError:
            print(f"Warning: Could not parse filename '{file}' due to unexpected format. Skipping.\n")
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found. Skipping.\n")
        except KeyError as e:
            print(f"Error: Missing expected column '{e}' in '{file}'. Skipping.\n")
        except Exception as e:
            print(f"An unexpected error occurred while processing '{file}': {e}. Skipping.\n")

# After the loop, save all results to a CSV file
if all_results:
    results_df = pd.DataFrame(all_results)
    
    # Define the desired column order as per the example
    column_order = [
        'Dataset', 'Model', 'Features', 'ICL Strategy', 'Reasoning',
        'Accuracy (Depression)', 'Macro F1 (Depression)', 'Precision (Depression)', 'Recall (Depression)',
        'Accuracy (Anxiety)', 'Macro F1 (Anxiety)', 'Precision (Anxiety)', 'Recall (Anxiety)',
        'Latency (sec)', 'Cost ($)', 'Input Token (Avg)', 'Total Data Prep Time (sec)', 'Mean ICL Sampling (sec)',
        'Throughput (Tokens/Sec)', 'Throughput (Samples/Minute)', 'Output Token (Avg)', 'Total Token',
        'Total Latency (sec) - Experiment', 'Total Cost ($) - Experiment', 'Total Input Token - Experiment', 'Total Output Token - Experiment'
    ]
    
    # Reorder columns to match the example and save
    results_df = results_df[column_order]
    output_csv_path = 'all_experiment_results.csv'
    results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig') # Use utf-8-sig for Excel compatibility with Korean characters
    print(f"\nAll experiment results saved to '{output_csv_path}'")
else:
    print("\nNo results to save.")
