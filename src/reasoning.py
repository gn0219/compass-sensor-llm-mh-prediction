"""
Reasoning Strategies Module - Advanced LLM reasoning methods.

Handles:
- Response parsing (extracting JSON predictions)
- Self-Feedback (iterative refinement)
- Chain-of-Thought
- Direct prediction
"""

import json
from typing import Dict, List, Optional, Tuple
from .llm_client import LLMClient


class LLMReasoner:
    """Handles LLM reasoning strategies and response parsing."""
    
    def __init__(self, model: str = "gpt-5-nano"):
        """Initialize LLM reasoner with client."""
        self.client = LLMClient(model=model)
        self.model = model
    
    def call_llm(self, prompt: str, temperature: float = 0.7, max_tokens: int = 6000,
                seed: Optional[int] = None) -> Tuple[Optional[str], Dict]:
        """Call LLM API via client."""
        return self.client.call_api(prompt, temperature, max_tokens, seed)
    
    def parse_response(self, response_text: str) -> Optional[Dict]:
        """Parse LLM response to extract prediction and reasoning."""
        if not response_text or len(response_text.strip()) == 0:
            print("[Error] Empty response from LLM")
            return None
        
        try:
            # Extract JSON from response
            if '```json' in response_text:
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                if end == -1:
                    # No closing ```, response was truncated
                    print("[Warning] Response appears truncated (no closing ```)")
                    json_str = response_text[start:].strip()
                else:
                    json_str = response_text[start:end].strip()
            elif '```' in response_text:
                start = response_text.find('```') + 3
                end = response_text.find('```', start)
                if end == -1:
                    print("[Warning] Response appears truncated (no closing ```)")
                    json_str = response_text[start:].strip()
                else:
                    json_str = response_text[start:end].strip()
            else:
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                json_str = response_text[start:end] if start != -1 and end > start else response_text
            
            if not json_str or json_str == response_text and '{' not in json_str:
                print(f"[Error] No JSON found in response")
                print(f"First 500 chars of response:\n{response_text[:500]}")
                return None
            
            result = json.loads(json_str)
            
            # Handle both 'Prediction' and 'Refined_Prediction' (for self-feedback iterations)
            if 'Refined_Prediction' in result and 'Prediction' not in result:
                result['Prediction'] = result['Refined_Prediction']
            
            # Validate structure
            if 'Prediction' not in result:
                print("[Warning] 'Prediction' key not found in response")
                print(f"Available keys: {list(result.keys())}")
                return None
            
            if 'Anxiety' not in result['Prediction'] or 'Depression' not in result['Prediction']:
                print("[Warning] Missing Anxiety or Depression in Prediction")
                print(f"Prediction keys: {list(result['Prediction'].keys())}")
                return None
            
            # Normalize to binary
            result['Prediction']['Anxiety_binary'] = 1 if 'high' in result['Prediction']['Anxiety'].lower() else 0
            result['Prediction']['Depression_binary'] = 1 if 'high' in result['Prediction']['Depression'].lower() else 0
            
            return result
        
        except json.JSONDecodeError as e:
            # Try to recover from duplicate keys or malformed JSON
            error_msg = str(e)
            print(f"[Warning] JSON parsing error: {e}")
            
            try:
                # Strategy 1: Handle "Extra data" error - extract only the first valid JSON object
                if "Extra data" in error_msg:
                    print(f"[Attempting recovery] Extracting first valid JSON object...")
                    # Find the position of the error
                    # Try to parse up to the error position
                    import re
                    match = re.search(r'char (\d+)', error_msg)
                    if match:
                        error_pos = int(match.group(1))
                        # Take everything up to the error position
                        truncated_json = json_str[:error_pos]
                        # Find the last complete '}' 
                        last_brace = truncated_json.rfind('}')
                        if last_brace != -1:
                            truncated_json = json_str[:last_brace + 1]
                            result = json.loads(truncated_json)
                            print(f"[Recovery] Successfully extracted valid JSON (truncated at char {last_brace})")
                        else:
                            raise ValueError("Could not find closing brace")
                    else:
                        raise ValueError("Could not extract error position")
                else:
                    # Strategy 2: For other errors, try to clean and re-parse
                    print(f"[Attempting recovery] Finding valid JSON boundaries...")
                    # Find the first '{' and last '}' to extract just the JSON object
                    first_brace = json_str.find('{')
                    last_brace = json_str.rfind('}')
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                        cleaned_json = json_str[first_brace:last_brace+1]
                        # Python's json.loads automatically uses the last value for duplicate keys
                        result = json.loads(cleaned_json)
                        print(f"[Recovery] Successfully parsed JSON with boundaries")
                    else:
                        raise ValueError("Could not find valid JSON boundaries")
                
                print(f"[Recovery] Successfully recovered from JSON error")
                
                # Apply the same validation as above
                if 'Refined_Prediction' in result and 'Prediction' not in result:
                    result['Prediction'] = result['Refined_Prediction']
                
                if 'Prediction' not in result:
                    print("[Warning] 'Prediction' key not found after recovery")
                    return None
                
                if 'Anxiety' not in result['Prediction'] or 'Depression' not in result['Prediction']:
                    print("[Warning] Missing Anxiety or Depression after recovery")
                    return None
                
                result['Prediction']['Anxiety_binary'] = 1 if 'high' in result['Prediction']['Anxiety'].lower() else 0
                result['Prediction']['Depression_binary'] = 1 if 'high' in result['Prediction']['Depression'].lower() else 0
                
                return result
                
            except Exception as recovery_error:
                print(f"[Error] Recovery failed: {recovery_error}")
                print(f"Attempted to parse: {json_str[:200] if 'json_str' in locals() else 'N/A'}...")
                print(f"\nFull response (first 500 chars):\n{response_text[:500]}")
                print(f"\nFull response (last 500 chars):\n{response_text[-500:]}")
                return None
        except Exception as e:
            print(f"[Error] Unexpected error parsing response: {e}")
            print(f"Response length: {len(response_text)}")
            return None
    
    # ========================================================================
    # REASONING STRATEGY: Self-Feedback (Self-Evolving)
    # ========================================================================
    def predict_with_self_feedback(self, prompt: str, max_iterations: int = 3,
                                   temperature: float = 0.7, seed: Optional[int] = None) -> Tuple[Optional[Dict], List[Dict]]:
        """
        Self-Feedback Reasoning Strategy.
        
        Iteratively refines predictions based on self-assessed difficulty.
        - Easy: Stop after first iteration
        - Medium/Hard: Continue refining up to max_iterations
        
        Args:
            prompt: Complete prompt string
            max_iterations: Maximum number of refinement iterations (default: 3)
            temperature: LLM temperature (default: 0.7)
            seed: Optional seed for reproducibility
        
        Returns:
            Tuple of (final_prediction, all_iterations)
        """
        iterations = []
        current_prompt = prompt
        
        print(f"  [Self-Feedback] Starting iterative reasoning (max {max_iterations} iterations)...")
        
        for iteration in range(max_iterations):
            print(f"  [Iteration {iteration + 1}/{max_iterations}] Generating prediction...")
            
            # Call LLM with increased max_tokens for refinement iterations
            max_tokens = 12000 if iteration > 0 else 6000
            response_text, usage_info = self.call_llm(
                current_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed
            )
            
            if response_text is None:
                print(f"  [ERROR] Iteration {iteration + 1} failed")
                break
            
            # Parse response
            parsed = self.parse_response(response_text)
            if parsed is None:
                print(f"  [ERROR] Could not parse iteration {iteration + 1}")
                break
            
            # Add metadata and iteration info
            parsed['usage'] = usage_info
            parsed['iteration'] = iteration + 1
            
            # Store predictions from this iteration with iteration suffix
            pred = parsed.get('Refined_Prediction') or parsed.get('Prediction')
            if pred:
                parsed[f'pred_iteration_{iteration + 1}'] = {
                    'Anxiety': pred.get('Anxiety'),
                    'Depression': pred.get('Depression'),
                    'Anxiety_binary': 1 if 'high' in str(pred.get('Anxiety', '')).lower() else 0,
                    'Depression_binary': 1 if 'high' in str(pred.get('Depression', '')).lower() else 0
                }
            
            conf = parsed.get('Confidence', {})
            if conf:
                parsed[f'conf_iteration_{iteration + 1}'] = {
                    'Anxiety': conf.get('Anxiety', 'Low'),
                    'Depression': conf.get('Depression', 'Low')
                }
            
            iterations.append(parsed)
            
            # Extract difficulty assessment
            difficulty = parsed.get('Difficulty', 'Hard')
            if isinstance(difficulty, dict):
                difficulty = difficulty.get('value', 'Hard')
            
            confidence_anx = conf.get('Anxiety', 'Low')
            confidence_dep = conf.get('Depression', 'Low')
            
            # Get predictions for logging
            pred_anx = pred.get('Anxiety', 'Unknown') if pred else 'Unknown'
            pred_dep = pred.get('Depression', 'Unknown') if pred else 'Unknown'
            
            print(f"  [Iteration {iteration + 1}/{max_iterations}] Complete")
            print(f"    Difficulty: {difficulty}")
            print(f"    Predictions: Anxiety={pred_anx}, Depression={pred_dep}")
            print(f"    Confidence: Anxiety={confidence_anx}, Depression={confidence_dep}")
            
            # If Easy difficulty, stop iteration
            if difficulty == 'Easy':
                print(f"  [Self-Feedback] Stopping - prediction marked as Easy")
                break
            
            # If this is the last iteration, stop
            if iteration >= max_iterations - 1:
                print(f"  [Self-Feedback] Reached maximum iterations ({max_iterations})")
                break
            
            # Prepare refinement prompt for next iteration
            print(f"  [Self-Feedback] Difficulty={difficulty} - preparing refinement for iteration {iteration + 2}...")
            refinement_prompt = self._build_refinement_prompt(prompt, response_text, difficulty)
            current_prompt = refinement_prompt
        
        if len(iterations) == 0:
            print("  [ERROR] All iterations failed")
            return None, []
        
        # Use the last iteration as final prediction
        final_iteration = iterations[-1]
        
        # Build final prediction dict with all iteration info
        final_pred = final_iteration.get('Refined_Prediction') or final_iteration.get('Prediction')
        
        final_prediction = {
            'Prediction': final_pred,
            'Reasoning': {
                'method': 'self_feedback',
                'total_iterations': len(iterations),
                'final_difficulty': final_iteration.get('Difficulty', 'Unknown'),
                'all_iterations': iterations
            }
        }
        
        # Add iteration-specific predictions and confidences to top level for easy access
        for i, iter_result in enumerate(iterations):
            iter_num = i + 1
            # Extract predictions
            pred_key = f'pred_iteration_{iter_num}'
            if pred_key in iter_result:
                final_prediction[pred_key] = iter_result[pred_key]
            
            # Extract confidences
            conf_key = f'conf_iteration_{iter_num}'
            if conf_key in iter_result:
                final_prediction[conf_key] = iter_result[conf_key]
            
            # Add difficulty
            final_prediction[f'difficulty_iteration_{iter_num}'] = iter_result.get('Difficulty', 'Unknown')
        
        print(f"  [Self-Feedback] Completed with {len(iterations)} iteration(s)")
        print(f"  [Self-Feedback] Final: Anxiety={final_pred.get('Anxiety')}, Depression={final_pred.get('Depression')}")
        
        return final_prediction, iterations
    
    def _build_refinement_prompt(self, original_prompt: str, previous_response: str, difficulty: str) -> str:
        """Build a refinement prompt for the next iteration."""
        refinement_instruction = f"""
You previously made a prediction that was marked as {difficulty}. Here is your previous analysis:

---
{previous_response}
---

Now, conduct a deeper analysis focusing on the challenges and areas identified above. Consider:
1. Re-examine the conflicting signals or ambiguous patterns
2. Look for subtle indicators that might have been missed
3. Consider the temporal dynamics and trends more carefully
4. Integrate insights from similar examples if provided

Please provide your refined response in the following JSON format:
{{
  "Refined_Prediction": {{
    "Anxiety": "Low Risk" or "High Risk",
    "Depression": "Low Risk" or "High Risk"
  }},
  "Confidence": {{
    "Anxiety": "High" or "Medium" or "Low",
    "Depression": "High" or "Medium" or "Low"
  }},
  "Refined_Analysis": "Explain how your deeper analysis led to this prediction",
  "Difficulty": "Easy" or "Medium" or "Hard",
  "Changes_from_Initial": "Describe what changed (if anything) and why"
}}
"""
        
        # Combine original data with refinement instruction
        return original_prompt + "\n\n" + refinement_instruction
    
    
    def get_usage_summary(self) -> Dict:
        """Get summary of API usage and costs."""
        return self.client.get_usage_summary()
