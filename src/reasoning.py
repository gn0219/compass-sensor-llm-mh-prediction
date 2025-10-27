"""
Reasoning Strategies Module - Advanced LLM reasoning methods.

Handles:
- Response parsing (extracting JSON predictions)
- Self-Consistency (multiple samples + majority vote)
- Tree-of-Thoughts (exploring multiple reasoning paths)
- Chain-of-Thought (via prompt engineering, no special logic needed)
- Direct prediction (via prompt engineering, no special logic needed)
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
    
    def call_llm(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000,
                seed: Optional[int] = None) -> Tuple[Optional[str], Dict]:
        """Call LLM API via client."""
        return self.client.call_api(prompt, temperature, max_tokens, seed)
    
    def parse_response(self, response_text: str) -> Optional[Dict]:
        """Parse LLM response to extract prediction and reasoning."""
        if not response_text or len(response_text.strip()) == 0:
            print("❌ Error: Empty response from LLM")
            return None
        
        try:
            # Extract JSON from response
            if '```json' in response_text:
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                json_str = response_text[start:end].strip()
            elif '```' in response_text:
                start = response_text.find('```') + 3
                end = response_text.find('```', start)
                json_str = response_text[start:end].strip()
            else:
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                json_str = response_text[start:end] if start != -1 and end > start else response_text
            
            if not json_str or json_str == response_text and '{' not in json_str:
                print(f"❌ Error: No JSON found in response")
                print(f"First 500 chars of response:\n{response_text[:500]}")
                return None
            
            result = json.loads(json_str)
            
            # Validate structure
            if 'Prediction' not in result:
                print("⚠️  Warning: 'Prediction' key not found in response")
                print(f"Available keys: {list(result.keys())}")
                return None
            
            if 'Anxiety' not in result['Prediction'] or 'Depression' not in result['Prediction']:
                print("⚠️  Warning: Missing Anxiety or Depression in Prediction")
                print(f"Prediction keys: {list(result['Prediction'].keys())}")
                return None
            
            # Normalize to binary
            result['Prediction']['Anxiety_binary'] = 1 if 'high' in result['Prediction']['Anxiety'].lower() else 0
            result['Prediction']['Depression_binary'] = 1 if 'high' in result['Prediction']['Depression'].lower() else 0
            
            return result
        
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Attempted to parse: {json_str[:200] if 'json_str' in locals() else 'N/A'}...")
            print(f"\nFull response (first 500 chars):\n{response_text[:500]}")
            print(f"\nFull response (last 500 chars):\n{response_text[-500:]}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error parsing response: {e}")
            print(f"Response length: {len(response_text)}")
            return None
    
    # ========================================================================
    # REASONING STRATEGY: Self-Feedback (Self-Evolving)
    # ========================================================================
    def predict_with_self_feedback(self, prompt: str, max_iterations: int = 3,
                                   seed: Optional[int] = None) -> Tuple[Optional[Dict], List[Dict]]:
        """
        Self-Feedback Reasoning Strategy.
        
        Iteratively refines predictions based on self-assessed difficulty.
        - Easy: Stop after first iteration
        - Medium/Hard: Continue refining up to max_iterations
        
        Args:
            prompt: Complete prompt string
            max_iterations: Maximum number of refinement iterations (default: 3)
            seed: Optional seed for reproducibility
        
        Returns:
            Tuple of (final_prediction, all_iterations)
        """
        iterations = []
        current_prompt = prompt
        
        print(f"  [Self-Feedback] Starting iterative reasoning (max {max_iterations} iterations)...")
        
        for iteration in range(max_iterations):
            print(f"  [Iteration {iteration + 1}] Generating prediction...")
            
            # Call LLM with increased max_tokens for refinement iterations
            max_tokens = 6000 if iteration > 0 else 3200
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
            
            # Add metadata
            parsed['usage'] = usage_info
            parsed['iteration'] = iteration + 1
            iterations.append(parsed)
            
            # Extract difficulty assessment
            difficulty = parsed.get('Difficulty', 'Hard')
            if isinstance(difficulty, dict):
                difficulty = difficulty.get('value', 'Hard')
            
            confidence_anx = parsed.get('Confidence', {}).get('Anxiety', 'Low')
            confidence_dep = parsed.get('Confidence', {}).get('Depression', 'Low')
            
            print(f"  [Iteration {iteration + 1}] Difficulty: {difficulty} | "
                  f"Confidence - Anx: {confidence_anx}, Dep: {confidence_dep}")
            
            # If Easy difficulty, stop iteration
            if difficulty == 'Easy':
                print(f"  [Self-Feedback] Stopping - prediction marked as Easy")
                break
            
            # If this is the last iteration, stop
            if iteration >= max_iterations - 1:
                print(f"  [Self-Feedback] Reached maximum iterations")
                break
            
            # Prepare refinement prompt for next iteration
            print(f"  [Self-Feedback] Difficulty {difficulty} - preparing refinement prompt...")
            refinement_prompt = self._build_refinement_prompt(prompt, response_text, difficulty)
            current_prompt = refinement_prompt
        
        if len(iterations) == 0:
            print("  [ERROR] All iterations failed")
            return None, []
        
        # Use the last iteration as final prediction
        final_iteration = iterations[-1]
        
        # Build final prediction dict
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
        
        print(f"  [Self-Feedback] Completed with {len(iterations)} iteration(s)")
        
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
