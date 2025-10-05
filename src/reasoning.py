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
    # REASONING STRATEGY: Self-Consistency
    # ========================================================================
    def predict_with_self_consistency(self, prompt: str, n_samples: int = 5, temperature: float = 0.9,
                                     seed: Optional[int] = None) -> Tuple[Optional[Dict], List[Dict]]:
        """
        Self-Consistency Reasoning Strategy.
        
        Sample multiple diverse reasoning paths and take a majority vote.
        Higher temperature encourages diverse reasoning paths.
        
        Args:
            prompt: Complete prompt string
            n_samples: Number of samples to generate (default: 5)
            temperature: Sampling temperature (default: 0.9 for diversity)
            seed: Optional base seed (each sample uses seed+i)
        
        Returns:
            Tuple of (final_prediction, all_samples)
        """
        samples = []
        
        print(f"  🔄 Generating {n_samples} diverse reasoning paths...")
        for i in range(n_samples):
            # Use different seed for each sample if base seed provided
            sample_seed = seed + i if seed is not None else None
            response_text, usage_info = self.call_llm(
                prompt, 
                temperature=temperature,
                seed=sample_seed
            )
            
            if response_text is None:
                print(f"     ⚠️  Sample {i+1} failed")
                continue
            
            parsed = self.parse_response(response_text)
            if parsed is not None:
                parsed['usage'] = usage_info
                samples.append(parsed)
                print(f"     ✓ Sample {i+1}: Anx={parsed['Prediction']['Anxiety_binary']}, "
                      f"Dep={parsed['Prediction']['Depression_binary']}")
        
        if len(samples) == 0:
            print("  ❌ All samples failed")
            return None, []
        
        # Majority vote
        anxiety_votes = [s['Prediction']['Anxiety_binary'] for s in samples]
        depression_votes = [s['Prediction']['Depression_binary'] for s in samples]
        
        anxiety_final = 1 if sum(anxiety_votes) > len(anxiety_votes) / 2 else 0
        depression_final = 1 if sum(depression_votes) > len(depression_votes) / 2 else 0
        
        print(f"  📊 Majority Vote: Anxiety {anxiety_votes} → {anxiety_final}")
        print(f"  📊 Majority Vote: Depression {depression_votes} → {depression_final}")
        
        final_prediction = {
            'Prediction': {
                'Anxiety': 'High Risk' if anxiety_final == 1 else 'Low Risk',
                'Depression': 'High Risk' if depression_final == 1 else 'Low Risk',
                'Anxiety_binary': anxiety_final,
                'Depression_binary': depression_final
            },
            'Reasoning': {
                'method': 'self_consistency',
                'n_samples': len(samples),
                'anxiety_votes': anxiety_votes,
                'depression_votes': depression_votes
            }
        }
        
        return final_prediction, samples
    
    # ========================================================================
    # REASONING STRATEGY: Tree-of-Thoughts (ToT)
    # ========================================================================
    def predict_with_tree_of_thoughts(self, prompt: str, depth: int = 2, breadth: int = 3,
                                     temperature: float = 0.7, seed: Optional[int] = None) -> Tuple[Optional[Dict], Dict]:
        """
        Tree-of-Thoughts Reasoning Strategy.
        
        Explores multiple reasoning paths in a tree structure, evaluating each path
        and selecting the best one.
        
        Args:
            prompt: Complete prompt string
            depth: How deep to explore the tree (default: 2)
            breadth: How many branches at each node (default: 3)
            temperature: Sampling temperature
            seed: Optional seed for reproducibility
        
        Returns:
            Tuple of (best_prediction, tree_structure)
        
        Note: This is a simplified implementation. Full ToT requires:
        - State evaluation at each node
        - Pruning of unpromising branches
        - Backtracking when needed
        """
        print(f"  🌳 Exploring Tree-of-Thoughts (depth={depth}, breadth={breadth})...")
        
        # TODO: Implement full Tree-of-Thoughts
        # For now, this is a placeholder that uses self-consistency as a fallback
        print("  ⚠️  Full ToT not yet implemented, using Self-Consistency as fallback")
        
        final_prediction, samples = self.predict_with_self_consistency(
            prompt, n_samples=breadth, temperature=temperature, seed=seed
        )
        
        tree_structure = {
            'method': 'tree_of_thoughts',
            'depth': depth,
            'breadth': breadth,
            'note': 'Simplified implementation using Self-Consistency'
        }
        
        return final_prediction, tree_structure
    
    def get_usage_summary(self) -> Dict:
        """Get summary of API usage and costs."""
        return self.client.get_usage_summary()
