"""
Prompt Template Manager - Loads and manages YAML-based prompt templates for mental health prediction.
"""

import yaml
from typing import Dict, List, Optional, Any
from pathlib import Path


class PromptManager:
    """Manages prompt templates from YAML configuration files."""
    
    def __init__(self, prompts_dir: str = None):
        """Initialize the prompt manager."""
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent / "prompts"
        
        self.prompts_dir = Path(prompts_dir)
        self.base_template = self._load_yaml("base.yaml")
        self.icl_template = self._load_yaml("icl.yaml")
        self.reasoning_template = self._load_yaml("reasoning.yaml")
    
    def _load_yaml(self, filename: str) -> Dict:
        """Load a YAML file and return as dictionary."""
        filepath = self.prompts_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Template file not found: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_task_instructions(self) -> str:
        """Construct the task instructions (Role + Task only)."""
        t = self.base_template
        parts = [t['structure']['title']]
        
        # Role section
        parts.extend([t['structure']['sections'][0]['header'], t['role']['description'].strip(), ""])
        
        # Task section
        parts.extend([t['structure']['sections'][1]['header'], t['task']['objective'].strip(), "",
                     t['task']['targets_header']])
        for target in t['task']['output_targets']:
            parts.append(f"- **{target['name']}**: {' or '.join(target['classes'])} (threshold: {target['threshold']})")
        parts.append("")
        
        return "\n".join(parts)
    
    def get_context_section(self) -> str:
        """Construct the context section only."""
        t = self.base_template
        ctx = t['context']
        parts = [t['structure']['sections'][2]['header'], ctx['background'].strip(), ""]
        
        # Clinical associations
        for condition, info in ctx['clinical_associations'].items():
            parts.append(f"**{info['title']}:**")
            for pattern in info['patterns']:
                parts.extend([f"- {pattern['indicator']}", f"  * {pattern['features']}", f"  * {pattern['patterns']}"])
            parts.append("")
        
        # Notes
        parts.append(t['notes_header'])
        for note in t['notes']:
            parts.append(f"- {note}")
        parts.append("")
        
        return "\n".join(parts)
    
    def get_base_prompt(self) -> str:
        """Construct the base prompt (Role + Task + Context). Kept for backward compatibility."""
        return self.get_task_instructions() + self.get_context_section()
    
    def get_icl_section(self, strategy: str = "generalized", custom_header: str = None) -> Dict[str, Any]:
        """Get ICL configuration for a specific strategy."""
        t = self.icl_template
        if strategy not in t['strategies']:
            raise ValueError(f"Unknown ICL strategy: {strategy}. Available: {list(t['strategies'].keys())}")
        
        if custom_header:
            header = custom_header
        else:
            header = f"{t['section_header']['title']}\n{t['section_header']['introduction'].strip()}"
        
        return {
            'header': header, 'strategy': strategy, 'config': t['strategies'][strategy],
            'example_format': t['example_format'], 'separator': t['separator']
        }
    
    def format_icl_examples(
        self,
        examples: List[Dict],
        strategy: str = "generalized"
    ) -> str:
        """
        Format ICL examples according to template.
        
        Args:
            examples: List of example dictionaries
            strategy: ICL strategy being used
        
        Returns:
            Formatted examples string
        """
        if not examples:
            return ""
        
        icl_config = self.get_icl_section(strategy)
        fmt = icl_config['example_format']
        
        prompt_parts = [icl_config['header'], ""]
        
        # Add strategy description
        strategy_descriptions = {
            'personalized': "The following examples are from the target user's historical data.",
            'generalized': "The following examples are from other users.",
            'hybrid': "The following examples include both the target user's historical data and data from other users."
        }
        
        if strategy in strategy_descriptions:
            prompt_parts.append(f"*{strategy_descriptions[strategy]}*\n")
        
        for i, example in enumerate(examples, 1):
            # Example header
            prompt_parts.append(fmt['header'].format(example_number=i))
            
            # User info (if available in example dict)
            if 'user_id' in example and 'date' in example:
                user_info = fmt['user_info_format'].format(
                    user_id=example.get('user_id', 'Unknown'),
                    date=example.get('date', 'Unknown')
                )
                prompt_parts.append(user_info)
            
            # Features (assume example has 'features_text' key)
            if 'features_text' in example:
                prompt_parts.append(example['features_text'])
            
            # Labels (if include_labels is True)
            if fmt['include_labels'] and 'anxiety_label' in example:
                label_text = fmt['label_format'].format(
                    anxiety_label=example.get('anxiety_label', 'Unknown'),
                    depression_label=example.get('depression_label', 'Unknown')
                )
                prompt_parts.append(label_text)
            
            # Separator between examples
            if i < len(examples):
                prompt_parts.append(icl_config['separator']['between_examples'])
        
        # Separator after all examples
        prompt_parts.append(icl_config['separator']['after_examples'])
        
        return "\n".join(prompt_parts)
    
    def get_reasoning_instruction(self, method: str = "cot") -> Dict[str, str]:
        """Get reasoning instructions for a specific method."""
        methods = self.reasoning_template['reasoning_methods']
        if method not in methods:
            raise ValueError(f"Unknown reasoning method: {method}. Available: {list(methods.keys())}")
        
        m = methods[method]
        return {
            'name': m['name'], 'description': m['description'],
            'instruction': m['instruction'].strip(), 'output_format': m['output_format'].strip()
        }
    
    # def get_output_constraints(self) -> str:
    #     """Get formatted output constraints (DO/DON'T guidelines)."""
    #     c = self.reasoning_template['output_constraints']
    #     parts = [c['header'], c['do_header']]
    #     parts.extend([f"- {item}" for item in c['do']])
    #     parts.append(c['dont_header'])
    #     parts.extend([f"- {item}" for item in c['dont']])
    #     parts.append("")
    #     return "\n".join(parts)
    
    def get_task_completion_prompt(self) -> str:
        """Get the task completion section prompt."""
        task = self.reasoning_template['task_completion']
        return "\n".join([task['header'], task['instruction'].strip(), "", task['reminder'].strip(), ""])
    
    def build_complete_prompt(
        self,
        input_data_text: str,
        icl_examples: Optional[List[Dict]] = None,
        icl_strategy: str = "generalized",
        reasoning_method: str = "cot",
        include_constraints: bool = True
    ) -> str:
        """
        Build a complete prompt with all components.
        
        Args:
            input_data_text: Formatted text of input data to predict
            icl_examples: Optional list of ICL examples
            icl_strategy: ICL strategy to use
            reasoning_method: Reasoning method to use
            include_constraints: Whether to include output constraints
        
        Returns:
            Complete formatted prompt
        
        Prompt order:
        1. Task Instructions (Role + Task)
        2. Context
        3. Examples (if provided)
        4. Input Data (User Data)
        5. Output Indicators (Reasoning Method, Guidelines, Output Format)
        """
        parts = []
        
        # 1. Task Instructions (Role + Task)
        parts.append(self.get_task_instructions())
        
        # 2. Context
        parts.append(self.get_context_section())
        
        # 3. ICL examples (if provided)
        if icl_examples:
            parts.append(self.format_icl_examples(icl_examples, icl_strategy))
        
        # 4. Input data (User Data to Predict)
        task_completion = self.reasoning_template['task_completion']
        parts.append(task_completion['header'])
        parts.append(task_completion['instruction'].strip())
        parts.append("")
        parts.append(task_completion['data_header'] + "\n")
        parts.append(input_data_text)
        parts.append("")
        
        # 5. Output Indicators
        reasoning = self.get_reasoning_instruction(reasoning_method)
        
        # 5a. Reasoning method
        parts.append(f"## Reasoning Method: {reasoning['name']}")
        parts.append(reasoning['instruction'])
        parts.append("")
        
        # # 5b. Guidelines (output constraints)
        # if include_constraints:
        #     parts.append(self.get_output_constraints())
        
        # 5c. Output format
        parts.append("## Output Format\n")
        parts.append(reasoning['output_format'])
        
        return "\n".join(parts)
    
    def save_prompt_to_file(self, prompt: str, filename: str):
        """
        Save a generated prompt to a file for inspection.
        
        Args:
            prompt: The prompt string to save
            filename: Output filename
        """
        output_dir = Path("generated_prompts")
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        print(f"Prompt saved to: {output_path}")


# Convenience Functions
def get_available_strategies() -> List[str]:
    """Get list of available ICL strategies."""
    return list(PromptManager().icl_template['strategies'].keys())


def get_available_reasoning_methods() -> List[str]:
    """Get list of available reasoning methods."""
    return list(PromptManager().reasoning_template['reasoning_methods'].keys())

