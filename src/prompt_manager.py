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
    
    def get_task_instructions(self, dataset: str = 'globem') -> str:
        """Construct the task instructions (Role + Task only).
        
        Args:
            dataset: 'globem' or 'ces' to use appropriate task description
        """
        t = self.base_template
        parts = [t['structure']['title']]
        
        # Role section
        parts.extend([t['structure']['sections'][0]['header'], t['role']['description'].strip(), ""])
        
        # Task section
        parts.append(t['structure']['sections'][1]['header'])
        
        # Use dataset-specific objective
        if dataset == 'ces':
            objective = t['task'].get('objective_ces', t['task']['objective'])
            output_targets = t['task'].get('output_targets_ces', t['task']['output_targets'])
        else:
            objective = t['task']['objective']
            output_targets = t['task']['output_targets']
        
        parts.extend([objective.strip(), "", t['task']['targets_header']])
        for target in output_targets:
            parts.append(f"- **{target['name']}**: {' or '.join(target['classes'])} (threshold: {target['threshold']})")
        parts.append("")
        
        return "\n".join(parts)
    
    def get_context_section(self, dataset: str = 'globem') -> str:
        """Construct the context section only.
        
        Args:
            dataset: 'globem' or 'ces' to use appropriate context
        """
        t = self.base_template
        ctx = t['context']
        parts = [t['structure']['sections'][2]['header']]
        
        # Use dataset-specific background
        if dataset == 'ces':
            background = ctx.get('background_ces', ctx['background'])
        else:
            background = ctx['background']
        
        parts.extend([background.strip(), ""])
        
        # Clinical associations
        # For CES, include stress indicators
        conditions_to_include = ['depression', 'anxiety']
        if dataset == 'ces':
            conditions_to_include.append('stress')
        
        for condition in conditions_to_include:
            if condition in ctx['clinical_associations']:
                info = ctx['clinical_associations'][condition]
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
            'cross_random': "The following examples are randomly selected from other users.",
            'cross_retrieval': "The following examples are selected from other users based on temporal similarity to the target user.",
            'personal_recent': "The following examples are the most recent data from the target user's historical records.",
            'hybrid_blend': "The following examples include both recent data from the target user and similar cases from other users."
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
                prompt_parts.append("")
            
            # Features (assume example has 'features_text' key)
            if 'features_text' in example:
                prompt_parts.append(example['features_text'])
            
            # Labels (if include_labels is True)
            if fmt['include_labels'] and 'anxiety_label' in example:
                # Check if stress label exists (CES dataset)
                if 'stress_label' in example:
                    label_text = f"Labels:\n"
                    label_text += f"  - Anxiety: {example.get('anxiety_label', 'Unknown')}\n"
                    label_text += f"  - Depression: {example.get('depression_label', 'Unknown')}\n"
                    label_text += f"  - Stress: {example.get('stress_label', 'Unknown')}"
                else:
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
    
    def get_reasoning_instruction(self, method: str = "cot", dataset: str = 'globem') -> Dict[str, str]:
        """Get reasoning instructions for a specific method.
        
        Args:
            method: Reasoning method ('cot', 'direct', 'self_feedback')
            dataset: 'globem' or 'ces' to use appropriate output format
        """
        methods = self.reasoning_template['reasoning_methods']
        if method not in methods:
            raise ValueError(f"Unknown reasoning method: {method}. Available: {list(methods.keys())}")
        
        m = methods[method]
        
        # Handle different reasoning method structures
        if method == 'self_feedback':
            # Self-feedback has initial_instruction instead of instruction
            # Use CES-specific format if dataset is 'ces' and it exists
            if dataset == 'ces':
                initial_output_format = m.get('initial_output_format_ces', m.get('initial_output_format', ''))
            else:
                initial_output_format = m.get('initial_output_format', '')
            
            return {
                'name': m['name'],
                'description': m['description'],
                'instruction': m.get('initial_instruction', m.get('instruction', '')).strip(),
                'output_format': initial_output_format.strip()
            }
        else:
            # Use CES-specific format if dataset is 'ces' and it exists
            if dataset == 'ces':
                output_format = m.get('output_format_ces', m.get('output_format', ''))
            else:
                output_format = m.get('output_format', '')
            
            return {
                'name': m['name'],
                'description': m['description'],
                'instruction': m.get('instruction', '').strip(),
                'output_format': output_format.strip()
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
    
    def get_task_completion_prompt(self, dataset: str = 'globem') -> str:
        """Get the task completion section prompt.
        
        Args:
            dataset: 'globem' or 'ces' to use appropriate instruction
        """
        task = self.reasoning_template['task_completion']
        
        # Use CES-specific instruction if dataset is 'ces' and it exists
        if dataset == 'ces':
            instruction = task.get('instruction_ces', task.get('instruction', ''))
        else:
            instruction = task.get('instruction', '')
        
        return "\n".join([task['header'], instruction.strip(), "", task['reminder'].strip(), ""])
    
    def build_complete_prompt(
        self,
        input_data_text: str,
        icl_examples: Optional[List[Dict]] = None,
        icl_strategy: str = "generalized",
        reasoning_method: str = "cot",
        include_constraints: bool = True,
        dataset: str = None
    ) -> str:
        """
        Build a complete prompt with all components.
        
        Args:
            input_data_text: Formatted text of input data to predict
            icl_examples: Optional list of ICL examples
            icl_strategy: ICL strategy to use
            reasoning_method: Reasoning method to use
            include_constraints: Whether to include output constraints
            dataset: 'globem' or 'ces' (auto-detected from config if None)
        
        Returns:
            Complete formatted prompt
        
        Prompt order:
        1. Task Instructions (Role + Task)
        2. Context
        3. Examples (if provided)
        4. Input Data (User Data)
        5. Output Indicators (Reasoning Method, Guidelines, Output Format)
        """
        # Auto-detect dataset if not provided
        if dataset is None:
            from . import config
            dataset = getattr(config, 'DATASET_TYPE', 'globem')
        
        parts = []
        
        # 1. Task Instructions (Role + Task)
        parts.append(self.get_task_instructions(dataset=dataset))
        
        # 2. Context
        parts.append(self.get_context_section(dataset=dataset))
        
        # 3. ICL examples (if provided)
        if icl_examples:
            parts.append(self.format_icl_examples(icl_examples, icl_strategy))
        
        # 4. Input data (User Data to Predict)
        task_completion = self.reasoning_template['task_completion']
        parts.append(task_completion['header'])
        
        # Use CES-specific instruction if dataset is 'ces' and it exists
        if dataset == 'ces':
            instruction = task_completion.get('instruction_ces', task_completion.get('instruction', ''))
        else:
            instruction = task_completion.get('instruction', '')
        
        parts.append(instruction.strip())
        parts.append("")
        parts.append(task_completion['data_header'] + "\n")
        parts.append(input_data_text)
        parts.append("")
        
        # 5. Output Indicators
        reasoning = self.get_reasoning_instruction(reasoning_method, dataset=dataset)
        
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

