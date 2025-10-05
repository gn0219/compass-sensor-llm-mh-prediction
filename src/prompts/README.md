# Prompt Template System

This directory contains YAML-based prompt templates for mental health prediction using LLMs.

## Overview

The prompt template system is organized into three main components:

1. **`base.yaml`** - Core prompt elements (Role, Task, Context)
2. **`icl.yaml`** - In-Context Learning (ICL) configuration
3. **`reasoning.yaml`** - Reasoning methods and output constraints

## File Structure

```
prompts/
├── README.md              # This file
├── base.yaml             # Base prompt template
├── icl.yaml              # ICL configuration
├── reasoning.yaml        # Reasoning methods & constraints
└── experiments/          # Custom experimental configurations
    └── ...
```

## Quick Start

### Using the Prompt Manager

```python
from prompt_manager import PromptManager

# Initialize
manager = PromptManager()

# Get base prompt
base_prompt = manager.get_base_prompt()

# Build complete prompt with CoT reasoning
prompt = manager.build_complete_prompt(
    input_data_text="<your formatted user data>",
    icl_examples=None,  # or list of examples
    icl_strategy="generalized",
    reasoning_method="cot",
    include_constraints=True
)
```

### Available Configurations

**ICL Strategies:**
- `personalized` - Use examples from the same user's history
- `generalized` - Use examples from different users
- `hybrid` - Mix of personal and general examples
- `zero_shot` - No examples provided

**Reasoning Methods:**
- `direct` - Immediate prediction without explicit reasoning
- `cot` - Chain-of-Thought (step-by-step reasoning)
- `tot` - Tree-of-Thoughts (explore multiple paths)
- `self_consistency` - Multiple independent reasoning chains

## Customization

### Modifying Templates

1. **Edit YAML files** - Modify the YAML configuration files
2. **No code changes needed** - The `PromptManager` automatically loads changes
3. **Version control** - Track prompt changes via git

### Creating Custom Configurations

You can create custom YAML files for experiments:

```yaml
# prompts/experiments/custom_experiment.yaml
metadata:
  name: "Custom Experiment"
  version: "1.0"
  description: "Testing new prompt strategy"

# Your custom configuration...
```

Then load with:

```python
manager = PromptManager()
custom_config = manager._load_yaml("experiments/custom_experiment.yaml")
```

## Template Structure

### base.yaml

```yaml
role:
  title: "Role name"
  description: "Detailed role description"

task:
  objective: "What the model should do"
  output_targets: [...]

context:
  background: "Domain knowledge"
  clinical_associations: {...}
  feature_categories: [...]
```

### icl.yaml

```yaml
section_header:
  title: "Section title"
  introduction: "Intro text"

example_format:
  header: "Example {example_number}"
  include_labels: true
  feature_format: "table"

strategies:
  personalized: {...}
  generalized: {...}
```

### reasoning.yaml

```yaml
reasoning_methods:
  cot:
    name: "Chain-of-Thought"
    instruction: "How to reason"
    output_format: "Expected JSON format"

output_constraints:
  do: [...]
  dont: [...]
```

## Best Practices

### 1. Keep Templates Modular
- Separate concerns (base, ICL, reasoning)
- Easy to mix and match components

### 2. Version Your Prompts
- Use git to track changes
- Add meaningful commit messages
- Tag important versions

### 3. Document Experiments
- Create experiment-specific YAML files
- Include metadata (name, version, description)
- Note what hypothesis you're testing

### 4. Test Prompts Before Deployment
```python
# Save generated prompt for inspection
manager.save_prompt_to_file(prompt, "test_prompt.txt")
```

### 5. Monitor Template Size
```python
# Check prompt length
prompt_length = len(prompt)
token_estimate = prompt_length / 4  # rough estimate
print(f"Estimated tokens: {token_estimate}")
```

## Advanced Usage

### Custom Prompt Builder

```python
from prompt_manager import PromptManager

class CustomPromptBuilder:
    def __init__(self):
        self.manager = PromptManager()
    
    def build_experiment_prompt(self, experiment_config):
        # Load custom configuration
        config = self.manager._load_yaml(f"experiments/{experiment_config}.yaml")
        
        # Build custom prompt
        # ... your logic here
        
        return prompt
```

### Batch Prompt Generation

```python
def generate_prompts_for_ablation_study():
    manager = PromptManager()
    
    configs = [
        ("zero_shot", "direct"),
        ("generalized", "cot"),
        ("personalized", "cot"),
        ("hybrid", "tot"),
    ]
    
    prompts = []
    for icl_strategy, reasoning_method in configs:
        prompt = manager.build_complete_prompt(
            input_data_text=input_data,
            icl_strategy=icl_strategy,
            reasoning_method=reasoning_method
        )
        prompts.append((icl_strategy, reasoning_method, prompt))
    
    return prompts
```

### A/B Testing

```python
# Version A: Original prompt
manager_v1 = PromptManager(prompts_dir="prompts/v1")
prompt_a = manager_v1.build_complete_prompt(...)

# Version B: Modified prompt
manager_v2 = PromptManager(prompts_dir="prompts/v2")
prompt_b = manager_v2.build_complete_prompt(...)

# Compare results...
```

## Integration with Evaluation Pipeline

See `../llm_evaluation.py` for integration examples:

```python
from prompt_manager import PromptManager
from reasoning import LLMReasoner

# Initialize
manager = PromptManager()
evaluator = LLMEvaluator(model="gpt-4o-mini")

# Build prompt
prompt = manager.build_complete_prompt(...)

# Evaluate
result = evaluator.predict(prompt)
```

## Troubleshooting

### YAML Parsing Errors
- Check indentation (use spaces, not tabs)
- Validate YAML syntax: https://www.yamllint.com/

### Template Not Found
```python
# Specify custom prompts directory
manager = PromptManager(prompts_dir="/path/to/prompts")
```

### Missing Required Fields
- Ensure all required fields are in YAML
- Check the schema in each template file

## Contributing

When adding new templates:

1. Follow the existing structure
2. Add clear descriptions and metadata
3. Document parameters and usage
4. Test with example data
5. Update this README

## References

- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
- [Tree-of-Thoughts](https://arxiv.org/abs/2305.10601)

