"""
Prompt Template System - Demo Script

This script demonstrates how to use the YAML-based prompt template system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prompt_manager import (
    PromptManager,
    get_available_strategies,
    get_available_reasoning_methods,
    print_strategy_info,
    print_reasoning_info
)


def demo_basic_usage():
    """Demonstrate basic prompt generation."""
    print("="*80)
    print("DEMO 1: Basic Prompt Generation")
    print("="*80)
    
    # Initialize manager
    manager = PromptManager()
    
    # Sample input data
    input_data = """
**User:** INS-W_507
**Date:** 2019-04-17

**Behavioral Features (7-day normalized):**

| Feature | Value | Interpretation |
|---------|-------|----------------|
| Sleep duration | -0.15 | Below personal average |
| Sleep efficiency | -0.25 | Below personal average |
| Step count | +0.42 | Above personal average |
| Screen time | +0.18 | Above personal average |
| Call count | -0.32 | Below personal average |
| Location entropy | +0.08 | Slightly above average |
| Time at home | -0.20 | Below personal average |
"""
    
    # Generate prompt with Chain-of-Thought reasoning
    print("\nGenerating prompt with CoT reasoning...")
    prompt = manager.build_complete_prompt(
        input_data_text=input_data,
        icl_examples=None,  # Zero-shot for demo
        icl_strategy="zero_shot",
        reasoning_method="cot",
        include_constraints=True
    )
    
    print(f"\n✓ Prompt generated successfully!")
    print(f"  - Length: {len(prompt)} characters")
    print(f"  - Estimated tokens: ~{len(prompt) // 4}")
    
    # Save to file
    manager.save_prompt_to_file(prompt, "demo_cot_prompt.txt")


def demo_different_configurations():
    """Demonstrate different configuration combinations."""
    print("\n" + "="*80)
    print("DEMO 2: Different Configuration Combinations")
    print("="*80)
    
    manager = PromptManager()
    
    # Simple input for demo
    input_data = "**User:** Test\n**Date:** 2024-01-01\n\nFeatures: ..."
    
    configs = [
        ("zero_shot", "direct", "Baseline"),
        ("generalized", "cot", "Standard"),
        ("personalized", "cot", "Personalized"),
        ("hybrid", "tot", "Advanced"),
    ]
    
    print("\nGenerating prompts with different configurations:\n")
    
    for icl_strategy, reasoning_method, label in configs:
        prompt = manager.build_complete_prompt(
            input_data_text=input_data,
            icl_strategy=icl_strategy,
            reasoning_method=reasoning_method,
            include_constraints=True
        )
        
        print(f"  {label:15} (ICL: {icl_strategy:12}, Reasoning: {reasoning_method:15})")
        print(f"    → {len(prompt):6,} chars (~{len(prompt) // 4:5,} tokens)")


def demo_components():
    """Demonstrate accessing individual components."""
    print("\n" + "="*80)
    print("DEMO 3: Accessing Individual Components")
    print("="*80)
    
    manager = PromptManager()
    
    # 1. Base prompt
    print("\n1. Base Prompt (Role + Task + Context):")
    base = manager.get_base_prompt()
    print(f"   Length: {len(base)} characters")
    print(f"   Preview:\n{base[:300]}...\n")
    
    # 2. ICL configuration
    print("2. ICL Configuration (Generalized):")
    icl_config = manager.get_icl_section("generalized")
    print(f"   Strategy: {icl_config['strategy']}")
    print(f"   N Examples: {icl_config['config']['n_examples']}")
    print(f"   Selection: {icl_config['config']['example_selection']}")
    
    # 3. Reasoning instruction
    print("\n3. Reasoning Instruction (Chain-of-Thought):")
    reasoning = manager.get_reasoning_instruction("cot")
    print(f"   Name: {reasoning['name']}")
    print(f"   Description: {reasoning['description']}")
    print(f"   Instruction length: {len(reasoning['instruction'])} chars")
    
    # 4. Output constraints
    print("\n4. Output Constraints:")
    constraints = manager.get_output_constraints()
    print(f"   Length: {len(constraints)} characters")
    print(f"   Preview:\n{constraints[:200]}...\n")


def demo_available_options():
    """Show available strategies and methods."""
    print("\n" + "="*80)
    print("DEMO 4: Available Options")
    print("="*80)
    
    print("\n" + "-"*80)
    print_strategy_info()
    
    print("\n" + "-"*80)
    print_reasoning_info()


def demo_customization():
    """Demonstrate how to customize prompts."""
    print("\n" + "="*80)
    print("DEMO 5: Customization Examples")
    print("="*80)
    
    manager = PromptManager()
    
    print("\n1. Custom ICL header:")
    icl_config = manager.get_icl_section(
        strategy="generalized",
        custom_header="## Learning from Similar Cases\nLet's look at comparable situations..."
    )
    print(f"   {icl_config['header'][:80]}...")
    
    print("\n2. Building prompt without constraints:")
    input_data = "Test data..."
    prompt_no_constraints = manager.build_complete_prompt(
        input_data_text=input_data,
        icl_strategy="zero_shot",
        reasoning_method="direct",
        include_constraints=False  # Shorter prompt
    )
    prompt_with_constraints = manager.build_complete_prompt(
        input_data_text=input_data,
        icl_strategy="zero_shot",
        reasoning_method="direct",
        include_constraints=True
    )
    
    print(f"   Without constraints: {len(prompt_no_constraints):,} chars")
    print(f"   With constraints:    {len(prompt_with_constraints):,} chars")
    print(f"   Difference:          {len(prompt_with_constraints) - len(prompt_no_constraints):,} chars")


def demo_integration_example():
    """Show integration with evaluation pipeline."""
    print("\n" + "="*80)
    print("DEMO 6: Integration Example")
    print("="*80)
    
    print("""
Example integration with LLM evaluation:

```python
from prompt_manager import PromptManager
from llm_evaluation import LLMEvaluator

# Initialize
manager = PromptManager()
evaluator = LLMEvaluator(model="gpt-4o-mini")

# Build prompt
prompt = manager.build_complete_prompt(
    input_data_text=user_data_text,
    icl_examples=examples,
    icl_strategy="generalized",
    reasoning_method="cot"
)

# Get prediction
result = evaluator.predict(prompt)
print(f"Anxiety: {result['prediction']['Anxiety']}")
print(f"Depression: {result['prediction']['Depression']}")
```

For ablation studies:

```python
# Compare different configurations
configs = [
    ("zero_shot", "direct"),
    ("generalized", "cot"),
    ("personalized", "cot"),
]

results = []
for icl_strategy, reasoning_method in configs:
    prompt = manager.build_complete_prompt(
        input_data_text=user_data_text,
        icl_strategy=icl_strategy,
        reasoning_method=reasoning_method
    )
    result = evaluator.predict(prompt)
    results.append({
        'config': (icl_strategy, reasoning_method),
        'prediction': result
    })
```
    """)


def main():
    """Run all demos."""
    print("\n" + "#"*80)
    print("# Prompt Template System - Interactive Demo")
    print("#"*80)
    
    demos = [
        ("Basic Usage", demo_basic_usage),
        ("Different Configurations", demo_different_configurations),
        ("Component Access", demo_components),
        ("Available Options", demo_available_options),
        ("Customization", demo_customization),
        ("Integration", demo_integration_example),
    ]
    
    for i, (name, func) in enumerate(demos, 1):
        try:
            func()
        except Exception as e:
            print(f"\n✗ Error in {name}: {e}")
        
        if i < len(demos):
            input("\nPress Enter to continue to next demo...")
    
    print("\n" + "="*80)
    print("Demo completed!")
    print("="*80)
    print("\nGenerated files can be found in: generated_prompts/")
    print("\nNext steps:")
    print("  1. Check the generated prompt files")
    print("  2. Modify YAML templates in src/prompts/")
    print("  3. Create custom configurations")
    print("  4. Integrate with your evaluation pipeline")
    print("\n")


if __name__ == "__main__":
    main()

