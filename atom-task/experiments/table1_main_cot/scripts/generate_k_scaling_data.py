#!/usr/bin/env python3
"""
K-Scaling Data Generation Script
Convert raw data into 3 CoT formats with varying reasoning chain length.

Core principle: All formats preserve the SAME core information.
The only difference is HOW the information is presented/formatted.

k=1 (Short):  Abbreviated field names + abbreviated operation names
k=2 (Standard): Current standard format
k=3 (Long):    Standard format + redundant words within the framework
"""

import json
import argparse
from pathlib import Path

# Operation name full -> abbreviation mapping
OP_ABBREV = {
    'Reverse': 'Rev',
    'Substitute': 'Sub',
    'Duplicate': 'Dup',
    'Deduplicate': 'Dedup',
    'Rotate': 'Rot'
}


def parse_task_chain(sample):
    """Parse task chain into a list of task names."""
    task_chain_str = sample.get('task_chain_readable', sample.get('task_chain', ''))
    if '→' in task_chain_str:
        task_chain = [t.strip() for t in task_chain_str.split('→')]
    elif '->' in task_chain_str:
        task_chain = [t.strip() for t in task_chain_str.split('->')]
    else:
        task_chain = [task_chain_str.strip()]
    return task_chain


def abbreviate_task_chain(task_chain):
    """Convert task chain to abbreviated form."""
    return [OP_ABBREV.get(t, t) for t in task_chain]


def format_k1(sample):
    """k=1 (Short): Abbreviated field names + abbreviated operations.
    S1: TC:[Rev,Dup,Rot] In:[xxx] Out:[xxx] Rem:[Dup,Rot]
    """
    reasoning_steps = sample.get('reasoning_steps', [])
    task_chain = parse_task_chain(sample)
    task_chain_abbrev = abbreviate_task_chain(task_chain)

    lines = []
    for i, step_info in enumerate(reasoning_steps):
        current_tasks_abbrev = task_chain_abbrev[i:]
        remaining_abbrev = task_chain_abbrev[i+1:]

        current_display = ','.join(current_tasks_abbrev)
        remaining_display = ','.join(remaining_abbrev) if remaining_abbrev else '-'

        step_input = step_info.get('input', '')
        step_output = step_info.get('output', '')

        line = f"S{i+1}: TC:[{current_display}] In:[{step_input}] Out:[{step_output}] Rem:[{remaining_display}]"
        lines.append(line)

    output = sample.get('output', '')
    return '\n'.join(lines) + f"\nFinal Answer: [{output}]"


def format_k2(sample):
    """k=2 (Standard): Current standard format.
    Step 1: Task_Chain: [Reverse, Duplicate, Rotate] | Input_Sequence: [xxx] | -> Output_Sequence: [xxx] | Task_Chain: [Duplicate, Rotate]
    """
    reasoning_steps = sample.get('reasoning_steps', [])
    task_chain = parse_task_chain(sample)

    lines = []
    for i, step_info in enumerate(reasoning_steps):
        current_tasks = task_chain[i:]
        remaining_tasks = task_chain[i+1:]

        current_display = '[' + ', '.join(current_tasks) + ']'
        remaining_display = '[' + ', '.join(remaining_tasks) + ']' if remaining_tasks else '[]'

        step_input = step_info.get('input', '')
        step_output = step_info.get('output', '')

        line = f"Step {i+1}: Task_Chain: {current_display} | Input_Sequence: [{step_input}] | -> Output_Sequence: [{step_output}] | Task_Chain: {remaining_display}"
        lines.append(line)

    output = sample.get('output', '')
    return '\n'.join(lines) + f"\nFinal Answer: [{output}]"


def format_k3(sample):
    """k=3 (Long): Standard format with redundant words within the framework.
    Same info as k2, but fields contain more verbose wording.
    """
    reasoning_steps = sample.get('reasoning_steps', [])
    task_chain = parse_task_chain(sample)

    lines = []
    for i, step_info in enumerate(reasoning_steps):
        current_tasks = task_chain[i:]
        remaining_tasks = task_chain[i+1:]

        current_display = '[' + ', '.join(current_tasks) + ']'
        remaining_display = '[' + ', '.join(remaining_tasks) + ']' if remaining_tasks else '[]'

        step_input = step_info.get('input', '')
        step_output = step_info.get('output', '')

        # Verbose version of the same framework
        line = f"Step {i+1}: Full_Task_Chain: {current_display} | Current_Input_Sequence: [{step_input}] | Produces_Output_Sequence: [{step_output}] | Remaining_Task_Chain: {remaining_display}"
        lines.append(line)

    output = sample.get('output', '')
    return '\n'.join(lines) + f"\nFinal Answer: [{output}]"


FORMATTERS = {
    'k1': format_k1,
    'k2': format_k2,
    'k3': format_k3,
}


def convert_dataset(input_file, output_file, k_level):
    """Convert input dataset to specified k-level CoT format."""
    formatter = FORMATTERS[k_level]

    count = 0
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            sample = json.loads(line.strip())

            # Convert to k-level CoT format
            cot_output = formatter(sample)

            # Build new sample
            new_sample = sample.copy()
            new_sample['output'] = cot_output

            # For k1, also modify task_chain_readable to use abbreviations
            if k_level == 'k1':
                task_chain = parse_task_chain(sample)
                task_chain_abbrev = abbreviate_task_chain(task_chain)
                new_sample['task_chain_readable'] = ' -> '.join(task_chain_abbrev)

            fout.write(json.dumps(new_sample, ensure_ascii=False) + '\n')
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description='K-Scaling Data Generation')
    parser.add_argument('--input', '-i', required=True, help='Input file path')
    parser.add_argument('--output', '-o', required=True, help='Output file path')
    parser.add_argument('--k', '-k', required=True, choices=['k1', 'k2', 'k3'],
                        help='CoT format level: k1=short, k2=standard, k3=long')
    args = parser.parse_args()

    count = convert_dataset(args.input, args.output, args.k)
    print(f"Conversion complete: {args.input} -> {args.output} (k={args.k}), {count} samples")


if __name__ == '__main__':
    main()
