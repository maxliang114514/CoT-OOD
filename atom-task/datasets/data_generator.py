"""
核心数据生成模块 - CoT-4版本
使用新分隔符 '·' 代替 '|'
"""

import random
from typing import List, Tuple, Dict, Any
from itertools import permutations



SYMBOLS = ['α', 'β', 'γ', 'δ', 'ε', 'ω']
SEPARATOR = '·'


SUBSTITUTION_MAP = {
    'α': 'β',
    'β': 'α',
    'γ': 'δ',
    'δ': 'γ',
    'ε': 'ω',
    'ω': 'ε'
}



def task_reverse(sequence: List[str]) -> List[str]:
    """t1: 逆序 - 将输入序列倒序排列"""
    return list(reversed(sequence))


def task_substitute(sequence: List[str]) -> List[str]:
    """t2: 替换 - 应用预定义的映射表"""
    result = []
    for symbol in sequence:
        if symbol == SEPARATOR:
            result.append(symbol)
        elif symbol in SUBSTITUTION_MAP:
            result.append(SUBSTITUTION_MAP[symbol])
        else:
            result.append(symbol)
    return result


def task_duplicate(sequence: List[str]) -> List[str]:
    """t3: 复制 - 将每个符号连续复制一次"""
    result = []
    for symbol in sequence:
        result.append(symbol)
        result.append(symbol)
    return result


def task_deduplicate(sequence: List[str]) -> List[str]:
    """t4: 去重 - 相邻重复元素只保留一个"""
    if not sequence:
        return []
    result = [sequence[0]]
    for i in range(1, len(sequence)):
        if sequence[i] != sequence[i-1]:
            result.append(sequence[i])
    return result


def task_rotate(sequence: List[str]) -> List[str]:
    """t5: 搬运 - 将序列的第一个元素移动到最后"""
    if len(sequence) <= 1:
        return list(sequence)
    return sequence[1:] + [sequence[0]]



ATOMIC_TASKS = {
    't1': task_reverse,
    't2': task_substitute,
    't3': task_duplicate,
    't4': task_deduplicate,
    't5': task_rotate
}

TASK_NAMES = {
    't1': 'Reverse',
    't2': 'Substitute',
    't3': 'Duplicate',
    't4': 'Deduplicate',
    't5': 'Rotate'
}

TASK_INSTRUCTIONS = {
    't1': "Reverse the sequence order.",
    't2': "Apply substitution mapping {α→δ, β→ω, γ→α, δ→β, ε→γ, ω→ε}.",
    't3': "Duplicate each symbol consecutively.",
    't4': "Remove consecutive duplicate elements.",
    't5': "Move first element to the end."
}




def generate_random_sequence(min_len: int = 2, max_len: int = 6) -> List[str]:
    """生成随机符号序列"""
    if min_len < 1:
        min_len = 1
    if max_len < min_len:
        max_len = min_len
    
    available = SYMBOLS
    result = []
    while len(result) < min_len:
        length = random.randint(min_len, max_len)
        result = random.choices(available, k=length)
    
    while len(result) == 0:
        result = [random.choice(SYMBOLS)]
    
    result = [s for s in result if s in SYMBOLS]
    if len(result) == 0:
        result = [random.choice(SYMBOLS)]
    
    return result


def generate_unique_sequences(count: int, min_len: int = 2, max_len: int = 6,
                               existing_sequences: set = None) -> List[List[str]]:
    """生成不重复的唯一序列"""
    sequences = set()
    existing = existing_sequences if existing_sequences else set()
    
    if min_len < 1:
        min_len = 1
    
    while len(sequences) < count:
        seq = generate_random_sequence(min_len, max_len)
        if len(seq) == 0:
            continue
        seq_tuple = tuple(seq)
        if seq_tuple not in sequences and seq_tuple not in existing:
            sequences.add(seq_tuple)
    
    return [list(s) for s in sequences]




def sequence_to_string(sequence: List[str]) -> str:
    """将序列转换为字符串（无空格）"""
    return ''.join(sequence)


def string_to_sequence(s: str) -> List[str]:
    """将空格分隔的字符串转换为序列"""
    return s.split()




def apply_task_chain(input_seq: List[str], task_chain: List[str]) -> Tuple[List[Dict], str]:
    """
    执行任务链并返回推理步骤和最终输出
    
    Args:
        input_seq: 输入序列（符号列表）
        task_chain: 任务链，如 ['t3', 't1', 't2']
    
    Returns:
        (推理步骤列表, 最终输出字符串)
    """
    current = list(input_seq)
    reasoning_steps = []
    
    for i, task_id in enumerate(task_chain):
        task_func = ATOMIC_TASKS[task_id]
        prev_state = current
        current = task_func(current)
        
        reasoning_steps.append({
            'step': i + 1,
            'task': TASK_NAMES[task_id],
            'input': sequence_to_string(prev_state),
            'output': sequence_to_string(current)
        })
    
    final_output = sequence_to_string(current)
    return reasoning_steps, final_output


def task_chain_to_string(task_chain: List[str]) -> str:
    """将任务链转换为可读字符串"""
    return ' → '.join([TASK_NAMES[tid] for tid in task_chain])
