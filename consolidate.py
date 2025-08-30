# -*- coding: utf-8 -*-
"""
代码整合工具

该脚本用于将项目中的多个源代码文件合并为一个单一的文本字符串。
它会按预定义的顺序读取指定的文件，为每个文件添加一个清晰的头部标识，
然后将所有内容拼接起来。最终，整合后的完整代码将被复制到系统剪贴板，
方便用户粘贴到文档、邮件或大语言模型（LLM）的输入框中。

主要功能:
  - 按指定顺序整合多个文件。
  - 为每个文件内容添加格式化的头部，标明文件路径。
  - 自动处理不同操作系统的路径差异。
  - 优雅地处理文件未找到或读取错误的情况。
  - 将最终结果复制到系统剪贴板，并提供清晰的执行反馈。

使用方法:
  1. 在 `FILES_TO_INCLUDE` 列表中配置需要整合的文件路径。
  2. 运行脚本: `python consolidate_code_optimized.py`
  3. 脚本执行成功后，内容将位于剪贴板中。
"""

import os
import pyperclip
from typing import List, Tuple

# ==============================================================================
# --- 1. 配置区: 指定需要整合的文件 ---
# ==============================================================================
# 在此列表中定义项目中所有需要被整合的核心代码文件的相对路径。
# 脚本将严格按照列表中的顺序进行文件读取与拼接。
# 建议按逻辑模块或功能对文件进行分组，以增强输出的可读性。

FILES_TO_INCLUDE: List[str] = [
    # --- 数据生成模块 ---
    "dag_generator.py",
    "expert_data_generator.py",

    # --- 强化学习核心框架 ---
    "environment.py",
    "model.py",
    "mcts.py",
    "replay_buffer.py",

    # --- 基础配置文件 ---
    "config.py",

    # --- 评估与基准测试 ---
    "main.py",
    "trainer.py",

    # --- 网页代码 ---
    "app.py",
    "static/css/style.css",
    "static/js/main.js",
    "templates/index.html",

    # --- 其他比较代码 ---
    "baseline_models.py",

    # --- md文件 ---
    # "README.md",
    # "改进方案.md"
]


# ==============================================================================
# --- 2. 脚本核心逻辑 (通常无需修改) ---
# ==============================================================================

def _create_file_header(filepath: str, max_width: int = 80) -> str:
    """为指定文件生成一个标准化的、居中的文本头部。

    Args:
        filepath (str): 需要创建头部的文件路径字符串。
        max_width (int): 头部的最大宽度，默认为 80 个字符。

    Returns:
        str: 格式化后的文件头部字符串。
    """
    title = f" FILE: {filepath} "

    # 防止文件名过长导致填充计算为负数
    if len(title) >= max_width:
        return f"\n\n{'=' * max_width}\n{title}\n{'=' * max_width}\n\n"

    padding_total = max_width - len(title)
    padding_left = padding_total // 2
    padding_right = padding_total - padding_left

    header_line = "=" * padding_left + title + "=" * padding_right
    return f"\n\n{header_line}\n\n"


def consolidate_files_to_clipboard(file_paths: List[str]) -> Tuple[bool, int]:
    """
    读取文件列表，将所有内容整合成一个字符串，并将其复制到系统剪贴板。

    该函数会遍历提供的文件路径列表，读取每个文件的内容，并在每个文件
    内容前添加一个格式化的头部。所有内容将被拼接成一个单一的字符串。

    Args:
        file_paths (List[str]): 需要整合的文件的相对路径列表。

    Returns:
        Tuple[bool, int]: 一个元组，第一个元素表示操作是否成功 (True/False)，
                          第二个元素是成功处理的文件数量。
    """
    print("🚀 开始整合项目代码...")

    consolidated_content = []
    processed_file_count = 0

    # 获取脚本所在的目录，用于构建绝对路径
    script_directory = os.path.dirname(__file__)

    for filepath in file_paths:
        try:
            # 构建跨平台兼容的绝对路径
            absolute_path = os.path.join(script_directory, filepath)

            with open(absolute_path, 'r', encoding='utf-8') as f:
                content = f.read()

            header = _create_file_header(filepath)
            consolidated_content.append(header)
            consolidated_content.append(content)
            processed_file_count += 1
            print(f"  ✅  已添加: {filepath}")

        except FileNotFoundError:
            print(f"  ⚠️  警告: 文件未找到，已跳过 -> {filepath}")
        except IOError as e:
            print(f"  ❌  错误: 读取文件 {filepath} 时发生 I/O 错误 -> {e}")
        except Exception as e:
            print(f"  ❌  错误: 处理文件 {filepath} 时遇到未知异常 -> {e}")

    if not consolidated_content:
        print("\n" + "=" * 50)
        print("🟡 警告: 未能成功处理任何文件。剪贴板内容未被修改。")
        print("=" * 50)
        return False, 0

    final_string = "".join(consolidated_content)

    try:
        pyperclip.copy(final_string)
        print("\n" + "=" * 50)
        print("🎉 操作成功！项目代码已整合并复制到系统剪贴板。")
        print(f"✨ 共包含 {processed_file_count} 个文件。现在您可以粘贴到任何需要的地方。")
        print("=" * 50)
        return True, processed_file_count
    except pyperclip.PyperclipException as e:
        print("\n" + "=" * 50)
        print("❌ 错误: 无法访问系统剪贴板。")
        print("   这通常发生在没有图形用户界面的环境（如纯SSH会话）中。")
        print(f"   错误详情: {e}")
        print("=" * 50)
        return False, processed_file_count


if __name__ == "__main__":
    consolidate_files_to_clipboard(FILES_TO_INCLUDE)