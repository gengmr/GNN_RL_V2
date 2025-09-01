import subprocess
from datetime import datetime


def run_command(command):
    """
    执行一个 shell 命令，并实时打印输出。
    如果命令执行失败，则抛出异常。
    """
    try:
        # 使用 subprocess.run 来执行命令
        process = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            encoding='utf-8'  # 明确指定编码以避免乱码问题
        )
        if process.stdout:
            print(process.stdout.strip())  # 打印标准输出并去除首尾空白
        if process.stderr:
            print(process.stderr.strip())  # 打印标准错误并去除首尾空白
        return True
    except FileNotFoundError:
        print(f"错误: 命令 '{command[0]}' 未找到。")
        print("请确保 Git 已经安装并且其路径已添加到系统的 PATH 环境变量中。")
        return False
    except subprocess.CalledProcessError as e:
        # 捕获命令执行失败的错误
        failed_command_str = " ".join(e.cmd)
        print(f"错误: 执行命令 '{failed_command_str}' 失败。")
        print(f"返回码: {e.returncode}")
        # 打印详细的输出和错误信息
        if e.stdout:
            print(f"输出:\n{e.stdout.strip()}")
        if e.stderr:
            print(f"错误输出:\n{e.stderr.strip()}")
        return False


def main(commit_message=None):
    """
    主执行函数

    参数:
    commit_message (str, optional): Git 提交时使用的信息。
                                      如果为 None 或空字符串, 则使用默认的时间戳信息。
                                      默认为 None。
    """
    # 检查是否提供了有效的提交信息
    if commit_message:
        # 如果传入了参数，则使用它
        print(f"信息: 使用您提供的自定义提交信息: '{commit_message}'")
    else:
        # 如果没有提供参数，则使用当前时间作为默认提交信息
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        commit_message = f"Auto commit at {timestamp}"
        print(f"信息: 未提供提交信息，使用默认时间戳: '{commit_message}'")

    # 定义要按顺序执行的 Git 命令
    git_commands = [
        ["git", "add", "."],
        ["git", "commit", "-m", commit_message],
        ["git", "push"]
    ]

    # 依次执行所有 Git 命令
    for command in git_commands:
        command_str = " ".join(command)
        # 为了美观，对 commit 命令中的消息做个截断显示
        if command[1] == 'commit':
            # 确保即使原始消息很短，截断也不会出错
            display_message = commit_message if len(commit_message) < 30 else f"{commit_message[:30]}..."
            command_str = f'git commit -m "{display_message}"'

        print(f"--- 正在执行: {command_str} ---")

        if not run_command(command):
            print("\n!!! 操作失败，脚本已中止。!!!")
            break  # 如果有任何一步失败，则停止后续所有操作
        print("-" * (len(command_str) + 12))  # 打印分隔线
    else:
        # 只有当 for 循环正常结束（没有被 break）时，才会执行 else 块
        print("\n===================================")
        print("  所有操作均已成功完成！")
        print("===================================\n")


if __name__ == "__main__":
    # --- 如何使用 ---

    # 示例 1: 不传入任何参数，使用默认的时间戳作为提交信息。
    # main()

    # 示例 2: 传入一个自定义的字符串作为提交信息。
    my_commit_msg = "版本备份-修改监督学习无效问题以及强化学习中各个loss为nan问题"
    main(commit_message=my_commit_msg)