import os
import subprocess
import sys
import time


def spawn_athena(exe_path: str, cwd: str | None = None) -> subprocess.Popen:
    if not os.path.isfile(exe_path):
        raise FileNotFoundError(f"找不到 athena 可执行文件: {exe_path}")

    proc = subprocess.Popen(
        [exe_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # 合并 stderr，确保读到提示符
        text=True,
        encoding="latin-1",  # Athena 老版本常见 8-bit 输出
        errors="replace",     # 容错，避免解码异常
        bufsize=1,
        cwd=cwd or os.getcwd(),
    )
    return proc


def wait_for_prompt(proc: subprocess.Popen, timeout: float = 30.0) -> str:
    """读取直到出现提示符（优先匹配 'ATHENA>'，否则回退到任意 '>'）。"""
    start = time.monotonic()
    output: list[str] = []
    tail = ""
    while True:
        if proc.poll() is not None:
            # 子进程已退出
            break
        ch = proc.stdout.read(1)
        if not ch:
            # 管道暂时无数据，做个短暂休眠并检查超时
            if time.monotonic() - start > timeout:
                raise TimeoutError("等待提示符超时")
            time.sleep(0.01)
            continue
        output.append(ch)
        sys.stdout.write(ch)
        sys.stdout.flush()
        # 仅保留末尾窗口，降低搜索成本
        tail = (tail + ch)[-16:]
        if tail.endswith("ATHENA>"):
            return "".join(output)
        # 兜底：任意 '>'
        if tail.endswith(">"):
            return "".join(output)

    return "".join(output)


def send_command(proc: subprocess.Popen, command: str, timeout: float = 30.0) -> str:
    if not command.endswith("\n"):
        command += "\n"
    if proc.stdin is None:
        raise RuntimeError("进程 stdin 不可用")
    proc.stdin.write(command)
    proc.stdin.flush()
    return wait_for_prompt(proc, timeout=timeout)


def main() -> int:
    # 零参数运行：优先使用环境变量 ATHENA_EXE；否则使用常见默认路径
    athena_exe = os.environ.get("ATHENA_EXE") or r"C:\sedatools\exe\athena.exe"
    timeout_sec = 30.0

    print(f"启动 Athena: {athena_exe}")
    proc = spawn_athena(athena_exe)

    try:
        print("\n[等待首个提示符...]\n")
        wait_for_prompt(proc, timeout=timeout_sec)

        tests = [
            "line x loc = 0.0  spacing=0.1",
            "line y loc = 0.0  spacing=0.1",
        ]

        for cmd in tests:
            print(f"\n[执行命令] {cmd}\n")
            send_command(proc, cmd, timeout=timeout_sec)

        print("\n[执行命令] quit\n")
        send_command(proc, "quit", timeout=timeout_sec)

        return 0
    except Exception as exc:  # noqa: BLE001 - 简单测试脚本允许宽泛捕获
        print(f"发生异常: {exc}")
        return 1
    finally:
        try:
            if proc.poll() is None:
                # 如果还在运行，尽量结束
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())


