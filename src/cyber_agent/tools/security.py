from langchain_core.tools import tool
import socket

@tool
def scan_port(target: str, port: int) -> str:
    """
    端口扫描。必须提供 target 参数（字符串，IP或域名）和 port 参数（整数，端口号）。
    示例: scan_port(target="127.0.0.1", port=80)
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)  # 设置超时时间
        result = sock.connect_ex((target, port))
        sock.close()
        if result == 0:
            return f"✅ 端口 {port} 在目标 {target} 上是开放的。"
        else:
            return f"❌ 端口 {port} 在目标 {target} 上是关闭的。"
    except socket.gaierror:
        return f"❌ 错误：无效的目标地址 '{target}'。"
    except Exception as e:
        return f"❌ 扫描时发生未知错误：{e}"
