"""Webhook 加解密工具：PKCS7 填充、AES-CBC 加解密。

从 webhook.py 中提取的专业加密逻辑，支持 pycryptodome 和 cryptography 双后端。
"""
from __future__ import annotations

from collections.abc import Callable


def pkcs7_unpad(payload: bytes, block_size: int) -> bytes:
    """去除 PKCS7 填充。"""
    if not payload:
        raise ValueError("加密数据不能为空。")
    padding_size = payload[-1]
    if padding_size < 1 or padding_size > block_size:
        raise ValueError("加密数据的填充字节非法。")
    if payload[-padding_size:] != bytes([padding_size]) * padding_size:
        raise ValueError("加密数据的填充内容非法。")
    return payload[:-padding_size]


def pkcs7_pad(payload: bytes, block_size: int) -> bytes:
    """PKCS7 填充到 block_size 对齐。"""
    padding_size = block_size - (len(payload) % block_size)
    if padding_size == 0:
        padding_size = block_size
    return payload + bytes([padding_size]) * padding_size


def _load_optional_aes_cipher() -> tuple[
    Callable[[bytes, bytes, bytes], bytes],
    Callable[[bytes, bytes, bytes], bytes],
]:
    """按可用性加载 AES-CBC 加解密实现，避免为 webhook 新增强制依赖。"""
    try:
        from Crypto.Cipher import AES  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        try:
            from cryptography.hazmat.backends import default_backend  # type: ignore[import-not-found]
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise ValueError(
                "当前运行环境缺少 AES 加解密依赖，请安装 pycryptodome 或 cryptography 后再启用官方加密回调。"
            ) from exc

        def decryptor(key: bytes, iv: bytes, payload: bytes) -> bytes:
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            decrypt_context = cipher.decryptor()
            return decrypt_context.update(payload) + decrypt_context.finalize()

        def encryptor(key: bytes, iv: bytes, payload: bytes) -> bytes:
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            encrypt_context = cipher.encryptor()
            return encrypt_context.update(payload) + encrypt_context.finalize()

        return decryptor, encryptor

    def decryptor(key: bytes, iv: bytes, payload: bytes) -> bytes:
        cipher = AES.new(key, AES.MODE_CBC, iv)
        return cipher.decrypt(payload)

    def encryptor(key: bytes, iv: bytes, payload: bytes) -> bytes:
        cipher = AES.new(key, AES.MODE_CBC, iv)
        return cipher.encrypt(payload)

    return decryptor, encryptor


def aes_cbc_decrypt(key: bytes, iv: bytes, payload: bytes) -> bytes:
    """AES-CBC 解密。"""
    decryptor, _ = _load_optional_aes_cipher()
    return decryptor(key, iv, payload)


def aes_cbc_encrypt(key: bytes, iv: bytes, payload: bytes) -> bytes:
    """AES-CBC 加密。"""
    _, encryptor = _load_optional_aes_cipher()
    return encryptor(key, iv, payload)
