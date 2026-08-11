#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rzplatform.dofun.work /api/auth/login-phone AES 构造与白名单对比测试（只读探测，不写数据）"""
import base64
import json
import urllib.request
import ssl
from Crypto.Cipher import AES

A = "[EB1SFmzNn5"
B = "1V1GqOXiGTE"
C = "Gbd4WnZ2SZb"
D = "ld3U{N2Oll>"
OFF_A = "N3F4[kGjPXV1"
OFF_B = "Z{KlPHZxZR>>"


def q1(s: str) -> bytes:
    # 逐字符 charCode - 1，再 base64 解码
    shifted = "".join(chr(ord(ch) - 1) for ch in s)
    return base64.b64decode(shifted)


def derive_key() -> bytes:
    concat = A + B + C + D
    return q1(concat)


def derive_iv() -> bytes:
    concat = OFF_A + OFF_B
    return q1(concat)


def js_zero_pad(data: bytes) -> bytes:
    # 与前端一致：补 0 至 16 倍数（不足 16 的补齐；恰好 16 倍数时前端逻辑会追加一整块 0）
    rem = len(data) % 16
    if rem == 0:
        return data + b"\x00" * 16
    return data + b"\x00" * (16 - rem)


def aes_encrypt_hex(payload_obj) -> str:
    key = derive_key()
    iv = derive_iv()
    plain = json.dumps(payload_obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    padded = js_zero_pad(plain)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return cipher.encrypt(padded).hex()


def post_login(data_hex: str) -> str:
    body = json.dumps({"data": data_hex}).encode("utf-8")
    req = urllib.request.Request(
        "https://rzplatform.dofun.work/api/auth/login-phone",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
        return f"HTTP {resp.status} | {resp.read().decode('utf-8', 'replace')}"


if __name__ == "__main__":
    key = derive_key()
    iv = derive_iv()
    print(f"KEY(len={len(key)}): {key!r}")
    print(f"IV (len={len(iv)}):  {iv!r}")

    variants = [
        ('areaCode=""', {"areaCode": "", "phone": "10086"}),
        ('areaCode="+86"', {"areaCode": "+86", "phone": "10086"}),
        ('areaCode="86"', {"areaCode": "86", "phone": "10086"}),
    ]
    for label, payload in variants:
        ct = aes_encrypt_hex(payload)
        print(f"\n=== {label} | plaintext={json.dumps(payload, ensure_ascii=False)}")
        print(f"ciphertext(hex): {ct}")
        try:
            print("response:", post_login(ct))
        except Exception as e:
            print("request error:", e)
