"""Webhook 加解密模块测试。"""
from __future__ import annotations

import unittest

from cyber_agent.cli.webhook_crypto import (
    aes_cbc_decrypt,
    aes_cbc_encrypt,
    pkcs7_pad,
    pkcs7_unpad,
)


class PKCS7TestCase(unittest.TestCase):
    """测试 PKCS7 填充/去填充。"""

    def test_pad_and_unpad_roundtrip(self) -> None:
        """填充后去填充还原原始数据。"""
        data = b"hello world"
        padded = pkcs7_pad(data, 16)
        unpadded = pkcs7_unpad(padded, 16)
        self.assertEqual(unpadded, data)

    def test_pad_block_aligned(self) -> None:
        """块对齐数据也需要填充。"""
        data = b"16-byte-message!"
        self.assertEqual(len(data), 16)
        padded = pkcs7_pad(data, 16)
        self.assertEqual(len(padded), 32)  # 增加一个完整块

    def test_unpad_empty_raises(self) -> None:
        """空数据去填充抛出异常。"""
        with self.assertRaises(ValueError):
            pkcs7_unpad(b"", 16)

    def test_unpad_bad_padding_raises(self) -> None:
        """非法的填充字节抛出异常。"""
        with self.assertRaises(ValueError):
            pkcs7_unpad(b"\x00" * 16, 16)


class AESTestCase(unittest.TestCase):
    """测试 AES-CBC 加解密。"""

    def setUp(self) -> None:
        self.key = b"0123456789abcdef"  # 16 bytes for AES-128
        self.iv = b"fedcba9876543210"   # 16 bytes IV

    def test_encrypt_decrypt_roundtrip(self) -> None:
        """加密后解密还原原始明文（调用方负责 PKCS7 填充/去填充）。"""
        plaintext = b"secret message 32bytes long!!"  # 32 bytes
        padded = pkcs7_pad(plaintext, 16)
        ciphertext = aes_cbc_encrypt(self.key, self.iv, padded)
        decrypted_padded = aes_cbc_decrypt(self.key, self.iv, ciphertext)
        decrypted = pkcs7_unpad(decrypted_padded, 16)
        self.assertEqual(decrypted, plaintext)

    def test_encrypt_produces_different_output(self) -> None:
        """加密输出不同于明文。"""
        plaintext = b"test data for enc"  # 恰好 16 字节
        padded = pkcs7_pad(plaintext, 16)
        ciphertext = aes_cbc_encrypt(self.key, self.iv, padded)
        self.assertNotEqual(ciphertext, padded)


if __name__ == "__main__":
    unittest.main()
