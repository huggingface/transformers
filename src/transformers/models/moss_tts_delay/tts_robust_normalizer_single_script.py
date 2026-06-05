#!/usr/bin/env python3

"""
TTS 输入鲁棒性正则化器（非语义 TN）

目标
----
1. 只做“鲁棒性清洗”，不做数字/单位/日期/金额等语义展开。
2. 优先保护高风险 token，避免把 `.map`、`app.js.map`、`v2.3.1`、URL、Email、@mention、#hashtag 清坏。
3. 保留 `[]` / `{}` 的内容，为后续声音事件、控制标签留接口。
4. 对结构性符号做“替换而非删除”：
   - `【】 / 〖〗 / 『』 / 「」` 在结构位置转成句边界。
   - `《》` 只在“独立标题/栏目名”场景拆开；嵌入式标题保持不变。
   - `—— / -- / ——...` 转成句边界。
   - `-> / => / →` 等流程连接符转成中文逗号，避免 TTS 读入崩溃。
5. 对社交平台常见噪声做弱归一化：
   - `...... / ……` -> `。`
   - `？？？！！！` -> `？！`
   - `！！！` -> `！`
6. 空格按脚本类型处理：
   - 西文片段内部：连续空格压缩为 1 个。
   - 汉字 / 日文假名片段内部：删除空格。
   - 汉字 / 日文假名 与“拉丁字母类 token / 受保护 token”相邻：保留或补 1 个空格。
   - 汉字 / 日文假名 与纯数字相邻：不强行补空格。
7. 轻量处理 Markdown 与换行：
   - `[text](url)` -> `text url`
   - 去掉标题 `#`、引用 `>`、列表前缀
   - 换行转句边界 `。`

非目标
------
1. 不决定“应该怎么读”。
2. 不删除 `[] / {}` 内容。
3. 不做 HTML/SSML/语义标签解释。
"""

from __future__ import annotations

import re
import unicodedata


# ---------------------------
# 基础常量与正则
# ---------------------------

# 不依赖空格分词的脚本：汉字 + 日文假名
_CJK_CHARS = r"\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff"
_CJK = f"[{_CJK_CHARS}]"

# 保护占位符
_PROT = r"___PROT\d+___"

# 需要保护的高风险 token
_URL_RE = re.compile(r"https?://[^\s\u3000，。！？；、）】》〉」』]+")
_EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])")
_MENTION_RE = re.compile(r"(?<![A-Za-z0-9_])@[A-Za-z0-9_]{1,32}")
_REDDIT_RE = re.compile(r"(?<![A-Za-z0-9_])(?:u|r)/[A-Za-z0-9_]+")
_HASHTAG_RE = re.compile(r"(?<![A-Za-z0-9_])#(?!\s)[^\s#]+")

# `.map` / `.env` / `.gitignore`
_DOT_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])\.(?=[A-Za-z0-9._-]*[A-Za-z0-9])[A-Za-z0-9._-]+")

# `app.js.map` / `index.d.ts` / `v2.3.1` / `foo/bar-baz.py` 等
_FILELIKE_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?=[A-Za-z0-9._/+:-]*[A-Za-z])"
    r"(?=[A-Za-z0-9._/+:-]*[._/+:-])"
    r"[A-Za-z0-9](?:[A-Za-z0-9._/+:-]*[A-Za-z0-9])?"
    r"(?![A-Za-z0-9_])"
)

# 参与“中英混排边界补空格”的 token：必须至少含 1 个拉丁字母，或本身就是受保护 token
_LATINISH = rf"(?:{_PROT}|(?=[A-Za-z0-9._/+:-]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9._/+:-]*)"

# 零宽字符
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\ufeff]")


# ---------------------------
# 主函数
# ---------------------------


def normalize_tts_text(text: str) -> str:
    """对 TTS 输入做鲁棒性正则化。"""
    text = _base_cleanup(text)
    text = _normalize_markdown_and_lines(text)
    text, protected = _protect_spans(text)

    text = _normalize_spaces(text)
    text = _normalize_structural_punctuation(text)
    text = _normalize_repeated_punctuation(text)
    text = _normalize_spaces(text)

    text = _restore_spans(text, protected)
    return text.strip()


# ---------------------------
# 具体规则
# ---------------------------


def _base_cleanup(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u3000", " ")
    text = _ZERO_WIDTH_RE.sub("", text)

    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in "\n\t " or not cat.startswith("C"):
            cleaned.append(ch)
    return "".join(cleaned)


def _normalize_markdown_and_lines(text: str) -> str:
    # Markdown 链接：[text](url) -> text url
    text = re.sub(r"\[([^\[\]]+?)\]\((https?://[^)\s]+)\)", r"\1 \2", text)

    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        line = re.sub(r"^#{1,6}\s+", "", line)  # 标题
        line = re.sub(r"^>\s+", "", line)  # 引用
        line = re.sub(r"^[-*+]\s+", "", line)  # 无序列表
        line = re.sub(r"^\d+[.)]\s+", "", line)  # 有序列表
        lines.append(line)

    return "。".join(lines) if lines else ""


def _protect_spans(text: str) -> tuple[str, list[str]]:
    protected: list[str] = []

    def repl(match: re.Match[str]) -> str:
        idx = len(protected)
        protected.append(match.group(0))
        return f"___PROT{idx}___"

    for pattern in (
        _URL_RE,
        _EMAIL_RE,
        _MENTION_RE,
        _REDDIT_RE,
        _HASHTAG_RE,
        _DOT_TOKEN_RE,
        _FILELIKE_RE,
    ):
        text = pattern.sub(repl, text)

    return text, protected


def _restore_spans(text: str, protected: list[str]) -> str:
    for idx, original in enumerate(protected):
        text = text.replace(f"___PROT{idx}___", original)
    return text


def _normalize_spaces(text: str) -> str:
    # 统一空白
    text = re.sub(r"[ \t\r\f\v]+", " ", text)

    # 汉字 / 日文片段内部：删除空格
    text = re.sub(rf"({_CJK})\s+(?={_CJK})", r"\1", text)

    # 汉字 / 日文 与纯数字之间：删除空格（不强行保留）
    text = re.sub(rf"({_CJK})\s+(?=\d)", r"\1", text)
    text = re.sub(rf"(\d)\s+(?={_CJK})", r"\1", text)

    # 汉字 / 日文 与拉丁字母类 token / protected token 相邻：保留或补 1 个空格
    text = re.sub(rf"({_CJK})(?=({_LATINISH}))", r"\1 ", text)
    text = re.sub(rf"(({_LATINISH}))(?={_CJK})", r"\1 ", text)

    # 再压一遍连续空格
    text = re.sub(r" {2,}", " ", text)

    # 中文标点前后不保留空格
    text = re.sub(r"\s+([，。！？；：、”’」』】）》])", r"\1", text)
    text = re.sub(r"([（【「『《“‘])\s+", r"\1", text)
    text = re.sub(r"([，。！？；：、])\s*", r"\1", text)

    # ASCII 标点前不留空格；后面的英文空格不强改
    text = re.sub(r"\s+([,.;!?])", r"\1", text)

    return re.sub(r" {2,}", " ", text).strip()


def _normalize_structural_punctuation(text: str) -> str:
    # 结构性括号：在“结构位置”解包并转成句边界
    # 连续块要支持收敛，因此做两轮
    for _ in range(2):
        text = re.sub(
            r"(^|[。！？!?；;]\s*)[【〖『「]([^】〗』」]+)[】〗』」]\s*",
            r"\1\2。",
            text,
        )

    # 《》只处理独立标题，不处理嵌入式标题
    # 例：重磅。《新品发布》——现在开始！ -> 重磅。新品发布。现在开始！
    text = re.sub(
        r"(^|[。！？!?；;]\s*)《([^》]+)》(?=\s*(?:___PROT\d+___|[—–―-]{2,}|$|[。！？!?；;，,]))",
        r"\1\2",
        text,
    )

    # 流程 / 映射箭头：转成中文逗号，保留链路结构但避免把 `->` 原样喂给 TTS。
    text = re.sub(
        r"\s*(?:<[-=]+>|[-=]+>|<[-=]+|[→←↔⇒⇐⇔⟶⟵⟷⟹⟸⟺↦↤↪↩])\s*",
        "，",
        text,
    )

    # 长破折号 / 多连字符：转句边界
    text = re.sub(r"\s*(?:—|–|―|-){2,}\s*", "。", text)

    return text


def _normalize_repeated_punctuation(text: str) -> str:
    # 省略号 / 连续句点
    text = re.sub(r"(?:\.{3,}|…{2,}|……+)", "。", text)

    # 同类重复标点
    text = re.sub(r"[。．]{2,}", "。", text)
    text = re.sub(r"[，,]{2,}", "，", text)
    text = re.sub(r"[!！]{2,}", "！", text)
    text = re.sub(r"[?？]{2,}", "？", text)

    # 混合问叹号：收敛到 ？！
    def _mixed_qe(match: re.Match[str]) -> str:
        s = match.group(0)
        has_q = any(ch in s for ch in "?？")
        has_e = any(ch in s for ch in "!！")
        if has_q and has_e:
            return "？！"
        return "？" if has_q else "！"

    text = re.sub(r"[!?！？]{2,}", _mixed_qe, text)
    return text
