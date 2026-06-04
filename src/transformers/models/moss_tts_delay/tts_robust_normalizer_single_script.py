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

        line = re.sub(r"^#{1,6}\s+", "", line)   # 标题
        line = re.sub(r"^>\s+", "", line)        # 引用
        line = re.sub(r"^[-*+]\s+", "", line)    # 无序列表
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


# ---------------------------
# 测试
# ---------------------------

TEST_CASES = [
    # 1) .map / dot-leading token / 文件名 / 版本号
    (
        "dot_map_sentence",
        "2026 年 3 月 31 日，安全研究员 Chaofan Shou (@Fried_rice) 发现 Anthropic 的 npm 包中暴露了 .map 文件，",
        "2026年3月31日，安全研究员 Chaofan Shou (@Fried_rice) 发现 Anthropic 的 npm 包中暴露了 .map 文件，",
    ),
    ("dot_tokens", "别把 .env、.npmrc、.gitignore 提交上去。", "别把 .env、.npmrc、.gitignore 提交上去。"),
    ("file_names", "请检查 bundle.min.js、package.json 和 processing_moss_tts.py。", "请检查 bundle.min.js、package.json 和 processing_moss_tts.py。"),
    ("index_d_ts", "index.d.ts 里也有同样的问题。", "index.d.ts 里也有同样的问题。"),
    ("version_build", "Bug 的讨论可以精确到 v2.3.1 (Build 15)。", "Bug 的讨论可以精确到 v2.3.1 (Build 15)。"),
    ("version_rc", "3.0.0-rc.1 还不能上生产。", "3.0.0-rc.1 还不能上生产。"),
    ("jar_name", "fabric-api-0.91.3+1.20.2.jar 需要单独下载。", "fabric-api-0.91.3+1.20.2.jar 需要单独下载。"),

    # 2) URL / Email / mention / hashtag / Reddit
    ("url", "仓库地址是 https://github.com/instructkr/claude-code", "仓库地址是 https://github.com/instructkr/claude-code"),
    ("email", "联系邮箱：ops+tts@example.ai", "联系邮箱：ops+tts@example.ai"),
    ("mention", "@Fried_rice 说这是 source map 暴露。", "@Fried_rice 说这是 source map 暴露。"),
    ("reddit", "去 r/singularity 看讨论。", "去 r/singularity 看讨论。"),
    ("hashtag_chain", "#张雪峰#张雪峰[话题]#张雪峰事件", "#张雪峰#张雪峰[话题]#张雪峰事件"),
    ("mention_hashtag_boundary", "关注@biscuit0228_并转发#thetime_tbs", "关注 @biscuit0228_ 并转发 #thetime_tbs"),

    # 3) bracket / 控制 token：保留，不删除
    ("speaker_bracket", "[S1]你好。[S2]收到。", "[S1]你好。[S2]收到。"),
    ("event_bracket", "请模仿 {whisper} 的语气说“别出声”。", "请模仿 {whisper} 的语气说“别出声”。"),
    ("order_bracket", "订单号：[AB-1234-XYZ]", "订单号：[AB-1234-XYZ]"),

    # 4) 结构性符号：替换成边界，而不是直接删除
    ("struct_headline", "〖重磅〗《新品发布》——现在开始！", "重磅。新品发布。现在开始！"),
    ("struct_notice", "【公告】今天 20:00 维护——预计 30 分钟。", "公告。今天20:00维护。预计30分钟。"),
    ("struct_quote_chain", "『特别提醒』「不要外传」", "特别提醒。不要外传。"),
    ("flow_arrow_chain", "请求接入 -> 身份与策略判定 -> 域服务处理", "请求接入，身份与策略判定，域服务处理"),
    ("flow_arrow_no_space", "A->B", "A，B"),
    ("flow_arrow_unicode", "配置中心→推理编排→运行时执行", "配置中心，推理编排，运行时执行"),
    (
        "flow_arrow_maas_example",
        "MaaS 主链遵循请求接入 -> 身份与策略判定 -> 域服务处理 -> 推理编排 -> 运行时执行的在线数据面结构。Dashboard 不直接承载单次在线推理请求，而是负责统一配置、统一治理、统一运营和统一展示。",
        "MaaS 主链遵循请求接入，身份与策略判定，域服务处理，推理编排，运行时执行的在线数据面结构。Dashboard 不直接承载单次在线推理请求，而是负责统一配置、统一治理、统一运营和统一展示。",
    ),

    # 5) 嵌入式标题：保留
    ("embedded_title", "我喜欢《哈姆雷特》这本书。", "我喜欢《哈姆雷特》这本书。"),

    # 6) 重复标点 / 社交噪声
    ("noise_qe", "真的假的？？？！！！", "真的假的？！"),
    ("noise_ellipsis", "这个包把 app.js.map 也发上去了......太离谱了！！！", "这个包把 app.js.map 也发上去了。太离谱了！"),
    ("noise_ellipsis_cn", "【系统提示】请模仿{sad}低沉语气，说“今天下雨了……”", "系统提示。请模仿{sad}低沉语气，说“今天下雨了。”"),

    # 7) 空格规则：英文压缩、中文删除、中英混排保留边界
    ("english_spaces", "This   is   a   test.", "This is a test."),
    ("chinese_spaces", "这 是　一 段  含有多种空白的文本。", "这是一段含有多种空白的文本。"),
    ("mixed_spaces_1", "这是Anthropic的npm包", "这是 Anthropic 的 npm 包"),
    ("mixed_spaces_2", "今天update到v2.3.1了", "今天 update 到 v2.3.1 了"),
    ("mixed_spaces_3", "处理app.js.map文件", "处理 app.js.map 文件"),

    # 8) Markdown / 列表 / 换行
    ("markdown_link", "详情见 [release note](https://github.com/example/release)", "详情见 release note https://github.com/example/release"),
    ("markdown_heading", "# I made a free open source app to help with markdown files", "I made a free open source app to help with markdown files"),
    ("list_lines", "- 修复 .map 泄露\n- 发布 v2.3.1", "修复 .map 泄露。发布 v2.3.1"),
    ("numbered_lines", "1. 安装依赖\n2. 运行测试\n3. 发布 v2.3.1", "安装依赖。运行测试。发布 v2.3.1"),
    ("newlines", "第一行\n第二行\n第三行", "第一行。第二行。第三行"),

    # 9) 零宽字符 / 幂等性
    ("zero_width_url", "详见 https://x.com/\u200bSafety", "详见 https://x.com/Safety"),
]


def run_tests(verbose: bool = True) -> None:
    failed = []

    for name, text, expected in TEST_CASES:
        actual = normalize_tts_text(text)
        if actual != expected:
            failed.append((name, text, expected, actual))
            continue

        # 幂等性：第二次归一化不应继续改动结果
        second = normalize_tts_text(actual)
        if second != actual:
            failed.append((name + "_idempotence", actual, actual, second))

    if failed:
        lines = ["\nTEST FAILED:\n"]
        for name, text, expected, actual in failed:
            lines.append(f"[{name}]")
            lines.append(f"input   : {text}")
            lines.append(f"expected: {expected}")
            lines.append(f"actual  : {actual}")
            lines.append("")
        raise AssertionError("\n".join(lines))

    if verbose:
        print(f"All {len(TEST_CASES)} tests passed.")


if __name__ == "__main__":
    run_tests()
