"""説明資料の md を、PDF に印刷する report/report.html にする。

使い方: python report/build_html.py <omnicampusアカウント名>
  動画と GitHub の URL は環境変数 YOUTUBE_URL・GITHUB_URL（無ければ「提出時に記入」と出す）
"""
import os
import re
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
SRC = HERE / "説明資料_PAI最終課題.md"
OUT = HERE / "report.html"

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: "Hiragino Sans", "Meiryo", sans-serif; font-size: 9.6pt; line-height: 1.65; color: #111; background: #fff; }
h1 { font-size: 17pt; margin: 0 0 4pt; line-height: 1.35; }
h1 + p { margin-top: 0; color: #444; }
h2 { font-size: 12.5pt; border-bottom: 1.2pt solid #111; padding-bottom: 2pt; margin: 16pt 0 6pt; break-after: avoid; }
h3 { font-size: 10.5pt; margin: 10pt 0 4pt; break-after: avoid; }
hr { display: none; }
table { border-collapse: collapse; width: 100%; margin: 5pt 0 8pt; font-size: 8.6pt; break-inside: avoid; }
th, td { border: 0.6pt solid #888; padding: 2.5pt 4pt; vertical-align: top; }
th { background: #e9e9e9; font-weight: 600; }
code { font-family: "Menlo", monospace; font-size: 8.4pt; background: #f2f2f2; padding: 0 2pt; }
ul, ol { margin: 3pt 0 6pt; padding-left: 16pt; }
li { margin: 1pt 0; }
figure { margin: 6pt 0 10pt; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; max-height: 92mm; border: 0.5pt solid #bbb; }
figcaption { font-size: 8.4pt; color: #444; margin-top: 2pt; }
"""


def main() -> None:
    account = sys.argv[1] if len(sys.argv) > 1 else ""
    if not account:
        sys.exit("omnicampus のアカウント名を渡してください")
    text = SRC.read_text(encoding="utf-8")
    subs = {
        "{{ACCOUNT}}": account,
        "{{YOUTUBE_URL}}": os.environ.get("YOUTUBE_URL") or "（提出時に記入）",
        "{{GITHUB_URL}}": os.environ.get("GITHUB_URL") or "（提出時に記入）",
    }
    for k, v in subs.items():
        text = text.replace(k, v)
    left = re.findall(r"\{\{[A-Z_]+\}\}", text)
    if left:
        sys.exit(f"埋めていない欄があります: {left}")

    body = markdown.markdown(text, extensions=["tables"])
    # 段落だけの画像を、見出し付きの図にする（番号は出てくる順）
    n = 0

    def fig(m: re.Match) -> str:
        nonlocal n
        n += 1
        alt, src = m.group(1), m.group(2)
        if not (HERE / src).is_file():
            sys.exit(f"図のファイルがありません: {src}")
        return f'<figure><img src="{src}" alt="{alt}"><figcaption>図 {n}　{alt}</figcaption></figure>'

    body = re.sub(r'<p><img alt="([^"]*)" src="([^"]*)" ?/?></p>',
                  fig, body)
    title = re.search(r"<h1>(.*?)</h1>", body).group(1)
    html = (f'<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>{title}</title>'
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    OUT.write_text(html, encoding="utf-8")
    print(f"{OUT}  図 {n} 枚")


if __name__ == "__main__":
    main()
