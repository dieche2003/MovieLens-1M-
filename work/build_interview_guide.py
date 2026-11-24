"""只编排备考文档并检查链接；不读取评分样本、不训练、不修改实验。"""
from pathlib import Path
import json
import re
from datetime import datetime
from html import escape
import markdown

ROOT = Path(r'D:\xiangmu-dianying')
source = ROOT / '面试备考手册.md'
raw = source.read_text(encoding='utf-8')
questions = [int(x) for x in re.findall(r'^### Q(\d+)[：:]', raw, re.M)]
anchors = re.findall(r'<a id="(s\d+)"', raw)
links = re.findall(r'\]\(([^)]+)\)', raw)
missing = [x for x in links if not x.startswith(('#', 'http')) and not (ROOT / x).exists()]
assert questions == list(range(1, 90)), questions
assert anchors == [f's{i:02d}' for i in range(1, 20)], anchors
assert len(re.findall(r'^```', raw, re.M)) % 2 == 0
assert not missing, missing
body = markdown.markdown(raw, extensions=['tables', 'fenced_code', 'sane_lists'])
sections = re.findall(r'^\d+\. \[([^]]+)\]\(#(s\d+)\)', raw, re.M)
nav = ''.join(f'<a href="#{anchor}">{i}. {escape(title)}</a>' for i, (title, anchor) in enumerate(sections, 1))
style = '''
:root{--ink:#172c3b;--muted:#536877;--accent:#086b79;--line:#dce5e8;--paper:#fff;--bg:#f1f5f6}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Microsoft YaHei","PingFang SC",sans-serif;font-size:16px;line-height:1.95}
aside{position:fixed;top:0;left:0;bottom:0;width:276px;overflow:auto;background:#123343;color:#e9f4f5;padding:28px 20px}
aside strong{display:block;font-size:21px;margin-bottom:9px}aside p{font-size:13px;color:#b9d1d7;line-height:1.6}
aside a{display:block;color:#d9e9ed;text-decoration:none;font-size:13px;line-height:1.6;padding:8px 5px;border-bottom:1px solid #31505c}
aside a:hover{background:#254958;color:white}main{max-width:1110px;margin:28px 32px 60px 308px;background:white;padding:44px 55px;border:1px solid var(--line);border-radius:14px}
h1{font-size:31px;line-height:1.5;margin-top:0;color:#123343}h2{font-size:25px;margin-top:55px;border-top:3px solid var(--accent);padding-top:22px;line-height:1.6}
h3{font-size:19px;margin-top:32px;line-height:1.75;color:#145566}p{margin:12px 0}a{color:#087487;text-underline-offset:3px}strong{color:#123d50}
blockquote{margin:18px 0;padding:12px 22px;background:#eff7f8;border-left:4px solid #228f99;color:#284956}
table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.75;margin:20px 0;display:block;overflow-x:auto}
td,th{padding:11px 13px;border:1px solid var(--line);vertical-align:top}th{background:#e9f2f4;text-align:left;color:#16485a}tr:nth-child(even) td{background:#f8fafb}
code{background:#edf2f5;border-radius:4px;padding:2px 5px;font-family:Consolas,monospace;font-size:.88em;overflow-wrap:anywhere}
pre{background:#16303d;color:#e3f0f3;padding:20px;border-radius:8px;overflow:auto;line-height:1.7}pre code{background:none;padding:0;color:inherit}
li{margin:8px 0}footer{margin-top:42px;border-top:1px solid var(--line);padding-top:16px;color:var(--muted);font-size:13px}
@media(min-width:1550px){main{margin-left:calc(276px + (100vw - 1450px)/2)}}
@media(max-width:1000px){aside{position:relative;width:100%;max-height:320px}main{margin:18px;padding:26px}h1{font-size:26px}}
@media print{@page{size:A4;margin:17mm}body{font-size:10pt;background:white;line-height:1.7}aside{display:none}main{margin:0;padding:0;max-width:none;border:0}h1{font-size:22pt}h2{break-before:page;font-size:17pt}h3{break-after:avoid;font-size:12pt}table{display:table;font-size:9pt}tr,blockquote{break-inside:avoid}a{color:inherit;text-decoration:none}pre{white-space:pre-wrap;background:#f1f4f6;color:#123343}footer{font-size:9pt}}
'''
html = f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MovieLens 项目面试备考手册</title><style>{style}</style></head><body><aside><strong>面试备考手册</strong><p>MovieLens-1M · 算法 / 数据科学<br>19 章 · 89 道原理与实现问题<br>先读 1、2、18 章，再逐章理解。<br>可使用浏览器打印并另存 PDF。</p>{nav}</aside><main>{body}<footer>依据本地源码与实验报告核查。HTML 与 Markdown 内容一致；本文件无外部字体、脚本或联网依赖。</footer></main></body></html>'''
target = ROOT / '面试备考手册.html'
target.write_text(html, encoding='utf-8')
report = {'checked_at': datetime.now().isoformat(), 'markdown': str(source), 'html': str(target), 'characters': len(raw), 'questions': len(questions), 'chapters': len(anchors), 'local_links_checked': len([x for x in links if not x.startswith(('#', 'http'))]), 'missing_links': missing, 'question_numbers_contiguous': True, 'anchors_contiguous': True, 'code_fences_balanced': True, 'note': '仅文档结构检查；未修改或重跑模型与实验。'}
(ROOT / 'reports' / 'interview_document_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
