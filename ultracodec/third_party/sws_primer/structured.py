# Deterministic, self-authored samples of common structured formats for the v4 primer.
import json, random, sys
rng = random.Random(4)
words = "id name type value status created updated user email title description items count total price url path version enabled default config data error message code level time date source target size key label".split()
names = ["Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Grace", "Heidi"]
out = []
def obj(depth=0):
    d = {}
    for k in rng.sample(words, rng.randint(3, 7)):
        r = rng.random()
        if depth < 2 and r < 0.15: d[k] = [obj(depth + 1) for _ in range(rng.randint(1, 3))]
        elif depth < 2 and r < 0.25: d[k] = obj(depth + 1)
        elif r < 0.45: d[k] = rng.randint(0, 100000)
        elif r < 0.55: d[k] = rng.random() < 0.5
        elif r < 0.62: d[k] = None
        elif r < 0.72: d[k] = round(rng.random() * 1000, 2)
        else: d[k] = rng.choice(names) + " " + rng.choice(words)
    return d
for i in range(40):
    out.append(json.dumps(obj(), indent=2 if i % 2 else None))
out.append(json.dumps({"name": "example-app", "version": "1.4.2", "private": True, "scripts": {"build": "tsc -p .", "test": "jest --coverage", "lint": "eslint src --ext .ts"},
    "dependencies": {"express": "^4.18.2", "react": "^18.2.0", "lodash": "^4.17.21"}, "devDependencies": {"typescript": "^5.2.2", "jest": "^29.7.0"}}, indent=2))
xml = ['<?xml version="1.0" encoding="UTF-8"?>', '<catalog xmlns="http://example.com/schema/catalog" version="2">']
for i in range(60):
    xml.append(f'  <item id="{1000+i}" type="{rng.choice(words)}" enabled="{str(rng.random()<0.7).lower()}">')
    xml.append(f'    <name>{rng.choice(names)} {rng.choice(words)}</name>')
    xml.append(f'    <price currency="USD">{rng.randint(1,999)}.{rng.randint(0,99):02d}</price>')
    xml.append(f'    <description><![CDATA[{rng.choice(words)} {rng.choice(words)} & {rng.choice(words)}]]></description>')
    xml.append('  </item>')
xml.append('</catalog>')
out.append("\n".join(xml))
html = ['<!DOCTYPE html>', '<html lang="en">', '<head>', '  <meta charset="utf-8">', '  <meta name="viewport" content="width=device-width, initial-scale=1">', '  <title>Dashboard</title>', '  <link rel="stylesheet" href="/static/css/main.css">', '</head>', '<body>', '  <nav class="navbar">', '    <ul>']
for w in words[:10]: html.append(f'      <li><a href="/{w}">{w.title()}</a></li>')
html += ['    </ul>', '  </nav>', '  <main class="container">', '    <table class="table">', '      <thead><tr><th>Name</th><th>Status</th><th>Updated</th></tr></thead>', '      <tbody>']
for i in range(40): html.append(f'        <tr><td>{rng.choice(names)}</td><td class="status-{rng.choice(["ok","warn","error"])}">{rng.choice(words)}</td><td>2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}</td></tr>')
html += ['      </tbody>', '    </table>', '  </main>', '  <script src="/static/js/app.js" defer></script>', '</body>', '</html>']
out.append("\n".join(html))
css = []
for w in words[:20]: css.append(f'.{w} {{\n  display: flex;\n  margin: {rng.randint(0,24)}px {rng.randint(0,24)}px;\n  color: #{rng.randint(0,0xffffff):06x};\n}}')
out.append("\n".join(css))
yaml = ['version: "3.9"', 'services:']
for w in ["web", "api", "db", "cache", "worker"]:
    yaml += [f'  {w}:', f'    image: example/{w}:latest', '    restart: unless-stopped', '    environment:', f'      - LOG_LEVEL={rng.choice(["info","debug","warn"])}', f'      - PORT={rng.randint(3000,9000)}', '    ports:', f'      - "{rng.randint(3000,9000)}:{rng.randint(3000,9000)}"']
out.append("\n".join(yaml))
csv = ["id,name,email,status,created,amount"]
for i in range(150): csv.append(f'{i+1},{rng.choice(names)},{rng.choice(names).lower()}@example.com,{rng.choice(["active","inactive","pending"])},2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d},{rng.randint(1,5000)}.{rng.randint(0,99):02d}')
out.append("\n".join(csv))
md = ['# Project', '', '## Installation', '', '```bash', 'pip install -r requirements.txt', 'npm install', '```', '', '## Usage', '', '| Option | Default | Description |', '|---|---|---|']
for w in words[:12]: md.append(f'| `--{w}` | `{rng.randint(0,64)}` | Sets the {w} for each {rng.choice(words)}. |')
out.append("\n".join(md))
sys.stdout.write("\n\n".join(out) + "\n")
