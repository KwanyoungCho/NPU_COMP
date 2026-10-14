"""Serve the existing presentation locally; never builds the compiler or deck.

Requires PyMuPDF. Only presentation artifacts and explicitly cited documents
are exposed. PDF slide images are cached in memory, not duplicated in docs/.
"""
import argparse
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
from urllib.parse import unquote, urlsplit

import fitz

ROOT = Path(__file__).resolve().parent
PDF = ROOT/'npu_compiler_explained.pdf'
RENDER_LOCK = threading.Lock()


def slide_data():
    manifest = json.loads((ROOT/'sources.json').read_text())
    notes = (ROOT/'speaker_notes.md').read_text()
    sections = re.split(r'^## (\d+)\. [^\n]+\n', notes, flags=re.M)
    bodies = {int(sections[i]): sections[i+1].split('\n근거:', 1)[0].strip()
              for i in range(1, len(sections), 2)}
    with RENDER_LOCK, fitz.open(PDF) as document:
        if len(document) != len(manifest['slides']):
            raise ValueError('PDF and sources.json have different slide counts')
    version = str(PDF.stat().st_mtime_ns)
    slides = [dict(item, notes=bodies.get(item['number'], ''),
                   image=f"slides/{item['number']}.png?v={version}")
              for item in manifest['slides']]
    return dict(slides=slides, main_count=sum(not s['appendix'] for s in slides))


@lru_cache(maxsize=48)
def render_slide(number, version):
    # PyMuPDF does not support concurrent use; serialize rasterization.
    with RENDER_LOCK, fitz.open(PDF) as document:
        if not 1 <= number <= len(document):
            raise IndexError(number)
        page = document[number-1]
        return page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).tobytes('png')


class Handler(BaseHTTPRequestHandler):
    def reply(self, payload, mime, status=200, cache='no-cache', attachment=None):
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', cache)
        self.send_header('X-Content-Type-Options', 'nosniff')
        if attachment:self.send_header('Content-Disposition', f'attachment; filename="{attachment}"')
        self.end_headers()
        if self.command != 'HEAD':
            try:self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):pass

    def do_HEAD(self):self.do_GET()

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path in ('/', '/index.html'):
            return self.reply((ROOT/'viewer.html').read_bytes(), 'text/html; charset=utf-8')
        if path == '/api/slides':
            return self.reply(json.dumps(slide_data(), ensure_ascii=False).encode(), 'application/json; charset=utf-8')
        match = re.fullmatch(r'/slides/([0-9]+)\.png', path)
        if match:
            try:
                payload = render_slide(int(match[1]), PDF.stat().st_mtime_ns)
            except IndexError:
                return self.reply(b'Slide not found', 'text/plain', 404)
            return self.reply(payload, 'image/png', cache='public, max-age=3600')
        downloads = {
            '/deck.pdf': ('npu_compiler_explained.pdf', 'application/pdf', False),
            '/deck.pptx': ('npu_compiler_explained.pptx', 'application/vnd.openxmlformats-officedocument.presentationml.presentation', True),
            '/notes.md': ('speaker_notes.md', 'text/plain; charset=utf-8', True),
        }
        if path in downloads:
            name, mime, download = downloads[path]
            return self.reply((ROOT/name).read_bytes(), mime, attachment=name if download else None)
        if path.startswith('/source/'):
            relative = path[len('/source/'):]
            allowed = {f['path'] for f in json.loads((ROOT/'sources.json').read_text())['files']}
            if relative in allowed:
                return self.reply((ROOT.parent/relative).read_bytes(), 'text/plain; charset=utf-8')
        return self.reply(b'Not found', 'text/plain; charset=utf-8', 404)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    data = slide_data()
    with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
        print(f"Presentation viewer: http://localhost:{args.port} ({len(data['slides'])} slides)", flush=True)
        try:server.serve_forever()
        except KeyboardInterrupt:pass


if __name__ == '__main__':main()
