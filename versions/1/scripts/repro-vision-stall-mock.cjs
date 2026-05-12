/**
 * See v0-imagination-ui/scripts/repro-vision-stall-mock.cjs (same file).
 * Copy into Colab or run locally: node versions/v1.3/scripts/repro-vision-stall-mock.cjs
 */
/* eslint-disable @typescript-eslint/no-require-imports */
const http = require('http');

const PORT = Number(process.env.REPRO_STALL_PORT || 18765);

const server = http.createServer((req, res) => {
  const url = (req.url || '').split('?')[0];
  if (req.method === 'GET' && (url === '/health' || url === '/api/health')) {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(
      JSON.stringify({
        status: 'ok',
        service: 'imagination-v1.3',
        model_name: 'repro-vision-stall-mock',
        ok: true,
        backend_version: 'repro-mock',
        vision_mode: 'clip_projector',
        clip_projector_active: true,
      })
    );
    return;
  }

  if (req.method === 'POST' && url === '/api/chat/stream') {
    req.on('end', () => {
      res.writeHead(200, {
        'Content-Type': 'application/x-ndjson',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
        Connection: 'keep-alive',
      });
      if (typeof res.flushHeaders === 'function') {
        res.flushHeaders();
      }
    });
    req.resume();
    return;
  }

  if (req.method === 'POST' && url === '/api/chat') {
    req.on('data', () => {});
    req.on('end', () => {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(
        JSON.stringify({ response: 'repro: use /api/chat/stream for this test', model: 'repro' })
      );
    });
    return;
  }

  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'not found', path: url }));
});

server.headersTimeout = 0;
server.requestTimeout = 0;
server.listen(PORT, '127.0.0.1', () => {
  console.log(`[repro-vision-stall-mock] http://127.0.0.1:${PORT}  (POST /api/chat/stream stalls after headers)`);
});
