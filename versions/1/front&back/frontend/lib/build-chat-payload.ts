/** Shared body shape for POST /api/chat and /api/chat/stream (proxied to Imagination). */

export type ChatRole = 'user' | 'assistant';

function pickTextField(body: Record<string, unknown>, keys: string[]): string {
  for (const k of keys) {
    const v = body[k];
    if (typeof v === 'string' && v.trim()) {
      return v.trim();
    }
  }
  return '';
}

function lastUserContent(messages: { role: string; content: string }[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role === 'user' && typeof m.content === 'string' && m.content.trim()) {
      return m.content.trim();
    }
  }
  return '';
}

export function buildChatRequestPayload(body: Record<string, unknown>) {
  let prompt = pickTextField(body, ['prompt', 'message', 'text', 'query', 'q']);

  const currentModel =
    (typeof body.currentModel === 'string' && body.currentModel.trim() && body.currentModel) ||
    (typeof body.model === 'string' && body.model.trim() && body.model) ||
    'imagination-1.3';

  let messages: { role: string; content: string }[] = [];
  if (Array.isArray(body.messages) && body.messages.length > 0) {
    messages = (body.messages as { role?: string; content?: string }[])
      .filter(m => m && typeof m.content === 'string')
      .map(m => ({
        role: (m.role as ChatRole) || 'user',
        content: m.content as string,
      }));
  } else if (prompt) {
    messages = [{ role: 'user', content: prompt }];
  }

  if (!prompt && messages.length > 0) {
    prompt = lastUserContent(messages);
  }

  let image: string | undefined;
  if (typeof body.image === 'string' && body.image.trim()) {
    image = body.image;
  } else if (Array.isArray(body.attachments)) {
    const firstImage = (body.attachments as Array<{ type?: unknown; url?: unknown }>).find(
      a => a && a.type === 'image' && typeof a.url === 'string' && a.url
    );
    if (firstImage && typeof firstImage.url === 'string') {
      image = firstImage.url;
    }
  }

  const out: Record<string, unknown> = { prompt, currentModel, messages };
  if (image) out.image = image;
  if (Array.isArray(body.attachments)) {
    out.attachments = body.attachments;
  }
  if (typeof body.max_new_tokens === 'number' && Number.isFinite(body.max_new_tokens)) {
    out.max_new_tokens = body.max_new_tokens;
  }
  if (body.ui_artisan_mode === true) {
    out.ui_artisan_mode = true;
  }
  if (body.allow_network_tools === false) {
    out.allow_network_tools = false;
  }
  return out;
}
