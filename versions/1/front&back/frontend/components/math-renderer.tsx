'use client';

import { memo, useEffect, useState } from 'react';

interface MathRendererProps {
  content: string;
}

interface KaTeXModule {
  renderToString: (latex: string, options: { displayMode: boolean; throwOnError: boolean; output: string; strict: boolean }) => string;
}

// Global KaTeX instance
let katexModule: KaTeXModule | null = null;
let katexLoading: Promise<KaTeXModule> | null = null;

// Load KaTeX from CDN
function loadKaTeX(): Promise<KaTeXModule> {
  if (katexModule) return Promise.resolve(katexModule);
  if (katexLoading) return katexLoading;

  katexLoading = new Promise((resolve, reject) => {
    // Load CSS first
    if (!document.querySelector('link[href*="katex"]')) {
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css';
      document.head.appendChild(link);
    }

    // Load JS
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js';
    script.onload = () => {
      katexModule = (window as unknown as { katex: KaTeXModule }).katex;
      resolve(katexModule);
    };
    script.onerror = reject;
    document.head.appendChild(script);
  });

  return katexLoading;
}

// Render LaTeX to HTML string using KaTeX
function renderMath(katex: KaTeXModule, latex: string, displayMode: boolean): string {
  try {
    return katex.renderToString(latex, {
      displayMode,
      throwOnError: false,
      output: 'html',
      strict: false,
    });
  } catch {
    return `<span class="text-red-400">${latex}</span>`;
  }
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// Parse and render content with math
function processContent(katex: KaTeXModule, text: string): string {
  let result = text;

  // Terminal fences: ```terminal ... ```
  result = result.replace(/```terminal\s*\n([\s\S]*?)```/g, (_, block) => {
    const body = String(block || '').trim();
    const lines = body.split('\n');
    let status = 'Running';
    let statusClass = 'text-amber-300 border-amber-300/40 bg-amber-400/10';
    const first = (lines[0] || '').trim().toLowerCase();
    if (first.startsWith('status:')) {
      const raw = first.replace('status:', '').trim();
      if (raw.includes('success')) {
        status = 'Success';
        statusClass = 'text-emerald-300 border-emerald-300/40 bg-emerald-400/10';
      } else if (raw.includes('fail')) {
        status = 'Fail';
        statusClass = 'text-red-300 border-red-300/40 bg-red-400/10';
      }
      lines.shift();
    }
    const payload = escapeHtml(lines.join('\n'));
    return `<div class="my-3 rounded-xl border border-border bg-card p-3">
      <div class="mb-2">
        <span class="inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-medium ${statusClass}">
          ${status}
        </span>
      </div>
      <pre class="whitespace-pre-wrap break-words rounded-md bg-black/70 p-2 text-xs text-zinc-100">${payload}</pre>
    </div>`;
  });
  
  // First, handle display math ($$...$$) - these should be on their own line
  result = result.replace(/\$\$([\s\S]*?)\$\$/g, (_, math) => {
    const rendered = renderMath(katex, math.trim(), true);
    return `<div class="my-4 overflow-x-auto flex justify-center">${rendered}</div>`;
  });
  
  // Then handle inline math ($...$)
  result = result.replace(/\$([^\$\n]+?)\$/g, (_, math) => {
    const rendered = renderMath(katex, math.trim(), false);
    return rendered;
  });
  
  // Format markdown-like syntax
  // Bold
  result = result.replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold">$1</strong>');
  // Italic (not preceded or followed by *)
  result = result.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '<em>$1</em>');
  // Inline code
  result = result.replace(/`([^`]+)`/g, '<code class="bg-secondary/50 px-1.5 py-0.5 rounded text-sm font-mono text-accent">$1</code>');
  // Line breaks
  result = result.replace(/\n/g, '<br/>');
  
  return result;
}

export const MathRenderer = memo(function MathRenderer({ content }: MathRendererProps) {
  const [renderedContent, setRenderedContent] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let mounted = true;

    loadKaTeX().then((katex) => {
      if (mounted) {
        const processed = processContent(katex, content);
        setRenderedContent(processed);
        setIsLoading(false);
      }
    }).catch(() => {
      if (mounted) {
        // Fallback: just show content with basic formatting
        let result = content;
        result = result.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        result = result.replace(/\n/g, '<br/>');
        setRenderedContent(result);
        setIsLoading(false);
      }
    });

    return () => { mounted = false; };
  }, [content]);

  if (isLoading) {
    return (
      <div className="prose prose-invert max-w-none text-foreground leading-relaxed animate-pulse">
        <div className="h-4 bg-secondary/30 rounded w-3/4 mb-2"></div>
        <div className="h-4 bg-secondary/30 rounded w-1/2"></div>
      </div>
    );
  }

  return (
    <div 
      className="prose prose-invert max-w-none text-foreground leading-relaxed [&_.katex]:text-foreground [&_.katex-display]:my-4 [&_.katex-display]:overflow-x-auto"
      dangerouslySetInnerHTML={{ __html: renderedContent || '' }}
    />
  );
});
