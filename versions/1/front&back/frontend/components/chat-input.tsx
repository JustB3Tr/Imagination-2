'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, Mic, X, FileText, FileCode, Image as ImageIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useChatContext } from '@/lib/chat-context';
import type { Attachment } from '@/lib/types';

// Accepted file types
const ACCEPTED_TYPES = {
  image: ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/svg+xml'],
  document: ['text/plain', 'text/markdown', 'application/pdf', 'text/csv'],
  code: [
    'text/javascript', 'application/javascript',
    'text/typescript', 'application/typescript',
    'text/html', 'text/css', 'application/json',
    'text/x-python', 'application/x-python',
    'text/x-java', 'text/x-c', 'text/x-c++',
  ],
};

const CODE_EXTENSIONS = ['.js', '.jsx', '.ts', '.tsx', '.py', '.java', '.c', '.cpp', '.h', '.css', '.html', '.json', '.md', '.xml', '.yaml', '.yml', '.sh', '.sql', '.go', '.rs', '.rb', '.php'];
const DOC_EXTENSIONS = ['.txt', '.md', '.csv', '.pdf'];
const IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg'];

function getFileType(file: File): 'image' | 'document' | 'code' {
  const ext = '.' + file.name.split('.').pop()?.toLowerCase();
  
  if (IMAGE_EXTENSIONS.includes(ext) || ACCEPTED_TYPES.image.includes(file.type)) {
    return 'image';
  }
  if (CODE_EXTENSIONS.includes(ext)) {
    return 'code';
  }
  if (DOC_EXTENSIONS.includes(ext) || ACCEPTED_TYPES.document.includes(file.type)) {
    return 'document';
  }
  // Default to document for unknown text files
  if (file.type.startsWith('text/')) {
    return 'document';
  }
  return 'document';
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

export function ChatInput() {
  const [input, setInput] = useState('');
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { sendChatToBackend, sendResearchToBackend, isAgentRunning } = useChatContext();

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    const newAttachments: Attachment[] = [];

    for (const file of Array.from(files)) {
      // Max 10MB per file
      if (file.size > 10 * 1024 * 1024) {
        alert(`File "${file.name}" is too large. Maximum size is 10MB.`);
        continue;
      }

      const fileType = getFileType(file);
      
      // Read file content
      const url = await new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        if (fileType === 'image') {
          reader.readAsDataURL(file);
        } else {
          reader.readAsDataURL(file);
        }
      });

      // For text files, also read the content
      let content: string | undefined;
      if (fileType === 'code' || fileType === 'document') {
        content = await new Promise<string>((resolve) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result as string);
          reader.readAsText(file);
        });
      }

      newAttachments.push({
        id: Math.random().toString(36).substring(2, 15),
        name: file.name,
        type: fileType,
        mimeType: file.type,
        size: file.size,
        url,
        content,
      });
    }

    setAttachments(prev => [...prev, ...newAttachments]);
    
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeAttachment = (id: string) => {
    setAttachments(prev => prev.filter(a => a.id !== id));
  };

  const handleSubmit = async () => {
    if ((!input.trim() && attachments.length === 0) || isLoading || isAgentRunning) return;

    const userMessage = input.trim();
    if (/^\/research(\s|$)/i.test(userMessage)) {
      const topicRest = userMessage.replace(/^\/research\s*/i, '').trim();
      if (!topicRest && attachments.length === 0) {
        alert('Add a topic after /research — e.g. /research Who is Alice?');
        return;
      }
    }

    const currentAttachments = [...attachments];
    setInput('');
    setAttachments([]);
    setIsLoading(true);

    try {
      if (/^\/research(\s|$)/i.test(userMessage)) {
        await sendResearchToBackend(
          userMessage || 'Attached files',
          currentAttachments.length > 0 ? currentAttachments : undefined
        );
      } else {
        await sendChatToBackend(
          userMessage || 'Attached files',
          currentAttachments.length > 0 ? currentAttachments : undefined
        );
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const getAttachmentIcon = (type: Attachment['type']) => {
    switch (type) {
      case 'image': return <ImageIcon className="w-4 h-4" />;
      case 'code': return <FileCode className="w-4 h-4" />;
      default: return <FileText className="w-4 h-4" />;
    }
  };

  return (
    <div className="border-t border-border bg-background p-4">
      <div className="max-w-3xl mx-auto">
        {/* Attachment previews */}
        {attachments.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-3">
            {attachments.map((attachment) => (
              <div
                key={attachment.id}
                className="relative group flex items-center gap-2 bg-secondary rounded-lg p-2 pr-8"
              >
                {attachment.type === 'image' ? (
                  <img
                    src={attachment.url}
                    alt={attachment.name}
                    className="w-10 h-10 object-cover rounded"
                  />
                ) : (
                  <div className="w-10 h-10 flex items-center justify-center bg-muted rounded">
                    {getAttachmentIcon(attachment.type)}
                  </div>
                )}
                <div className="flex flex-col min-w-0">
                  <span className="text-sm text-foreground truncate max-w-[150px]">
                    {attachment.name}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {formatFileSize(attachment.size)}
                  </span>
                </div>
                <button
                  onClick={() => removeAttachment(attachment.id)}
                  className="absolute top-1 right-1 p-1 rounded-full bg-muted hover:bg-destructive hover:text-destructive-foreground transition-colors"
                  aria-label={`Remove ${attachment.name}`}
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="relative flex items-end gap-2 bg-card rounded-2xl border border-border p-2">
          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".jpg,.jpeg,.png,.gif,.webp,.svg,.txt,.md,.pdf,.csv,.js,.jsx,.ts,.tsx,.py,.java,.c,.cpp,.h,.css,.html,.json,.xml,.yaml,.yml,.sh,.sql,.go,.rs,.rb,.php"
            onChange={handleFileSelect}
            className="hidden"
          />
          
          <Button
            variant="ghost"
            size="icon"
            className="shrink-0 text-muted-foreground hover:text-foreground"
            aria-label="Attach file"
            onClick={() => fileInputRef.current?.click()}
          >
            <Paperclip className="w-5 h-5" />
          </Button>

          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message Imagination..."
            rows={1}
            className="flex-1 resize-none bg-transparent border-0 focus:outline-none focus:ring-0 text-foreground placeholder:text-muted-foreground py-2 px-2 max-h-[200px]"
          />

          <div className="flex items-center gap-1 shrink-0">
            <Button
              variant="ghost"
              size="icon"
              className="text-muted-foreground hover:text-foreground"
              aria-label="Voice input"
            >
              <Mic className="w-5 h-5" />
            </Button>

            <Button
              onClick={handleSubmit}
              disabled={(!input.trim() && attachments.length === 0) || isLoading || isAgentRunning}
              size="icon"
              className="bg-accent text-accent-foreground hover:bg-accent/90 disabled:opacity-50"
              aria-label="Send message"
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </div>

        <p className="text-center text-xs text-muted-foreground mt-3">
          {isAgentRunning
            ? 'Agent loop running: review terminal and diff cards as events stream in.'
            : 'Imagination AI can make mistakes. Consider checking important information.'}
        </p>
      </div>
    </div>
  );
}
