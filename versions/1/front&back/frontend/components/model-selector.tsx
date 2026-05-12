'use client';

import { ChevronDown, Sparkles, Zap, Code } from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Button } from '@/components/ui/button';
import { useChatContext } from '@/lib/chat-context';
import type { ModelType } from '@/lib/types';

const models: { id: ModelType; name: string; description: string; icon: React.ReactNode }[] = [
  {
    id: 'imagination-1.3',
    name: 'Imagination 1.3',
    description: 'Balanced performance for everyday tasks',
    icon: <Sparkles className="w-4 h-4" />,
  },
  {
    id: 'imagination-1.3-pro',
    name: 'Imagination 1.3 Pro',
    description: 'Advanced reasoning and creativity',
    icon: <Zap className="w-4 h-4" />,
  },
  {
    id: 'imagination-1.3-coder',
    name: 'Imagination 1.3 Coder',
    description: 'Optimized for code generation',
    icon: <Code className="w-4 h-4" />,
  },
];

export function ModelSelector() {
  const { currentModel, setCurrentModel } = useChatContext();
  const fallback = models[0];
  const selectedModel =
    models.find((m) => m.id === currentModel) ?? fallback;

  if (!selectedModel) {
    return null;
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          className="gap-2 text-foreground hover:bg-secondary px-3 py-2 h-auto"
        >
          <span className="text-accent">{selectedModel.icon}</span>
          <span className="font-medium">{selectedModel.name}</span>
          <ChevronDown className="w-4 h-4 text-muted-foreground" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-72">
        {models.map((model) => (
          <DropdownMenuItem
            key={model.id}
            onClick={() => setCurrentModel(model.id)}
            className={`flex items-start gap-3 p-3 cursor-pointer ${
              currentModel === model.id ? 'bg-secondary' : ''
            }`}
          >
            <span className="text-accent mt-0.5">{model.icon}</span>
            <div className="flex flex-col">
              <span className="font-medium">{model.name}</span>
              <span className="text-xs text-muted-foreground">{model.description}</span>
            </div>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
