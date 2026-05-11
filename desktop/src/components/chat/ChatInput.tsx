import { useState, useRef, useCallback, KeyboardEvent } from "react";
import { Send, Square } from "lucide-react";

interface Props {
  onSend: (text: string) => void;
  onStop: () => void;
  streaming: boolean;
}

export function ChatInput({ onSend, onStop, streaming }: Props) {
  const [text, setText] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = useCallback(() => {
    const trimmed = text.trim();
    if (!trimmed || streaming) return;
    onSend(trimmed);
    setText("");
  }, [text, streaming, onSend]);

  const onKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  return (
    <div className="p-3 border-t border-border-glass">
      <div className="flex items-end gap-2">
        <textarea
          ref={inputRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={streaming ? "AI 回复中..." : "输入消息... (Enter 发送, Shift+Enter 换行)"}
          rows={1}
          disabled={streaming}
          className="flex-1 glass-input resize-none text-xs min-h-[32px] max-h-[120px]"
          style={{ height: Math.min(120, Math.max(32, text.split("\n").length * 18 + 8)) }}
        />
        {streaming ? (
          <button
            onClick={onStop}
            className="glass-button p-2 rounded-glass-sm flex-shrink-0"
            title="停止生成"
          >
            <Square size={14} fill="#ef4444" color="#ef4444" />
          </button>
        ) : (
          <button
            onClick={handleSend}
            disabled={!text.trim()}
            className="glass-button glass-button-accent p-2 rounded-glass-sm flex-shrink-0"
            title="发送"
          >
            <Send size={14} />
          </button>
        )}
      </div>
    </div>
  );
}
