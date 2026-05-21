import { memo } from "react";

interface StreamingTextProps {
  content: string;
  isStreaming?: boolean;
}

function StreamingTextInner({ content, isStreaming }: StreamingTextProps) {
  if (!content) return null;

  // Simple markdown rendering: code blocks, inline code, bold, italic
  const renderLine = (line: string, inCodeBlock: boolean): [string, boolean] => {
    if (line.startsWith("```")) {
      return [inCodeBlock ? "</code></pre>" : "<pre><code>", !inCodeBlock];
    }
    if (inCodeBlock) {
      return [escapeHtml(line) + "\n", true];
    }
    // Inline code
    let html = line.replace(/`([^`]+)`/g, "<code>$1</code>");
    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    // Italic
    html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    return [html, false];
  };

  const lines = content.split("\n");
  let inCode = false;
  const htmlLines = lines.map((line) => {
    const [html, newInCode] = renderLine(line, inCode);
    inCode = newInCode;
    return html;
  });

  // Close any remaining code block
  if (inCode) {
    htmlLines.push("</code></pre>");
  }

  return (
    <div
      style={{
        fontSize: 13,
        lineHeight: 1.7,
        whiteSpace: "pre-wrap",
        wordBreak: "break-word",
      }}
    >
      {htmlLines.map((html, i) => (
        <span
          key={i}
          dangerouslySetInnerHTML={{ __html: html }}
          style={{ display: "block" }}
        />
      ))}
      {isStreaming && (
        <span style={{
          animation: "blink 1s step-end infinite",
          color: "var(--accent-light)",
          fontWeight: 700,
        }}>
          ▌
        </span>
      )}
    </div>
  );
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

export default memo(StreamingTextInner);
