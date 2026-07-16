import { memo } from "react";

interface StreamingTextProps {
  content: string;
  isStreaming?: boolean;
}

function StreamingTextInner({ content, isStreaming }: StreamingTextProps) {
  if (!content) return null;

  // 轻量 Markdown 渲染：支持代码块、行内代码、粗体和斜体。
  const renderLine = (line: string, inCodeBlock: boolean): [string, boolean] => {
    if (line.startsWith("```")) {
      return [inCodeBlock ? "</code></pre>" : "<pre><code>", !inCodeBlock];
    }
    if (inCodeBlock) {
      return [escapeHtml(line) + "\n", true];
    }
    let html = escapeHtml(line);
    // 行内代码
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    // 粗体
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    // 斜体
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

  // 补齐未闭合的代码块，避免后续内容被吞进 pre/code。
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
