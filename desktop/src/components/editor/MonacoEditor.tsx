import { useCallback } from "react";
import Editor, { loader } from "@monaco-editor/react";
import { useWorkspaceStore } from "@/stores/useWorkspaceStore";
import { api } from "@/services/api";

loader.config({
  paths: { vs: "https://cdn.jsdelivr.net/npm/monaco-editor@0.52.0/min/vs" },
});

interface Props {
  path: string;
  language?: string;
  value: string;
}

export function MonacoEditor({ path, language, value }: Props) {
  const updateTabContent = useWorkspaceStore((s) => s.updateTabContent);

  const onChange = useCallback(
    (val: string | undefined) => {
      if (val !== undefined) updateTabContent(path, val);
    },
    [path, updateTabContent]
  );

  const onSave = useCallback(() => {
    const tab = useWorkspaceStore.getState().openTabs.find((t) => t.path === path);
    if (tab?.dirty && tab.content !== undefined) {
      api.fsWrite(path, tab.content).then(() => {
        useWorkspaceStore.getState().markTabDirty(path, false);
      }).catch(() => {});
    }
  }, [path]);

  return (
    <Editor
      height="100%"
      language={language || "plaintext"}
      value={value}
      onChange={onChange}
      theme="glass-dark"
      beforeMount={(monaco) => {
        monaco.editor.defineTheme("glass-dark", {
          base: "vs-dark",
          inherit: true,
          rules: [
            { token: "comment", foreground: "6A9955", fontStyle: "italic" },
            { token: "keyword", foreground: "569CD6" },
            { token: "string", foreground: "CE9178" },
            { token: "number", foreground: "B5CEA8" },
            { token: "type", foreground: "4EC9B0" },
            { token: "function", foreground: "DCDCAA" },
            { token: "variable", foreground: "9CDCFE" },
            { token: "constant", foreground: "4FC1FF" },
          ],
          colors: {
            "editor.background": "#0f172a",
            "editor.foreground": "#e2e8f0",
            "editor.lineHighlightBackground": "#1e293b",
            "editor.selectionBackground": "#334155",
            "editor.inactiveSelectionBackground": "#1e293b",
            "editorCursor.foreground": "#14b8a6",
            "editorLineNumber.foreground": "#475569",
            "editorLineNumber.activeForeground": "#94a3b8",
            "editorWidget.background": "#111827",
            "editorWidget.border": "#475569",
            "input.background": "#1e293b",
            "input.border": "#475569",
            "scrollbarSlider.background": "#33415580",
            "scrollbarSlider.activeBackground": "#47556680",
            "scrollbarSlider.hoverBackground": "#47556660",
          },
        });
      }}
      onMount={(editor, monaco) => {
        editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, onSave);
      }}
      options={{
        fontSize: 14,
        fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
        lineNumbers: "on",
        minimap: { enabled: false },
        scrollBeyondLastLine: false,
        renderWhitespace: "selection",
        tabSize: 2,
        wordWrap: "off",
        smoothScrolling: true,
        cursorBlinking: "smooth",
        cursorSmoothCaretAnimation: "on",
        padding: { top: 8 },
      }}
    />
  );
}
