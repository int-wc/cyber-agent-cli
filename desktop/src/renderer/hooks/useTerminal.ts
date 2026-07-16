import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

interface TerminalSession {
  sessionId: string;
  pid: number;
}

interface TerminalOutputPayload {
  session_id: string;
  data: string;
}

export function useTerminal() {
  const [session, setSession] = useState<TerminalSession | null>(null);
  const outputCallbacks = useRef<Set<(data: string) => void>>(new Set());
  const unlistenRef = useRef<UnlistenFn | null>(null);

  const create = useCallback(async (shell?: string) => {
    // 创建新会话前先清理旧监听器，避免重复写入输出。
    if (unlistenRef.current) {
      unlistenRef.current();
      unlistenRef.current = null;
    }

    const sess = await invoke<TerminalSession>("terminal_create", {
      options: { shell: shell || null },
    });
    setSession(sess);

    // 只监听当前终端会话的输出事件。
    unlistenRef.current = await listen<TerminalOutputPayload>(
      "terminal:output",
      (event) => {
        if (event.payload.session_id === sess.sessionId) {
          outputCallbacks.current.forEach((cb) => cb(event.payload.data));
        }
      },
    );

    return sess;
  }, []);

  const write = useCallback(
    (data: string) => {
      if (!session) return;
      invoke("terminal_write", {
        data: { sessionId: session.sessionId, data },
      }).catch(console.error);
    },
    [session],
  );

  const resize = useCallback(
    (cols: number, rows: number) => {
      if (!session) return;
      invoke("terminal_resize", {
        data: { sessionId: session.sessionId, cols, rows },
      }).catch(console.error);
    },
    [session],
  );

  const kill = useCallback(async () => {
    if (!session) return;
    await invoke("terminal_kill", {
      data: { sessionId: session.sessionId },
    });
    if (unlistenRef.current) {
      unlistenRef.current();
      unlistenRef.current = null;
    }
    setSession(null);
  }, [session]);

  const onOutput = useCallback((cb: (data: string) => void) => {
    outputCallbacks.current.add(cb);
    return () => {
      outputCallbacks.current.delete(cb);
    };
  }, []);

  useEffect(() => {
    create();
    return () => {
      kill();
    };
  }, []);

  return { session, create, write, resize, kill, onOutput };
}
