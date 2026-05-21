import { useCallback, useEffect, useRef, useState } from "react";

interface TerminalSession {
  sessionId: string;
  pid: number;
}

export function useTerminal() {
  const [session, setSession] = useState<TerminalSession | null>(null);
  const outputCallbacks = useRef<Set<(data: string) => void>>(new Set());

  const create = useCallback(async (shell?: string) => {
    if (!window.electronAPI) return;
    const sess = await window.electronAPI.terminalCreate(shell);
    setSession(sess);

    window.electronAPI.onTerminalOutput((sessId, data) => {
      if (sessId === sess.sessionId) {
        outputCallbacks.current.forEach((cb) => cb(data));
      }
    });

    return sess;
  }, []);

  const write = useCallback((data: string) => {
    if (!session || !window.electronAPI) return;
    window.electronAPI.terminalWrite(session.sessionId, data);
  }, [session]);

  const resize = useCallback((cols: number, rows: number) => {
    if (!session || !window.electronAPI) return;
    window.electronAPI.terminalResize(session.sessionId, cols, rows);
  }, [session]);

  const kill = useCallback(async () => {
    if (!session || !window.electronAPI) return;
    await window.electronAPI.terminalKill(session.sessionId);
    setSession(null);
  }, [session]);

  const onOutput = useCallback((cb: (data: string) => void) => {
    outputCallbacks.current.add(cb);
    return () => { outputCallbacks.current.delete(cb); };
  }, []);

  useEffect(() => {
    // Auto-create terminal session
    create();
    return () => { kill(); };
  }, []);

  return { session, create, write, resize, kill, onOutput };
}
