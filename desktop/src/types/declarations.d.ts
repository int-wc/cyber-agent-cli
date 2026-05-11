declare module "lucide-react" {
  import type { FC } from "react";

  interface LucideProps {
    size?: number | string;
    className?: string;
    color?: string;
    fill?: string;
    strokeWidth?: number;
    absoluteStrokeWidth?: boolean;
  }

  export const Files: FC<LucideProps>;
  export const Search: FC<LucideProps>;
  export const GitBranch: FC<LucideProps>;
  export const Puzzle: FC<LucideProps>;
  export const Settings: FC<LucideProps>;
  export const Circle: FC<LucideProps>;
  export const X: FC<LucideProps>;
  export const Send: FC<LucideProps>;
  export const Square: FC<LucideProps>;
  export const Wrench: FC<LucideProps>;
  export const User: FC<LucideProps>;
  export const Bot: FC<LucideProps>;
  export const Brain: FC<LucideProps>;
  export const AlertCircle: FC<LucideProps>;
  export const AlertTriangle: FC<LucideProps>;
  export const MessageSquare: FC<LucideProps>;
  export const FolderOpen: FC<LucideProps>;
  export const Folder: FC<LucideProps>;
  export const File: FC<LucideProps>;
  export const FileCode: FC<LucideProps>;
  export const ChevronRight: FC<LucideProps>;
  export const RefreshCw: FC<LucideProps>;
  export const Plus: FC<LucideProps>;
  export const Minus: FC<LucideProps>;
  export const Terminal: FC<LucideProps>;
}

declare module "xterm" {
  export class Terminal {
    constructor(options?: Record<string, unknown>);
    open(el: HTMLElement): void;
    write(data: string): void;
    writeln(data: string): void;
    loadAddon(addon: unknown): void;
    onData(cb: (data: string) => void): void;
    dispose(): void;
    element?: HTMLElement;
  }
}

declare module "xterm-addon-fit" {
  export class FitAddon {
    activate(term: unknown): void;
    dispose(): void;
    fit(): void;
  }
}

declare module "xterm/css/xterm.css" {
  const css: string;
  export default css;
}
