declare module "lucide-react" {
  import type { FC, SVGProps } from "react";
  export const Files: FC<SVGProps<SVGSVGElement>>;
  export const Search: FC<SVGProps<SVGSVGElement>>;
  export const GitBranch: FC<SVGProps<SVGSVGElement>>;
  export const Puzzle: FC<SVGProps<SVGSVGElement>>;
  export const Settings: FC<SVGProps<SVGSVGElement>>;
  export const Circle: FC<SVGProps<SVGSVGElement>>;
  export const X: FC<SVGProps<SVGSVGElement>>;
  export const Send: FC<SVGProps<SVGSVGElement>>;
  export const Square: FC<SVGProps<SVGSVGElement>>;
  export const Wrench: FC<SVGProps<SVGSVGElement>>;
  export const User: FC<SVGProps<SVGSVGElement>>;
  export const Bot: FC<SVGProps<SVGSVGElement>>;
  export const Brain: FC<SVGProps<SVGSVGElement>>;
  export const AlertCircle: FC<SVGProps<SVGSVGElement>>;
  export const AlertTriangle: FC<SVGProps<SVGSVGElement>>;
  export const MessageSquare: FC<SVGProps<SVGSVGElement>>;
  export const FolderOpen: FC<SVGProps<SVGSVGElement>>;
  export const Folder: FC<SVGProps<SVGSVGElement>>;
  export const File: FC<SVGProps<SVGSVGElement>>;
  export const FileCode: FC<SVGProps<SVGSVGElement>>;
  export const ChevronRight: FC<SVGProps<SVGSVGElement>>;
  export const RefreshCw: FC<SVGProps<SVGSVGElement>>;
  export const Plus: FC<SVGProps<SVGSVGElement>>;
  export const Minus: FC<SVGProps<SVGSVGElement>>;
  export const Terminal: FC<SVGProps<SVGSVGElement>>;
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
