export interface FileEntry {
  name: string;
  path: string;
  type: "file" | "dir" | "unknown";
  size: number | null;
  modified: number | null;
  children?: FileEntry[];
  loaded?: boolean;
  loading?: boolean;
}

export interface OpenTab {
  path: string;
  name: string;
  content?: string;
  dirty?: boolean;
  language?: string;
}
