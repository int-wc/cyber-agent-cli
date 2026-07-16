/** Per-extension icon + color mapping (VS Code–inspired). */
import {
  File, FileCode, FileJson, FileText, FileImage, FileArchive,
  FileLock, FileTerminal, Globe, Folder, FolderGit, FolderOpen,
  Braces, Binary, Database, Cog, Shield, Bug, BarChart,
  type LucideIcon,
} from "lucide-react";

export interface FileIconMeta {
  icon: LucideIcon;
  color: string;   // CSS colour string
  label: string;   // tooltip category
}

const EXT_MAP: Record<string, FileIconMeta> = {
  // ── Web ──
  ts:  { icon: FileCode,     color: "#3b82f6", label: "TypeScript" },
  tsx: { icon: FileCode,     color: "#60a5fa", label: "TSX" },
  js:  { icon: FileCode,     color: "#eab308", label: "JavaScript" },
  jsx: { icon: FileCode,     color: "#facc15", label: "JSX" },
  html:{ icon: Globe,        color: "#f97316", label: "HTML" },
  htm: { icon: Globe,        color: "#f97316", label: "HTML" },
  css: { icon: Braces,       color: "#3b82f6", label: "CSS" },
  scss:{ icon: Braces,       color: "#ec4899", label: "SCSS" },
  less:{ icon: Braces,       color: "#3b82f6", label: "Less" },
  svg: { icon: FileImage,    color: "#f59e0b", label: "SVG" },

  // ── Data / Config ──
  json:{ icon: FileJson,     color: "#eab308", label: "JSON" },
  yaml:{ icon: FileCode,     color: "#ef4444", label: "YAML" },
  yml: { icon: FileCode,     color: "#ef4444", label: "YAML" },
  toml:{ icon: FileCode,     color: "#6b7280", label: "TOML" },
  xml: { icon: FileCode,     color: "#f59e0b", label: "XML" },
  env: { icon: FileLock,     color: "#22c55e", label: "Env" },
  lock:{ icon: FileLock,     color: "#6b7280", label: "Lock" },

  // ── Python ──
  py:  { icon: FileCode,     color: "#3b82f6", label: "Python" },
  pyc: { icon: Binary,       color: "#6b7280", label: "Python" },
  ipynb:{icon: FileCode,     color: "#f59e0b", label: "Jupyter" },

  // ── Shell / Script ──
  sh:  { icon: FileTerminal, color: "#22c55e", label: "Shell" },
  bash:{ icon: FileTerminal, color: "#22c55e", label: "Bash" },
  zsh: { icon: FileTerminal, color: "#22c55e", label: "Zsh" },
  fish:{ icon: FileTerminal, color: "#22c55e", label: "Fish" },
  ps1: { icon: FileTerminal, color: "#3b82f6", label: "PowerShell" },
  bat: { icon: FileTerminal, color: "#6b7280", label: "Batch" },

  // ── Compiled / Binary ──
  exe: { icon: Binary,       color: "#6b7280", label: "EXE" },
  dll: { icon: Binary,       color: "#6b7280", label: "DLL" },
  so:  { icon: Binary,       color: "#6b7280", label: "Shared Object" },
  o:   { icon: Binary,       color: "#6b7280", label: "Object" },
  class:{icon: Binary,       color: "#ef4444", label: "Class" },
  jar: { icon: FileArchive,  color: "#ef4444", label: "JAR" },
  bin: { icon: Binary,       color: "#6b7280", label: "Binary" },

  // ── Archives ──
  zip: { icon: FileArchive,  color: "#f59e0b", label: "ZIP" },
  tar: { icon: FileArchive,  color: "#f59e0b", label: "TAR" },
  gz:  { icon: FileArchive,  color: "#f59e0b", label: "Gzip" },
  rar: { icon: FileArchive,  color: "#f59e0b", label: "RAR" },
  "7z":{ icon: FileArchive,  color: "#f59e0b", label: "7z" },

  // ── Images ──
  png: { icon: FileImage,    color: "#a855f7", label: "PNG" },
  jpg: { icon: FileImage,    color: "#a855f7", label: "JPEG" },
  jpeg:{ icon: FileImage,    color: "#a855f7", label: "JPEG" },
  gif: { icon: FileImage,    color: "#a855f7", label: "GIF" },
  ico: { icon: FileImage,    color: "#a855f7", label: "ICO" },
  webp:{ icon: FileImage,    color: "#a855f7", label: "WebP" },

  // ── Docs ──
  md:  { icon: FileText,     color: "#3b82f6", label: "Markdown" },
  txt: { icon: FileText,     color: "#6b7280", label: "Text" },
  pdf: { icon: FileText,     color: "#ef4444", label: "PDF" },
  log: { icon: FileText,     color: "#6b7280", label: "Log" },

  // ── Security ──
  pcap:{ icon: Shield,       color: "#22c55e", label: "pcap" },
  cap: { icon: Shield,       color: "#22c55e", label: "cap" },
  pcapng:{icon: Shield,      color: "#22c55e", label: "pcapng" },
  har: { icon: BarChart,     color: "#f59e0b", label: "HAR" },

  // ── DB / Config ──
  db:  { icon: Database,     color: "#22c55e", label: "Database" },
  sqlite:{icon: Database,    color: "#22c55e", label: "SQLite" },
  sql: { icon: Database,     color: "#3b82f6", label: "SQL" },
  ini: { icon: Cog,          color: "#6b7280", label: "INI" },
  cfg: { icon: Cog,          color: "#6b7280", label: "Config" },
  conf:{ icon: Cog,          color: "#6b7280", label: "Config" },

  // ── C / C++ / Rust / Go ──
  c:   { icon: FileCode,     color: "#3b82f6", label: "C" },
  cpp: { icon: FileCode,     color: "#60a5fa", label: "C++" },
  cc:  { icon: FileCode,     color: "#60a5fa", label: "C++" },
  cxx: { icon: FileCode,     color: "#60a5fa", label: "C++" },
  h:   { icon: FileCode,     color: "#93c5fd", label: "Header" },
  hpp: { icon: FileCode,     color: "#93c5fd", label: "Header" },
  rs:  { icon: FileCode,     color: "#f59e0b", label: "Rust" },
  go:  { icon: FileCode,     color: "#06b6d4", label: "Go" },
  java:{ icon: FileCode,     color: "#ef4444", label: "Java" },
  kt:  { icon: FileCode,     color: "#a855f7", label: "Kotlin" },
  swift:{icon: FileCode,     color: "#f97316", label: "Swift" },
};

// ── Special file names ──

const NAME_MAP: Record<string, FileIconMeta> = {
  dockerfile:  { icon: FileLock,   color: "#3b82f6", label: "Dockerfile" },
  makefile:    { icon: Cog,        color: "#6b7280", label: "Makefile" },
  claude:      { icon: Bug,        color: "#a855f7", label: "Claude" },
  license:     { icon: FileText,   color: "#eab308", label: "License" },
  readme:      { icon: FileText,   color: "#3b82f6", label: "README" },
  changelog:   { icon: FileText,   color: "#22c55e", label: "Changelog" },
  ".gitignore":{ icon: Cog,        color: "#ef4444", label: "Gitignore" },
  ".env":      { icon: FileLock,   color: "#22c55e", label: "Env" },
};

// ── Folder special names ──

const FOLDER_MAP: Record<string, FileIconMeta> = {
  ".git":       { icon: FolderGit,    color: "#ef4444", label: "Git" },
  node_modules: { icon: Folder,       color: "#22c55e", label: "Node" },
  src:          { icon: Folder,       color: "#3b82f6", label: "Source" },
  dist:         { icon: Folder,       color: "#6b7280", label: "Dist" },
  build:        { icon: Folder,       color: "#6b7280", label: "Build" },
  public:       { icon: Folder,       color: "#f59e0b", label: "Public" },
  tests:        { icon: Bug,          color: "#22c55e", label: "Tests" },
  test:         { icon: Bug,          color: "#22c55e", label: "Tests" },
  __pycache__:  { icon: Folder,       color: "#6b7280", label: "Cache" },
  ".venv":      { icon: Folder,       color: "#22c55e", label: "venv" },
  venv:         { icon: Folder,       color: "#22c55e", label: "venv" },
  desktop:      { icon: Folder,       color: "#3b82f6", label: "Desktop" },
  components:   { icon: Folder,       color: "#a855f7", label: "Components" },
  hooks:        { icon: Folder,       color: "#f59e0b", label: "Hooks" },
  stores:       { icon: Folder,       color: "#eab308", label: "Stores" },
  styles:       { icon: Folder,       color: "#ec4899", label: "Styles" },
  services:     { icon: Folder,       color: "#06b6d4", label: "Services" },
  assets:       { icon: Folder,       color: "#f97316", label: "Assets" },
};

const DEFAULT_FILE: FileIconMeta = { icon: File, color: "#94a3b8", label: "File" };
const DEFAULT_FOLDER: FileIconMeta = { icon: Folder, color: "#eab308", label: "Folder" };

export function getFileIcon(name: string, isDir: boolean, open: boolean): FileIconMeta {
  const key = name.toLowerCase();
  if (isDir) {
    const m = FOLDER_MAP[key];
    if (m) return { ...m, icon: open ? FolderOpen : m.icon };
    return { ...DEFAULT_FOLDER, icon: open ? FolderOpen : DEFAULT_FOLDER.icon };
  }
  // 优先匹配特殊文件名。
  const nm = NAME_MAP[key];
  if (nm) return nm;
  // 再按扩展名匹配通用图标。
  const ext = key.includes(".") ? key.split(".").pop()! : "";
  return EXT_MAP[ext] || DEFAULT_FILE;
}
