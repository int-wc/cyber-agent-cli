export interface GitStatus {
  branch: string;
  staged: { path: string; status: string }[];
  unstaged: { path: string; status: string }[];
  untracked: string[];
}

export interface GitDiff {
  diff: string;
  error?: string;
}

export interface GitCommit {
  hash: string;
  message: string;
  author: string;
  date: string;
}
