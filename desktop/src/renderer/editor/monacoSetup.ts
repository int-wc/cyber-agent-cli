type MonacoWorkerModule = {
  default: new () => Worker;
};

function loadWorker(loader: () => Promise<MonacoWorkerModule>): Promise<Worker> {
  return loader().then(({ default: WorkerCtor }) => new WorkerCtor());
}

self.MonacoEnvironment = {
  getWorker(_workerId: string, label: string): Promise<Worker> {
    switch (label) {
      case "typescript":
      case "javascript":
        return loadWorker(() =>
          import("monaco-editor/esm/vs/language/typescript/ts.worker.js?worker")
        );
      case "json":
        return loadWorker(() =>
          import("monaco-editor/esm/vs/language/json/json.worker.js?worker")
        );
      case "css":
      case "scss":
      case "less":
        return loadWorker(() =>
          import("monaco-editor/esm/vs/language/css/css.worker.js?worker")
        );
      case "html":
      case "handlebars":
      case "razor":
        return loadWorker(() =>
          import("monaco-editor/esm/vs/language/html/html.worker.js?worker")
        );
      default:
        return loadWorker(() =>
          import("monaco-editor/esm/vs/editor/editor.worker.js?worker")
        );
    }
  },
};
