import { App } from './app';

try {
  new App();
} catch (err) {
  const div = document.createElement('div');
  div.style.cssText =
    'position:fixed;inset:0;display:flex;align-items:center;justify-content:center;color:#fff;background:#222;font:16px system-ui;';
  div.textContent = '初始化失败:' + (err instanceof Error ? err.message : String(err));
  document.body.appendChild(div);
}
