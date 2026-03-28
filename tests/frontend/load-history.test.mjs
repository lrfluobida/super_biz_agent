import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

function loadAppClass() {
  const source = fs.readFileSync('static/app.js', 'utf8');
  const sandbox = {
    console,
    setTimeout,
    clearTimeout,
    localStorage: {
      getItem() {
        return null;
      },
      setItem() {},
      removeItem() {},
    },
    document: {
      createElement() {
        return {
          style: {},
          appendChild() {},
          classList: { add() {}, remove() {} },
          querySelector() {
            return null;
          },
          querySelectorAll() {
            return [];
          },
          set textContent(_) {},
          set innerHTML(_) {},
        };
      },
      head: { appendChild() {} },
      addEventListener() {},
      body: { style: {} },
      querySelector() {
        return null;
      },
      getElementById() {
        return null;
      },
    },
    window: {},
    fetch: async () => ({ ok: true, json: async () => ({ history: [] }) }),
  };

  vm.createContext(sandbox);
  vm.runInContext(`${source}\nglobalThis.__SuperBizAgentApp = SuperBizAgentApp;`, sandbox);
  return {
    SuperBizAgentApp: sandbox.__SuperBizAgentApp,
    sandbox,
  };
}

(async () => {
  const { SuperBizAgentApp, sandbox } = loadAppClass();
  const backendHistory = [
    { role: 'user', content: '历史问题1', timestamp: '2026-03-28T10:00:00' },
    { role: 'assistant', content: '历史回答1', timestamp: '2026-03-28T10:00:01' },
  ];

  sandbox.fetch = async () => ({
    ok: true,
    async json() {
      return { history: backendHistory };
    },
  });

  const app = {
    chatHistories: [{ id: 'session-1', messages: [] }],
    currentChatHistory: [],
    sessionId: 'other-session',
    isCurrentChatFromHistory: false,
    chatMessages: { innerHTML: '' },
    checkAndSetCentered() {},
    renderChatHistory() {},
    saveCurrentChat() {},
    updateCurrentChatHistory() {},
    addMessage(type, content, isStreaming = false, saveToHistory = true) {
      if (!isStreaming && saveToHistory && content) {
        this.currentChatHistory.push({
          type,
          content,
          timestamp: new Date().toISOString(),
        });
      }
    },
  };

  await SuperBizAgentApp.prototype.loadChatHistory.call(app, 'session-1');

  assert.deepEqual(
    app.currentChatHistory.map((message) => ({
      type: message.type,
      content: message.content,
    })),
    [
      { type: 'user', content: '历史问题1' },
      { type: 'assistant', content: '历史回答1' },
    ],
  );
  console.log('frontend load-history test passed');
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
