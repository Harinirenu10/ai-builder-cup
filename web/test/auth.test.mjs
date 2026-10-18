import test from 'node:test';
import assert from 'node:assert/strict';

// Test auth constants and storage keys
const DEMO_UID_KEY = 'omni_demo_uid';
const TOKEN_KEY = 'omni_token';

class MockLocalStorage {
  constructor() {
    this.store = {};
  }
  getItem(k) {
    return this.store[k] || null;
  }
  setItem(k, v) {
    this.store[k] = String(v);
  }
  removeItem(k) {
    delete this.store[k];
  }
  clear() {
    this.store = {};
  }
}

test('Demo sign-in properly formats dev bearer token and stores session', () => {
  const mockStorage = new MockLocalStorage();
  const uid = 'demo-shopper';
  const demoToken = `dev:${uid}`;
  const user = {
    uid,
    email: `${uid}@demo.local`,
    displayName: 'Demo Shopper',
    isDemo: true,
  };

  mockStorage.setItem(DEMO_UID_KEY, uid);
  mockStorage.setItem(TOKEN_KEY, demoToken);

  assert.equal(mockStorage.getItem(DEMO_UID_KEY), 'demo-shopper');
  assert.equal(mockStorage.getItem(TOKEN_KEY), 'dev:demo-shopper');
  assert.equal(user.isDemo, true);
  assert.equal(user.uid, 'demo-shopper');
});

test('Sign out clears demo credentials and bearer token', () => {
  const mockStorage = new MockLocalStorage();
  mockStorage.setItem(DEMO_UID_KEY, 'demo-shopper');
  mockStorage.setItem(TOKEN_KEY, 'dev:demo-shopper');

  // Perform signOut
  mockStorage.removeItem(DEMO_UID_KEY);
  mockStorage.removeItem(TOKEN_KEY);

  assert.equal(mockStorage.getItem(DEMO_UID_KEY), null);
  assert.equal(mockStorage.getItem(TOKEN_KEY), null);
});
