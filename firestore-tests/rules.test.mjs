// Firestore security rules tests for firestore.rules. Covers:
//  - the twins/{twinId} ownership fix (owner_id -> user_id, matching
//    packages/fitcore/fitcore/schemas.py's TwinDocument and services/geometry/worker.py's write).
//  - the gifts/{giftId} ownership fix (sender_id/recipient_id -> giver_uid/recipient_uid, matching
//    services/api/apilib/gifts.py; writes are server-managed only, reads are giver-or-recipient).
//  - consents/{consentId}: real (fit_verdict) consents are owner-read, server-write-only
//    (services/api/apilib/fitquery.py, field `uid`); no client forging/tampering or cross-user read.
//  - fits/{fitId}: aspirational default (no app code touches this collection yet) -- owner-read,
//    server-write-only (write: false).
//
// Run with: firebase emulators:exec --project demo-rules-test --only firestore
//   "node --experimental-strip-types --test rules.test.mjs"
// (from this directory, after `npm install` here). Requires a Java runtime (the Firestore
// emulator needs one) and the firebase-tools CLI -- NOT verified to run in this session's
// sandboxed environment (no `java` on PATH was found when this was written). See the audit
// report for that limitation; this file is written to the same contract validate_security_config
// and _get_owned_twin already enforce at the API layer, so it documents and is ready to confirm
// the same guarantee at the rules layer once a Java-capable environment runs it.

import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {
  initializeTestEnvironment,
  assertSucceeds,
  assertFails,
} from '@firebase/rules-unit-testing';

let testEnv;

const TWIN = {
  id: 'twin_alice_1',
  user_id: 'alice',
  gender: 'neutral',
  height_cm: 170.0,
  betas: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  measurements: { zones: {} },
  created_at: '2026-01-01T00:00:00Z',
};

before(async () => {
  testEnv = await initializeTestEnvironment({
    projectId: 'demo-rules-test',
    firestore: {
      rules: fs.readFileSync('../firestore.rules', 'utf8'),
    },
  });
});

after(async () => {
  if (testEnv) await testEnv.cleanup();
});

test('the owner can read their own twin document', async () => {
  // Seed as an unauthenticated admin context, bypassing rules, as the real write path
  // (services/geometry/worker.py via the Admin SDK) does.
  await testEnv.withSecurityRulesDisabled(async (ctx) => {
    await ctx.firestore().collection('twins').doc(TWIN.id).set(TWIN);
  });

  const alice = testEnv.authenticatedContext('alice').firestore();
  await assertSucceeds(alice.collection('twins').doc(TWIN.id).get());
});

test('a different signed-in user is denied read access to someone else\'s twin', async () => {
  await testEnv.withSecurityRulesDisabled(async (ctx) => {
    await ctx.firestore().collection('twins').doc(TWIN.id).set(TWIN);
  });

  const bob = testEnv.authenticatedContext('bob').firestore();
  await assertFails(bob.collection('twins').doc(TWIN.id).get());
});

test('an unauthenticated request is denied read access entirely', async () => {
  await testEnv.withSecurityRulesDisabled(async (ctx) => {
    await ctx.firestore().collection('twins').doc(TWIN.id).set(TWIN);
  });

  const anon = testEnv.unauthenticatedContext().firestore();
  await assertFails(anon.collection('twins').doc(TWIN.id).get());
});

test('a user can create a twin document naming themself as user_id', async () => {
  const alice = testEnv.authenticatedContext('alice').firestore();
  await assertSucceeds(
    alice.collection('twins').doc('twin_alice_2').set({ ...TWIN, id: 'twin_alice_2' })
  );
});

test('a user cannot create a twin document naming someone else as user_id', async () => {
  const alice = testEnv.authenticatedContext('alice').firestore();
  await assertFails(
    alice.collection('twins').doc('twin_spoofed').set({ ...TWIN, id: 'twin_spoofed', user_id: 'bob' })
  );
});

test('the owner can update their own twin document', async () => {
  await testEnv.withSecurityRulesDisabled(async (ctx) => {
    await ctx.firestore().collection('twins').doc(TWIN.id).set(TWIN);
  });

  const alice = testEnv.authenticatedContext('alice').firestore();
  await assertSucceeds(
    alice.collection('twins').doc(TWIN.id).update({ height_cm: 171.0 })
  );
});

test('a different signed-in user is denied update/delete on someone else\'s twin', async () => {
  await testEnv.withSecurityRulesDisabled(async (ctx) => {
    await ctx.firestore().collection('twins').doc(TWIN.id).set(TWIN);
  });

  const bob = testEnv.authenticatedContext('bob').firestore();
  await assertFails(bob.collection('twins').doc(TWIN.id).update({ height_cm: 999.0 }));
  await assertFails(bob.collection('twins').doc(TWIN.id).delete());
});

// --------------------------------------------------------------------------- gifts
// Real fields are giver_uid / recipient_uid (services/api/apilib/gifts.py). All gift writes are
// server-mediated via the Admin SDK (which bypasses these rules); clients may only READ their own
// gift (as giver or assigned recipient), never write one directly.

// A pending gift from alice, not yet claimed (recipient_uid null), as create_gift writes it.
const GIFT_PENDING = {
  id: 'gift_1',
  giver_uid: 'alice',
  recipient_uid: null,
  status: 'pending',
  title: 'a gift',
  note: null,
  product: { category: 'shirt' },
  created_at: '2026-01-01T00:00:00Z',
  expires_at: '2026-01-15T00:00:00Z',
  recommended_size: null,
  confidence: null,
};

// An approved gift: alice gave it, bob is the assigned recipient (approve_gift set recipient_uid).
const GIFT_APPROVED = { ...GIFT_PENDING, id: 'gift_2', recipient_uid: 'bob', status: 'approved', recommended_size: 'M', confidence: 0.82 };

async function seedGifts() {
  await testEnv.withSecurityRulesDisabled(async (ctx) => {
    const db = ctx.firestore();
    await db.collection('gifts').doc(GIFT_PENDING.id).set(GIFT_PENDING);
    await db.collection('gifts').doc(GIFT_APPROVED.id).set(GIFT_APPROVED);
  });
}

test('the giver can read their own gift', async () => {
  await seedGifts();
  const alice = testEnv.authenticatedContext('alice').firestore();
  await assertSucceeds(alice.collection('gifts').doc(GIFT_PENDING.id).get());
});

test('the assigned recipient can read the gift they were assigned', async () => {
  await seedGifts();
  const bob = testEnv.authenticatedContext('bob').firestore();
  await assertSucceeds(bob.collection('gifts').doc(GIFT_APPROVED.id).get());
});

test('a third signed-in user (neither giver nor recipient) is denied read', async () => {
  await seedGifts();
  const mallory = testEnv.authenticatedContext('mallory').firestore();
  await assertFails(mallory.collection('gifts').doc(GIFT_PENDING.id).get());
  await assertFails(mallory.collection('gifts').doc(GIFT_APPROVED.id).get());
});

test('an unassigned prospective recipient cannot read a pending gift directly (server-mediated only)', async () => {
  // recipient_uid is null on a pending gift; nobody but the giver can read it client-side.
  await seedGifts();
  const bob = testEnv.authenticatedContext('bob').firestore();
  await assertFails(bob.collection('gifts').doc(GIFT_PENDING.id).get());
});

test('an unauthenticated request is denied read access to a gift', async () => {
  await seedGifts();
  const anon = testEnv.unauthenticatedContext().firestore();
  await assertFails(anon.collection('gifts').doc(GIFT_PENDING.id).get());
});

test('no client may create a gift directly, even naming themself as giver (server-managed only)', async () => {
  const alice = testEnv.authenticatedContext('alice').firestore();
  await assertFails(
    alice.collection('gifts').doc('gift_new_self').set({ ...GIFT_PENDING, id: 'gift_new_self', giver_uid: 'alice' })
  );
});

test('a client cannot create a gift spoofing another user as giver', async () => {
  const mallory = testEnv.authenticatedContext('mallory').firestore();
  await assertFails(
    mallory.collection('gifts').doc('gift_spoof').set({ ...GIFT_PENDING, id: 'gift_spoof', giver_uid: 'alice' })
  );
});

test('no client may update a gift directly -- not the giver, not the recipient, not a stranger', async () => {
  await seedGifts();
  const alice = testEnv.authenticatedContext('alice').firestore();       // giver
  const bob = testEnv.authenticatedContext('bob').firestore();            // assigned recipient
  const mallory = testEnv.authenticatedContext('mallory').firestore();   // stranger
  // Even the legitimate parties cannot write directly: the size/recipient assignment is computed
  // and set server-side (approve_gift), so every client update is denied.
  await assertFails(alice.collection('gifts').doc(GIFT_APPROVED.id).update({ recommended_size: 'XXL' }));
  await assertFails(bob.collection('gifts').doc(GIFT_APPROVED.id).update({ recommended_size: 'XXL' }));
  await assertFails(mallory.collection('gifts').doc(GIFT_APPROVED.id).update({ recommended_size: 'XXL' }));
});

test('no client may delete a gift directly -- not the giver, not the recipient, not a stranger', async () => {
  await seedGifts();
  const alice = testEnv.authenticatedContext('alice').firestore();
  const bob = testEnv.authenticatedContext('bob').firestore();
  const mallory = testEnv.authenticatedContext('mallory').firestore();
  await assertFails(alice.collection('gifts').doc(GIFT_APPROVED.id).delete());
  await assertFails(bob.collection('gifts').doc(GIFT_APPROVED.id).delete());
  await assertFails(mallory.collection('gifts').doc(GIFT_APPROVED.id).delete());
});

// --------------------------------------------------------------------------- consents
// Every REAL consent is a "store consent" written by services/api/apilib/fitquery.py::create_consent
// as {uid, merchant_id, scope: "fit_verdict", created_at, revoked_at} -- field is `uid` (not
// `user_id`), scope is always "fit_verdict". These are the authority-bearing records the
// merchant fit-query flow honors, and they are created/revoked ONLY server-side (Admin SDK). The
// rule's create/update/delete clauses are gated on `scope != 'fit_verdict'`, so a client can
// neither forge nor tamper with a real consent; it may only read its own. (The `user_id` /
// non-fit_verdict "capture consent" branch is an anticipated-but-unbuilt kind; its create path is
// tested for self-vs-spoof behavior to lock the rule down, not because the app writes one.)

const CONSENT_STORE = {
  id: 'consent_1',
  uid: 'alice',
  merchant_id: 'm_demo',
  scope: 'fit_verdict',
  created_at: '2026-01-01T00:00:00Z',
  revoked_at: null,
};

async function seedStoreConsent() {
  await testEnv.withSecurityRulesDisabled(async (ctx) => {
    await ctx.firestore().collection('consents').doc(CONSENT_STORE.id).set(CONSENT_STORE);
  });
}

test('the owner can read their own store consent', async () => {
  await seedStoreConsent();
  const alice = testEnv.authenticatedContext('alice').firestore();
  await assertSucceeds(alice.collection('consents').doc(CONSENT_STORE.id).get());
});

test('a different signed-in user cannot read someone else\'s consent', async () => {
  await seedStoreConsent();
  const bob = testEnv.authenticatedContext('bob').firestore();
  await assertFails(bob.collection('consents').doc(CONSENT_STORE.id).get());
});

test('an unauthenticated request cannot read a consent', async () => {
  await seedStoreConsent();
  const anon = testEnv.unauthenticatedContext().firestore();
  await assertFails(anon.collection('consents').doc(CONSENT_STORE.id).get());
});

test('a client cannot create a fit_verdict (store) consent directly -- server-managed only', async () => {
  // The authority-bearing kind: creation is denied for any client (scope == 'fit_verdict'),
  // even one naming itself as uid, so a merchant consent can never be forged client-side.
  const alice = testEnv.authenticatedContext('alice').firestore();
  await assertFails(
    alice.collection('consents').doc('consent_forged').set({ ...CONSENT_STORE, id: 'consent_forged', uid: 'alice' })
  );
});

test('a client cannot update or delete a store consent (e.g. cannot forge or clear revoked_at)', async () => {
  await seedStoreConsent();
  const alice = testEnv.authenticatedContext('alice').firestore();   // even the owner
  const bob = testEnv.authenticatedContext('bob').firestore();       // and a stranger
  await assertFails(alice.collection('consents').doc(CONSENT_STORE.id).update({ revoked_at: null }));
  await assertFails(alice.collection('consents').doc(CONSENT_STORE.id).delete());
  await assertFails(bob.collection('consents').doc(CONSENT_STORE.id).update({ revoked_at: '2026-02-01T00:00:00Z' }));
});

test('a client cannot create a consent spoofing another user as owner', async () => {
  // Non-fit_verdict "capture consent" create path requires user_id == auth.uid; naming another
  // user is denied, so there is no client-side ownership spoofing on create.
  const mallory = testEnv.authenticatedContext('mallory').firestore();
  await assertFails(
    mallory.collection('consents').doc('consent_spoof').set({ scope: 'capture', user_id: 'alice', created_at: '2026-01-01T00:00:00Z' })
  );
});

// --------------------------------------------------------------------------- fits
// No application code reads or writes a `fits` Firestore collection today (the only "fits" symbol
// is a FitsTwinVerdict enum value, not a collection). The rule is an aspirational default: writes
// are server-only (write: false) and a doc is readable only by the user named in its user_id.
// These tests lock in that default against the schema CLAUDE.md anticipates (user_id ownership).

const FIT = { id: 'fit_1', user_id: 'alice', recommended_size: 'M', confidence: 0.9 };

async function seedFit() {
  await testEnv.withSecurityRulesDisabled(async (ctx) => {
    await ctx.firestore().collection('fits').doc(FIT.id).set(FIT);
  });
}

test('the owner can read their own fit verdict; others and anon cannot', async () => {
  await seedFit();
  const alice = testEnv.authenticatedContext('alice').firestore();
  const bob = testEnv.authenticatedContext('bob').firestore();
  const anon = testEnv.unauthenticatedContext().firestore();
  await assertSucceeds(alice.collection('fits').doc(FIT.id).get());
  await assertFails(bob.collection('fits').doc(FIT.id).get());
  await assertFails(anon.collection('fits').doc(FIT.id).get());
});

test('no client may write a fit verdict -- server-generated only, even by the named owner', async () => {
  await seedFit();
  const alice = testEnv.authenticatedContext('alice').firestore();
  await assertFails(alice.collection('fits').doc('fit_new').set({ ...FIT, id: 'fit_new' }));
  await assertFails(alice.collection('fits').doc(FIT.id).update({ recommended_size: 'XXL' }));
  await assertFails(alice.collection('fits').doc(FIT.id).delete());
});
