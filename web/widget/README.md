# Measured Fit size widget

`fit-widget.js`: vanilla JavaScript, no framework, about 10 KB (a test fails above 15 KB). It adds a "Find my size" button to a product page and shows an answer like

> L fits you · high confidence · sleeves may need 2 cm shortening

## Use

```html
<script src="/widget/fit-widget.js" data-merchant="pk_your_key"
        data-api="https://your-site.example/api" data-app="https://measured-fit.example" defer></script>

<div data-fit-product='{"size_chart":{"sizes":{"S":{"chest":108},"M":{"chest":114}},"chart_type":"garment","unit":"cm"},
                        "category":"shirt","fit_type":"regular","fabric":[{"fiber":"cotton","percent":100}]}'></div>
```

| Attribute | Meaning |
|---|---|
| `data-merchant` | Your publishable key. It identifies you; it is not a secret. |
| `data-api` | Base URL of the API. Default `/api` (Firebase Hosting rewrites `/api/**` to Cloud Run). |
| `data-app` | Origin of the Measured Fit app that hosts the consent popup `/app/connect`. Default: the page's own origin. |
| `data-fit-product` | The garment: `size_chart` (cm, or `"unit":"in"`; `chart_type` body or garment), `category`, `fit_type`, `fabric`, optional `stretch_class`, `pre_shrunk`, `ref`. |

When the answer arrives the widget fires `measuredfit:result` (it bubbles; `detail` is the response). The demo store uses it to preselect the size.

## How it works
1. The first click opens `{data-app}/app/connect?merchant=...&origin=...` in a pop-up. The shopper signs in and approves.
2. The pop-up posts a signed consent token (30 days) to the opener, and only to the store origin the API accepted. The widget accepts messages only from `data-app`'s origin and only from its own pop-up.
3. The widget sends `POST {data-api}/v1/fit-query` with `X-Merchant-Key` and the token. The response holds a size, a confidence, zone verdict words, an optional garment hint and a guarantee flag. It has no field for measurements or shape.
4. The token is kept in `localStorage` (`mfw:consent:<key>`). If the shopper revokes it in the customer app, the next query returns 403 and the widget offers "Connect again".

Your store origin must be listed in the merchant's `allowed_origins`; otherwise the pop-up refuses to approve and the API answers 403.

## Production
**Call the fit query from your own backend in production**, not from the browser. A publishable key is visible to anyone, so the browser path relies on the origin allow-list, the shopper's consent token and the per-merchant rate limit alone. A backend call carries no `Origin` header, keeps your key out of page source, and lets you cache answers. The widget suits demos and low-risk pages.

## Browsers
Tested in Chromium and WebKit (Safari's engine) with Playwright (`services/api/tests/test_widget_e2e.py`). WebKit is not Safari itself: check once on a Mac and an iPhone. Pop-up blockers can stop the consent window; the widget says so and asks the shopper to allow pop-ups.
