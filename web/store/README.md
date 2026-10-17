# Demo Threads Co. mock storefront

A fictional brand with six simulated products (kurta, shirt, trousers, t-shirt, jacket, blouse). Everything here is invented and labelled **SIMULATED** on the pages.

- Each product page has two pictures, fabric, a size chart in a dialog, five or six reviews with zone comments ("runs small at the chest", "too short for my long torso", "shrank after the first wash"), the fit widget, and a bag.
- Charts mix kinds: the kurta is a body chart in inches; the blouse is a body chart in cm; the jacket is a garment chart with flat half-chest; the others are garment charts in cm.
- The bag (`cart.html`) allows two sizes of one item and says so, for the try-and-keep flow.
- The reader reads these pages: `python services/reader/cli.py --store --compare` prints extracted against true chart values.

## Edit and rebuild
Edit `build_store.py` (it holds the products), then run `python web/store/build_store.py`. A test fails if the committed HTML drifts from the generator. `truth.json` and `products.json` are generated too.

`python web/store/build_store.py --out DIR --api http://localhost:8000 --app http://localhost:5173` builds a copy for local work with the widget beside the pages.

## Local demo
```
python services/api/seed.py                       # merchant pk_demo_threads + simulated twin "demo-shopper"
cd services/api && AUTH_MODE=dev CONSENT_JWT_SECRET=dev uvicorn main:app --port 8000
cd web && VITE_AUTH_MODE=dev VITE_API_URL=http://localhost:8000 npm run dev            # app on :5173
python web/store/build_store.py --out .store-dev --api http://localhost:8000 --app http://localhost:5173
python -m http.server 8081 --directory .store-dev  # open http://localhost:8081/shirt.html
```
The seed script registers `http://localhost:8081` as an allowed origin. `npm run build` copies this folder and `web/widget` into `dist/store/` and `dist/widget/` for Firebase Hosting.
