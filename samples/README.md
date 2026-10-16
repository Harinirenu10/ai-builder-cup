# Sample Test Assets

This directory contains offline test assets and samples for the measured fit pipeline.

## Required Assets to Add

1. **Body Photos (3D Twin Reconstruction)**
   - **4 Orthogonal Photos**: Front, back, left profile, and right profile in fitted attire (e.g., activewear).
   - **Subject Metadata**: Confirmed subject height in centimeters (e.g., `height_cm: 178.5`), age confirmation (18+), and optional ground-truth manual tape measurements.
   - Example folder: `samples/body/subject_01/`

2. **Garment Photos (Garment Measurement)**
   - **Flat-lay Photo**: Front and back flat-lay photos of the garment placed flat on a contrasting surface.
   - **A4 Calibration Sheet**: An A4 reference sheet placed adjacent to the garment with visible corners for perspective correction and pixel-to-millimeter metric calibration.
   - **Manual Ground-truth Measurements**: Accompanying tape measurements (e.g., chest width, total length, sleeve length, waist width) recorded in centimeters for evaluation.
   - Example folder: `samples/garment/tee_white_01/`

3. **Saved Product Pages (Reader & Extractor)**
   - **HTML / MHTML Snapshots**: Saved product detail pages from ecommerce stores (e.g., ASOS, Zara, Uniqlo).
   - **Ground-truth Size Charts & Materials**: Manual extracts of sizing tables (alpha sizes to body/garment measurements) and fabric compositions for testing the Playwright/reader parser offline without live scraping.
   - Example folder: `samples/pages/product_01/`
