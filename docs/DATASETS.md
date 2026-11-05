# Agricultural dataset acquisition plan (India-first)

The original resume says “agricultural field prediction” but does not identify the original dataset, target, label-generation process, geography or measurement units. Do not claim that district crop yield is the same as plot-level field performance. The sources below support a rigorous new research track.

## 1. Ground truth: District Area, Production and Yield (APY)

- Official Directorate of Economics & Statistics portal: https://data.desagri.gov.in/website/crops-apy-report-web
- APY records contain state/district/crop/season/year observations for area, production and yield. Check the exact downloadable file, revision date, crop codes, district boundaries and units before modelling.
- A discoverability mirror with documented district/year/crop/season coverage is available at https://ckandev.indiadataportal.com/dataset/area-production-yield-apy/resource/f980409d-49a2-42ae-9eb0-182365005c04. Treat the official DES file as canonical and verify the mirror against it.

This is district-level administrative output—not an individual farm, plot or sensor reading. Label the task “district-level crop yield estimate” when using APY.

## 2. Historical panel: ICRISAT VDSA

- Documentation: https://vdsa.icrisat.org/Include/document/all-apportioned-web-document.pdf
- Historical district database covering selected Indian states and years, with crop area/production/yield and additional agriculture variables.
- Districts are harmonized to historical boundaries in the documented series. Do not silently merge those polygons with current district boundaries.

Use as a complementary historical robustness source after checking availability, terms and district crosswalks. It is not a substitute for current APY records.

## 3. Weather covariates: NASA POWER

- Daily API documentation: https://power.larc.nasa.gov/docs/services/api/temporal/daily/
- Provides daily solar and meteorological time series at gridded locations. Query district centroids or a documented set of points; store coordinates, grid resolution, UTC/local-day convention and retrieval date.
- Gridded reanalysis is not equivalent to a farm weather station. Cache requests and document spatial/temporal aggregation.

## 4. Soil covariates: ISRIC SoilGrids

- Documentation: https://docs.isric.org/globaldatabases/soilgrids/
- 250 m gridded soil properties include pH, organic carbon, bulk density, texture and uncertainty layers.
- ISRIC documentation currently notes a pause in its REST API; use documented WCS/WebDAV/VRT or Google Earth Engine access where available. Do not build a hard dependency on an unavailable endpoint.

## Recommended training design

1. Verify APY units and crop/state/district identifiers. Retain source file, retrieval date and SHA-256 digest.
2. Define the forecast origin. Use only predictors known before that date.
3. Establish a previous-year same-crop/district/season baseline.
4. Train on earlier years and test on later years. Add a separate district-held-out test.
5. Exclude same-year production/yield fields and other post-outcome variables from predictors.
6. Report MAE/RMSE in original target units, R², and error by crop/state/season/year. Compare to the baseline.
7. Keep APY administrative output separate from any genuine plot-level dataset.

A genuine plot-level project needs georeferenced plot IDs, crop/variety, planting/harvest dates, measured yield, soil measurements, management events and sensor/weather observations—with a documented sampling protocol.
