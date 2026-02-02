#!/usr/bin/env node
/**
 * Extracts and simplifies Natural Earth 110m land polygons for the globe.
 * GeoJSON uses [lng, lat]; we output [lat, lng] for our projection.
 * Run: node scripts/extract-earth-land.js
 */

const https = require('https');
const fs = require('fs');

const URL = 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_land.geojson';

function fetch(url) {
  return new Promise((resolve, reject) => {
    https.get(url, (res) => {
      let data = '';
      res.on('data', (ch) => (data += ch));
      res.on('end', () => resolve(JSON.parse(data)));
    }).on('error', reject);
  });
}

function simplifyRing(ring, step = 2) {
  const simplified = [];
  for (let i = 0; i < ring.length; i += step) {
    const [lng, lat] = ring[i];
    simplified.push([lat, lng]);
  }
  if (ring.length > 0 && simplified[simplified.length - 1].join(',') !== simplified[0].join(',')) {
    simplified.push(simplified[0]);
  }
  return simplified;
}

async function main() {
  const geojson = await fetch(URL);
  const polygons = [];

  for (const feature of geojson.features) {
    const geom = feature.geometry;
    if (!geom || geom.type !== 'Polygon') continue;

    const rings = geom.coordinates;
    for (const ring of rings) {
      if (ring.length < 4) continue;
      const bbox = feature.properties?.bbox;
      const minZoom = feature.properties?.min_zoom ?? 1;

      let step = 3;
      if (ring.length > 500) step = 8;
      else if (ring.length > 200) step = 5;
      else if (ring.length > 80) step = 4;

      const simplified = simplifyRing(ring, step);
      if (simplified.length >= 3) {
        polygons.push(simplified);
      }
    }
  }

  const output = `// Auto-generated from Natural Earth 110m land - run: node scripts/extract-earth-land.js
// Format: [lat, lng] per point, GeoJSON [lng, lat] converted

export const EARTH_LAND_POLYGONS: [number, number][][] = ${JSON.stringify(polygons, null, 2)};
`;

  const outPath = 'frontend/lib/earthLand.ts';
  fs.writeFileSync(outPath, output);
  console.log(`Wrote ${polygons.length} polygons to ${outPath}`);
}

main().catch(console.error);
