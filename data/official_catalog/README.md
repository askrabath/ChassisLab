# Legal inventory and downloaded CAD

Snapshot: September 5, 2026. `legal_rows.csv` is the downloaded R16-linked official spreadsheet: 425 rows, 418 unique SKUs. `legal_parts.csv` is the separate Home tab, not the part rows. `catalog.json` is the AI-readable normalized inventory; `index.html` is searchable.

Legality authority: https://link.vex.com/v5rc-legal-parts and https://www.vexrobotics.com/override-manual. Listing a part does not establish that an entire robot, modification, motor allocation or mechanism is legal. This snapshot does not automatically track rule changes.

`community_parts.zip` preserves all 1,198 F3D files from https://github.com/VEX-CAD/VEX-CAD-Fusion-360-Library. `fusion_catalog.json` records archive paths, per-file hashes and source tree commit. `cad_gallery.html` displays 1,193 embedded CAD thumbnails. These include washers, collars, bearings, shafts, fasteners, motion, structure and electronics. Five files lack embedded thumbnails. This is community geometry, not a source of legal rulings. File names are not verified SKU identities.

The community files require Fusion-compatible STEP/mesh export before MuJoCo can consume their geometry. They are downloaded but are NOT a complete simulator-ready, legally cross-referenced parts library. Only 16 official-list SKUs currently cross-reference exact labels in the supplied Clawbot mesh cache; source configurations may be cut lengths. Mating coordinates, materials, measured masses and part compatibility are not verified.

`Override_library.zip` is a separate community field CAD archive, not the robot parts archive. Direct VEX store CAD downloads and the public forum STEP link returned HTTP 403 during retrieval; do not mistake `pages/structure.html` for downloaded CAD.

Rebuild indexes from the saved downloads:

```sh
python3 -m tools.official_catalog
python3 -m tools.index_fusion_library
```

Do not infer that every official-list part has matching downloaded CAD. The official inventory and community CAD inventory intentionally remain separate until identity matching is verified.
