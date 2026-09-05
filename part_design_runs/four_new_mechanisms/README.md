# Four new scoring mechanism layouts

Open `gallery.html` for full and reduced-detail CAD previews. `four_mechanisms.png` compares all four. These use the same source chassis and four different mechanisms: parallel four-bar fork, inclined roller feeder, tilting channel tray, and elevated cantilever claw.

Actual Responses API generation used `gpt-5.6-sol`, reasoning `low`, two bounded calls with a 5,000-output-token limit each. The first proposal was rejected for unknown SKU 276-1272. The second passed the official-list SKU and distinct-mechanism checks. Usage: 28,467 total tokens. `ai_designs.json` preserves the accepted proposal; `api_record_*.json` preserve prompts, response IDs, status and usage. The first rejected full proposal was not retained, but its failure is recorded. The accepted full proposal is preserved.

AI authorship: names, rationale, mechanism-family selections, static angle/spacing parameters, hypotheses, scoring sequence and requested part list. Hand-authored code supplies the four layout templates and retained source chassis. These are constrained AI concepts, not autonomous connection-solving CAD or four independently invented architectures.

All rendered robot geometry is unscaled source CAD; simple geometry appears only as the floor. High/low images keep the same instances including small hardware. The source meshes remain available separately from display LODs. The source claw retains its internal washer and standoff placements.

**Not build-ready or simulation-ready.** No physics steps or performance tests were run. These static MJCF files contain visual geometry with contacts disabled. They must not be used as physics models. No solved mates, complete fastener stacks, continuous shafts, gear engagement, chains, structural bracing or collision-clearance validation is supplied. Legal SKU checks are not a buildability or competition legality certification.

`rendered_bom.json` describes what is visible; `required_parts.json` describes the AI's requested parts, including missing CAD. They are not the same list. In particular the feeder layout uses available source omni-wheel CAD while the proposal requests flex-wheel hardware; this is a layout stand-in, not a physically equivalent assembly. The claw kit identity is also unverified. No scoring performance can be inferred from these previews.

Re-render the saved proposals without any new API calls or simulations, from the project root:

```sh
/private/tmp/chassis-lab-venv/bin/python -m chassis_lab.catalog_concepts --render-only
```

Generate a fresh bounded set (uses the previously authorized `.env.local` API credential; choose a new output directory to preserve this run):

```sh
/private/tmp/chassis-lab-venv/bin/python -m chassis_lab.catalog_concepts --output part_design_runs/next_mechanisms
```

To reach buildable robot generation, next work is CAD format conversion, exact SKU/configuration matching, hole/shaft/gear mate definitions, transmission routing and structural/clearance checks. No benchmark changes were made for these concepts.
