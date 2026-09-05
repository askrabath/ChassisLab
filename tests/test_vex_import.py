import json

from tools.vex_import import category, products


def test_step_product_inventory_handles_multiline_and_part_numbers(tmp_path):
    step=tmp_path/'sample.step'
    step.write_text("""ISO-10303-21;
#1=PRODUCT('V5 Smart Motor (276-4842)','',
'',(#2));
#3=PRODUCT('4\" Omni-Directional Wheel (276-2185)','','',(#4));
#5 = PRODUCT ( '276-9250-001', '', '', ( #6 ) ) ;
END-ISO-10303-21;
""")
    assert products(step)==['V5 Smart Motor (276-4842)','4" Omni-Directional Wheel (276-2185)','276-9250-001']
    assert category(products(step)[0])=='motors'
    assert category(products(step)[1])=='wheels'


def test_generated_clawbot_index_preserves_unknowns_as_null():
    index=json.load(open('data/vex_parts/index.json'))
    assert len(index['parts'])>=30
    assert any(p['part_number']=='276-4842' for p in index['parts'].values())
    assert all('mass_kg' in p for p in index['parts'].values())
