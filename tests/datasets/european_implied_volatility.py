"""Exercise the generator writer via the migration CLI, using analytical fixtures."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def normal(x):
    return math.erfc(-x / math.sqrt(2)) / 2


def check(binary, directory, side):
    years, spot, strike, rate, dividend, vol = 0.5, 2., 2.25, 0.03, 0.01, 0.37
    a, b = spot * math.exp(-dividend * years), strike * math.exp(-rate * years)
    v = vol * math.sqrt(years)
    d1 = math.log(a / b) / v + v / 2
    sign = 1 if side == 'call' else -1
    price = sign * (a * normal(sign * d1) - b * normal(sign * (d1-v)))
    header = dict(model_family='Heston', product_family='European Options', option_side=side, row_count=6,
                  time_convention=dict(unit='business_day', days_per_year=252),
                  quote_convention=dict(strike=dict(mode='ratio_to_central_spot')))
    rows = [dict(id=i, price=p, model_parameters=dict(spot=spot, risk_free_rate=rate,
            dividend_yield=dividend), product_parameters=dict(strike=strike/spot, maturity=t))
            for i, (p,t) in enumerate([(price,126),(-1.,126),(0.,0),(4.,126),(0.,126),(5.,126)])]
    # Exact boundaries using zero carry and an at-the-money strike.
    for row in rows[2:]:
        row['product_parameters']['strike'] = 1.
        row['model_parameters'].update(risk_free_rate=0., dividend_yield=0.)
    rows[3]['price'] = spot
    source, output = directory/'input', directory/'output'
    source.write_text(json.dumps(dict(**header, results=rows)))
    subprocess.run([binary, source, output], check=True)
    contents = output.read_bytes()
    subprocess.run([binary, output, source], check=True)
    assert contents == source.read_bytes()
    actual = json.loads(contents)['results']
    assert all(set(r) == set(original) | {'implied_volatility'} for r, original in zip(actual, rows))
    assert abs(actual[0]['implied_volatility'] - vol) < 5e-7
    assert all(actual[i]['implied_volatility'] == 0 for i in (1, 2, 4))
    high_iv = actual[3]['implied_volatility']
    assert 0 < high_iv == actual[5]['implied_volatility']
    price_at_cap = spot * math.erf(high_iv * math.sqrt(years / 8))
    assert abs(price_at_cap - (spot - 2**-23)) < 1e-11

    # Legacy enrichment should be safe to run again on the new schema.
    for row in actual:
        row['implied_volatility_status'] = 'ok'
    source.write_text(json.dumps(dict(**header, results=actual)))
    subprocess.run([binary, source, output], check=True)
    assert output.read_bytes() == contents

    header['model_family'] = 'Black-Scholes'
    source.write_text(json.dumps(dict(**header, results=actual)))
    subprocess.run([binary, source, output], check=True)
    assert json.loads(output.read_bytes())['results'] == rows


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as directory:
        for side in ('call','put'):
            check(sys.argv[1], Path(directory), side)
