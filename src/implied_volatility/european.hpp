// Annualized Black-Scholes volatility from a European option price.
#pragma once

#include <span>

namespace ai_factory::workbench::implied_volatility {

struct Quote {
    float spot, strike, years, rate, dividend_yield, price;
};

void invert_batch(std::span<const Quote> quotes, bool call, std::span<float> results);
}
