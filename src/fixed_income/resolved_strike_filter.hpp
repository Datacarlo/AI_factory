// Stable selection of fixed-income contracts with positive resolved rate strikes.
#pragma once

#include "model/fixed_income/cir/parameters.hpp"
#include "model/fixed_income/g2/parameters.hpp"
#include "model/fixed_income/ornstein_uhlenbeck/parameters.hpp"
#include "model/fixed_income/vasicek/parameters.hpp"
#include "pricing_rows/result_index.cuh"

#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::fixed_income::strike_filter {

inline constexpr double kMinimumResolvedStrike = 1.0e-5;

// These host t=0 formulas select rows only. The pricing kernels resolve the
// absolute contract from the same source coordinates on the device.
template<class Model>
inline double model_bond(const Model& model, double t) {
    if constexpr (std::is_same_v<Model,
            model::fixed_income::g2::ModelParameters>) {
        const auto& p=model.process;
        const double a=p.mean_reversion_x,b=p.mean_reversion_y;
        const auto loading=[&](double rate) {
            return -std::expm1(-rate*t)/rate;
        };
        const auto variance=[&](double rate,double sigma) {
            const double z=loading(rate);
            return sigma*sigma/(rate*rate)
                *(t-2.0*z-std::expm1(-2.0*rate*t)/(2.0*rate));
        };
        const double cross=p.correlation*p.volatility_x*p.volatility_y
            *(t-loading(a)-loading(b)+loading(a+b))/(a*b);
        const double log_bond=-loading(a)*model.initial_state.state_x
            -loading(b)*model.initial_state.state_y
            +0.5*(variance(a,p.volatility_x)
                  +variance(b,p.volatility_y)+2.0*cross);
        return std::exp(log_bond);
    } else {
    const double a = model.process.mean_reversion;
    const double sigma = model.process.volatility;
    const double sigma2 = sigma * sigma;
    if constexpr (std::is_same_v<Model,
            model::fixed_income::cir::ModelParameters>) {
        const double gamma = std::sqrt(a*a + 2.0*sigma2);
        const double delta = 2.0*sigma2 / (gamma+a);
        const double one_minus = -std::expm1(-gamma*t);
        const double denominator = std::fma(-delta,one_minus,2.0*gamma);
        const double log_base = -std::log1p(-delta*one_minus/(2.0*gamma))
            - 0.5*delta*t;
        return std::exp(2.0*a*model.process.long_term_mean/sigma2*log_base
            - 2.0*one_minus/denominator*model.initial_state);
    } else {
        const double b = -std::expm1(-a*t)/a;
        const double variance = sigma2/(a*a)
            * (t - 2.0*b - std::expm1(-2.0*a*t)/(2.0*a));
        double exponent = -b*model.initial_state + 0.5*variance;
        if constexpr (std::is_same_v<Model,
                model::fixed_income::vasicek::ModelParameters>) {
            exponent -= model.process.long_term_mean*(t-b);
        }
        return std::exp(exponent);
    }
    }
}

template<class Curve>
inline double curve_bond(const Curve& curve, double t) {
    if constexpr (requires { curve.rate; }) {
        return std::exp(-curve.rate*t);
    } else {
        const auto loading = [](double x) {
            return std::abs(x)<1.0e-4
                ? 1.0-x/2.0+x*x/6.0-x*x*x/24.0
                : -std::expm1(-x)/x;
        };
        if constexpr (requires { curve.tau; }) {
            const double x=t/curve.tau;
            return std::exp(-t*(curve.beta0+curve.beta1*loading(x)
                +curve.beta2*(loading(x)-std::exp(-x))));
        } else {
            const double x1=t/curve.tau1, x2=t/curve.tau2;
            return std::exp(-t*(curve.beta0+curve.beta1*loading(x1)
                +curve.beta2*(loading(x1)-std::exp(-x1))
                +curve.beta3*(loading(x2)-std::exp(-x2))));
        }
    }
}

template<class Bonds, class Product>
inline double resolved_strike(const Bonds& bonds, const Product& product,
                              double day_fraction) {
    if (!product.quote_is_relative) return product.strike;
    if constexpr (requires { product.fixing_time_days; }) {
        const double fixing=product.fixing_time_days*day_fraction;
        const double payment=product.payment_time_days*day_fraction;
        const double accrual=product.accrual_period_days*day_fraction;
        return (bonds(fixing)/bonds(payment)-1.0)/accrual
            + product.strike;
    } else {
        const double start=[&] {
            if constexpr (requires { product.first_exercise_time_days; })
                return product.first_exercise_time_days*day_fraction;
            else return product.exercise_time_days*day_fraction;
        }();
        const double interval=product.payment_interval_days*day_fraction;
        double annuity=0.0,last_bond=0.0;
        for (std::uint32_t i=1;i<=product.payment_count;++i) {
            last_bond=bonds(std::fma(static_cast<double>(i),interval,start));
            annuity=std::fma(static_cast<double>(product.accrual_fraction),
                             last_bond,annuity);
        }
        return (bonds(start)-last_bond)/annuity+product.strike;
    }
}

template<class Product, class Bonds>
inline bool accepted_contract(const Bonds& bonds, const Product& product,
                              double day_fraction, bool nonnegative_rates,
                              bool option_is_call) {
    if constexpr (requires { product.fixing_time_days; }
        || requires { product.payment_count; }) {
        const double strike=resolved_strike(bonds,product,day_fraction);
        return std::isfinite(strike) && strike>kMinimumResolvedStrike;
    } else if constexpr (requires {
        product.option_expiry_days; product.bond_maturity_days;
        product.strike; product.quote_is_relative;
    }) {
        double strike=product.strike;
        double barrier=0.0;
        if (product.quote_is_relative) {
            const double expiry=product.option_expiry_days*day_fraction;
            const double maturity=product.bond_maturity_days*day_fraction;
            const double p0t=bonds(expiry), p0u=bonds(maturity);
            strike *= p0u/p0t;
            if constexpr (requires { product.barrier; })
                barrier=product.barrier*p0u;
        } else if constexpr (requires { product.barrier; }) {
            barrier=product.barrier;
        }
        if (!std::isfinite(strike) || !(strike>0.0)) return false;
        if constexpr (requires { product.barrier; }) {
            if (!std::isfinite(barrier) || !(strike<barrier)) return false;
            // This product is a call. With nonnegative short rates its bond
            // cannot exceed face value at expiry.
            if (nonnegative_rates && !(strike<1.0)) return false;
        } else if (nonnegative_rates && option_is_call && !(strike<1.0)) {
            return false;
        }
        return true;
    } else {
        return true;
    }
}

template<class Model,class Product,class Time>
std::vector<std::uint32_t> accepted_rows(
    std::span<const Model> models,std::span<const Product> products,
    PriceConstruction construction,const Time& time,
    bool option_is_call=false, bool select_nontrivial_bond=false) {
    constexpr bool selected_product =
        (requires { Product{}.quote_is_relative; Product{}.strike; }
         && (requires { Product{}.fixing_time_days; }
             || requires { Product{}.payment_count; }
             || requires { Product{}.option_expiry_days;
                           Product{}.bond_maturity_days; }));
    if constexpr (!selected_product) return {};
    else {
        if constexpr (requires { Product{}.option_expiry_days;
                                 Product{}.bond_maturity_days; })
            if (!select_nontrivial_bond) return {};
        const auto total=price_row_count(models.size(),products.size(),construction);
        if (total>std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("Fixed-income candidate row index overflow");
        std::vector<std::uint32_t> selected;
        selected.reserve(total);
        const double day_fraction=[&] {
            if constexpr (requires { time.day_fraction; })
                return static_cast<double>(time.day_fraction);
            else return static_cast<double>(time.dt)
                *time.simulation_steps_per_day;
        }();
        constexpr bool nonnegative_rates=std::is_same_v<Model,
            model::fixed_income::cir::ModelParameters>;
        for (std::uint32_t row=0;row<total;++row) {
            const auto index=decode_model_product_result_index(
                row,products.size(),construction);
            const auto bonds=[&](double t) {
                return model_bond(models[index.model_index],t);
            };
            if (accepted_contract(bonds,products[index.product_index],
                                  day_fraction,nonnegative_rates,option_is_call))
                selected.push_back(row);
        }
        if (selected.empty()) throw std::invalid_argument(
            "No nontrivial fixed-income contract remains");
        return selected.size()==total ? std::vector<std::uint32_t>{}
                                      : selected;
    }
}

template<class Model,class Curve,class Product,class Time>
std::vector<std::uint32_t> accepted_rows(
    std::span<const Model> models,std::span<const Curve> curves,
    std::span<const Product> products,PriceConstruction construction,
    const Time& time,bool option_is_call=false, bool select_nontrivial_bond=false) {
    constexpr bool selected_product =
        (requires { Product{}.quote_is_relative; Product{}.strike; }
         && (requires { Product{}.fixing_time_days; }
             || requires { Product{}.payment_count; }
             || requires { Product{}.option_expiry_days;
                           Product{}.bond_maturity_days; }));
    if constexpr (!selected_product) return {};
    else {
        if constexpr (requires { Product{}.option_expiry_days;
                                 Product{}.bond_maturity_days; })
            if (!select_nontrivial_bond) return {};
        const auto total=price_row_count(
            models.size(),curves.size(),products.size(),construction);
        if (total>std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("Fixed-income candidate row index overflow");
        std::vector<std::uint32_t> selected;
        selected.reserve(total);
        const double day_fraction=[&] {
            if constexpr (requires { time.day_fraction; })
                return static_cast<double>(time.day_fraction);
            else return static_cast<double>(time.dt)
                *time.simulation_steps_per_day;
        }();
        for (std::uint32_t row=0;row<total;++row) {
            const auto index=decode_model_curve_product_result_index(
                row,curves.size(),products.size(),construction);
            const auto bonds=[&](double t) {
                return curve_bond(curves[index.curve_index],t);
            };
            if (accepted_contract(bonds,products[index.product_index],
                                  day_fraction,false,option_is_call))
                selected.push_back(row);
        }
        if (selected.empty()) throw std::invalid_argument(
            "No nontrivial fixed-income contract remains");
        return selected.size()==total ? std::vector<std::uint32_t>{}
                                      : selected;
    }
}

} // namespace ai_factory::workbench::fixed_income::strike_filter
