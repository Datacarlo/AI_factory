// Offline runner for correlated equity baskets under one short-rate policy.
#pragma once

#include "multi_asset/equity_rate_price_kernel.cuh"
#include "tools/pricing/heterogeneous_rough_price_generation.cuh"
#include "tools/cuda/generation_checkpoint.hpp"
#include "tools/cuda/generation_progress.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ai_factory::workbench::offline::pricing {

struct CrossAssetBatchGeometry {
    std::size_t results_per_launch=4096U;
    unsigned blocks_per_result=0U; // Zero selects from SM count and path work.
    unsigned threads_per_block=128U;
    std::size_t rough_rows_per_launch=1U;
    unsigned streams=0U; // Zero retains memory-adaptive historical stream selection.
    unsigned paths_per_team=0U; // Zero retains TeamSize paths per tile.
};

template<typename EquityPack,typename RoughPack,typename RatePolicy,
         typename Product,unsigned TeamSize=1U>
cuda::MonteCarloRun execute_cross_asset_prices(
    const MultiAssetHostModels<EquityPack>& equities,
    const MultiRoughHostModels<RoughPack>& rough,
    const std::vector<typename RatePolicy::Parameters>& rate_models,
    const std::vector<typename Product::ProductParameters>& products,
    const std::vector<correlation::MatrixRow>& correlations,
    const std::vector<std::string>& external_factor_order,
    const std::vector<std::string>& internal_factor_order,
    const CompiledRowIndexPlan& row_plan,
    std::size_t paths_per_price,std::size_t path_chunk_size,
    float dt,std::uint32_t steps_per_day,std::uint32_t direct_tile_size,
    std::uint64_t base_seed,
    const std::vector<typename RatePolicy::CurveParameters>& curves={},
    CrossAssetBatchGeometry geometry={}
) {
    static_assert(EquityPack::kAssetCount+RoughPack::kAssetCount>0U);
    static_assert(Product::kTerminalCashflow);
    constexpr std::size_t rough_count=RoughPack::kAssetCount;
    constexpr std::size_t dimension=rough_count
        +EquityPack::kFactorCount+RatePolicy::kBrownianDimension
        +RoughPack::kTwoFactorCount;
    constexpr std::size_t lower_count=dimension*(dimension+1U)/2U;
    const std::size_t result_count=row_plan.result_count;
    if(row_plan.selectors.size()!=EquityPack::kAssetCount
            +RoughPack::kAssetCount+3U
            +(RatePolicy::kUsesCurve ? 1U : 0U)
        || products.empty() || correlations.empty()
        || result_count==0U || paths_per_price<2U
        || !(dt>0.0f) || steps_per_day==0U
        || external_factor_order.size()!=dimension
        || internal_factor_order.size()!=dimension
        || rate_models.empty()
        || (RatePolicy::kUsesCurve && curves.empty())
        || geometry.results_per_launch==0U
        || geometry.threads_per_block<32U
        || geometry.threads_per_block>1024U
        || geometry.threads_per_block%32U!=0U
        || geometry.threads_per_block%TeamSize!=0U)
        throw std::invalid_argument("invalid cross-asset inputs");
    if constexpr (rough_count>0U) {
        if(paths_per_price<128U || paths_per_price%128U!=0U
            || path_chunk_size==0U
            || path_chunk_size%128U!=0U
            || (direct_tile_size!=32U && direct_tile_size!=64U))
            throw std::invalid_argument("invalid cross-asset FFT geometry");
        if(geometry.blocks_per_result!=0U
            || (geometry.threads_per_block!=128U && geometry.threads_per_block!=256U))
            throw std::invalid_argument(
                "rough cross-asset batch geometry is not supported");
    }
    equities.validate(row_plan.selectors,0U);
    rough.validate(row_plan.selectors,
        EquityPack::kAssetCount);
    const auto rate_slot=EquityPack::kAssetCount+rough_count;
    if(row_plan.selectors[rate_slot].source_count!=rate_models.size()
        || row_plan.selectors[rate_slot+1U].source_count!=products.size()
        || row_plan.selectors[rate_slot+2U].source_count!=correlations.size()
        || (RatePolicy::kUsesCurve
            && row_plan.selectors[rate_slot+3U].source_count
                !=curves.size()))
        throw std::invalid_argument(
            "compact cross-asset selector disagrees with dataset");
    for(const auto& rate:rate_models)
        if(!RatePolicy::valid_parameters(rate))
            throw std::invalid_argument("invalid short-rate parameters");
    if constexpr (RatePolicy::kUsesCurve)
        for(const auto& curve:curves)
            if(!RatePolicy::valid_curve(curve))
                throw std::invalid_argument("invalid initial curve");
    std::vector<std::string> variance_names,spot_names;
    if constexpr (rough_count>0U)
        RoughPack::append_factor_names(external_factor_order,
            EquityPack::kFactorCount,variance_names,spot_names);
    std::vector<std::string> required=variance_names;
    required.insert(required.end(),external_factor_order.begin(),
        external_factor_order.begin()+EquityPack::kFactorCount);
    required.insert(required.end(),external_factor_order.end()
        -RatePolicy::kBrownianDimension,external_factor_order.end());
    required.insert(required.end(),spot_names.begin(),spot_names.end());
    if(internal_factor_order!=required)
        throw std::invalid_argument("cross-asset factor order mismatch");
    std::vector<correlation::MatrixRow> ordered;
    std::vector<float> factors;
    ordered.reserve(correlations.size());
    factors.reserve(correlations.size()*lower_count);
    for(const auto& matrix:correlations) {
        if(matrix.factors!=external_factor_order
            || matrix.lower.size()!=lower_count)
            throw std::invalid_argument("cross-asset external factors mismatch");
        ordered.push_back(correlation::reorder_factors(
            matrix,internal_factor_order));
        factors.insert(factors.end(),ordered.back().lower.begin(),
            ordered.back().lower.end());
    }
    std::uint32_t max_steps=0U;
    for(const auto& product:products) {
        if constexpr (requires { Product::valid_parameters(
            product,steps_per_day); })
            if(!Product::valid_parameters(product,steps_per_day))
                throw std::invalid_argument("invalid cross-asset product");
        const auto count=static_cast<std::uint64_t>(
            product.maturity_days)*steps_per_day;
        const auto limit=rough_count==0U
            ? philox::kMaximumDomainStep
            : (RoughPack::kAllCausal ? 8192U : 4096U);
        if(count==0U || count>limit)
            throw std::invalid_argument("cross-asset maturity exceeds capacity");
        max_steps=std::max(max_steps,static_cast<std::uint32_t>(count));
    }
    const auto chunk=rough_count>0U
        ? std::min(paths_per_price,path_chunk_size) : paths_per_price;
    const auto raw_workspace_bytes=MultiRoughDeviceModels<RoughPack>::workspace_bytes(max_steps,chunk);
    const auto workspace_bytes = volterra::checked_hybrid_fft_sum(
        raw_workspace_bytes, 15U, "rough workspace alignment overflow") / 16U * 16U;
    const auto volatility_values = volterra::checked_hybrid_fft_product(
        volterra::checked_hybrid_fft_product(rough_count, chunk,
            "rough path capacity overflow"),
        max_steps, "rough path capacity overflow");
    const unsigned path_threads=geometry.threads_per_block;
    const unsigned paths_per_team=geometry.paths_per_team ? geometry.paths_per_team : TeamSize;
    const auto launch_rows=rough_count>0U ? geometry.rough_rows_per_launch : 1U;
    if (!launch_rows || launch_rows>65535U || geometry.streams>64U || paths_per_team>1024U)
        throw std::invalid_argument("invalid cross-asset rough row geometry");
    const std::size_t paths_per_block=(path_threads/TeamSize)*static_cast<std::size_t>(paths_per_team);
    const auto chunk_blocks=(chunk+paths_per_block-1U)/paths_per_block;
    const auto max_partials=(paths_per_price/chunk)*chunk_blocks
        +((paths_per_price%chunk)+paths_per_block-1U)/paths_per_block;
    const auto batch_rows=std::min<std::size_t>(
        geometry.results_per_launch,65535U);
    const auto bytes_per_row=volterra::checked_hybrid_fft_sum(workspace_bytes,
        volterra::checked_hybrid_fft_sum(volterra::checked_hybrid_fft_product(volatility_values,sizeof(float),"rough row memory overflow"),
            volterra::checked_hybrid_fft_product(max_partials,sizeof(reductions::MomentSums),"rough moments overflow"),"rough row memory overflow"),
        "rough row memory overflow");
    const auto rough_batch_size=rough_count>0U
        ? (geometry.streams ? geometry.streams : choose_rough_price_batch_size(result_count,
            std::min<std::size_t>(geometry.results_per_launch,4U),
            volterra::checked_hybrid_fft_product(launch_rows,bytes_per_row,"rough launch memory overflow"))) : 1U;
    const auto slot_count=rough_batch_size*launch_rows;
    std::size_t free_bytes=0U,total_bytes=0U;
    check_cuda(cudaMemGetInfo(&free_bytes,&total_bytes),"cross-asset workspace budget");
    if (volterra::checked_hybrid_fft_product(slot_count,bytes_per_row,"rough workspace overflow") > free_bytes*3U/4U)
        throw std::invalid_argument("cross-asset row/stream workspace exceeds available memory; reduce L, P or streams");
    cudaDeviceProp properties{};
    if constexpr (rough_count==0U) {
        int device=0;
        check_cuda(cudaGetDevice(&device),"cross-asset batch device");
        check_cuda(cudaGetDeviceProperties(&properties,device),
            "cross-asset batch properties");
    }
    const auto blocks_for_count=[&](std::size_t count)->unsigned {
        const auto available=static_cast<unsigned>(max_partials);
        if(geometry.blocks_per_result>0U)
            return std::min(available,geometry.blocks_per_result);
        if(count==1U) return available;
        const unsigned occupancy=static_cast<unsigned>(
            (2U*properties.multiProcessorCount+count-1U)/count);
        const unsigned tiles_per_block=dimension>=16U ? 2U : 4U;
        const unsigned work=static_cast<unsigned>(
            (max_partials+tiles_per_block-1U)/tiles_per_block);
        return std::min(available,std::max({1U,occupancy,work}));
    };
    std::size_t moment_capacity=rough_count>0U
        ? max_partials*slot_count : 1U;
    if constexpr (rough_count==0U)
        for(std::size_t offset=0U;offset<result_count;offset+=batch_rows) {
            const auto count=std::min(batch_rows,result_count-offset);
            const auto blocks=blocks_for_count(count);
            if(blocks>1U)
                moment_capacity=std::max(moment_capacity,count*blocks);
        }
    const auto wall_start=std::chrono::steady_clock::now();
    cuda::GenerationCheckpoint checkpoint(
        result_count,{"price","standard_error"});
    cuda::GenerationProgress progress(
        result_count,checkpoint.completed_prices());
    cuda::KernelDurationAccumulator batch_timer;
    const bool tracked_batches=checkpoint.enabled() || progress.enabled();
    MultiAssetDeviceModels<EquityPack> device_equities(equities);
    MultiRoughDeviceModels<RoughPack> device_rough(rough);
    cuda::DeviceBuffer<typename RatePolicy::Parameters>
        device_rates(rate_models.size());
    cuda::DeviceBuffer<RowIndexSelector> device_selectors(
        row_plan.selectors.size());
    cuda::DeviceBuffer<typename RatePolicy::CurveParameters> device_curves(
        RatePolicy::kUsesCurve ? curves.size() : 1U);
    cuda::DeviceBuffer<typename Product::ProductParameters>
        device_products(products.size());
    cuda::DeviceBuffer<float> device_factors(factors.size());
    cuda::DeviceBuffer<float> device_prices(result_count);
    cuda::DeviceBuffer<float> device_errors(result_count);
    cuda::DeviceBuffer<reductions::MomentSums> device_moments(
        moment_capacity);
    cuda::DeviceBuffer<float> device_volatility(
        rough_count>0U ? slot_count*volatility_values : 1U);
    cuda::DeviceBuffer<std::uint8_t> workspace(
        workspace_bytes>0U ? slot_count*workspace_bytes : 1U);
    check_cuda(cudaMemset(device_moments.data(),0,moment_capacity*sizeof(reductions::MomentSums)),"cross-asset moments initialization");
    device_rates.copy_from(rate_models.data());
    device_selectors.copy_from(row_plan.selectors.data());
    if constexpr (RatePolicy::kUsesCurve)
        device_curves.copy_from(curves.data());
    device_products.copy_from(products.data());
    device_factors.copy_from(factors.data());
    const monte_carlo::CrossAssetDeviceInputs<
        EquityPack,RoughPack,RatePolicy,Product> inputs{
            device_equities.inputs(device_selectors.data()),
            device_rough.inputs(device_selectors.data(),
                EquityPack::kAssetCount),
            device_rates.data(),
            RatePolicy::kUsesCurve ? device_curves.data() : nullptr,
            device_products.data(),device_factors.data(),
            device_selectors.data()};
    cuda::Event start,stop;
    check_cuda(cudaEventRecord(start.get()),"cross-asset timer start");
    std::vector<float> checkpoint_prices,checkpoint_errors;
    const auto commit_batch=[&](std::size_t offset,std::size_t count) {
        if(checkpoint.enabled()) {
            checkpoint_prices.resize(count);
            checkpoint_errors.resize(count);
            check_cuda(cudaMemcpy(
                checkpoint_prices.data(),device_prices.data()+offset,
                count*sizeof(float),cudaMemcpyDeviceToHost),
                "cross-asset checkpoint prices");
            check_cuda(cudaMemcpy(
                checkpoint_errors.data(),device_errors.data()+offset,
                count*sizeof(float),cudaMemcpyDeviceToHost),
                "cross-asset checkpoint standard errors");
            checkpoint.commit(offset,count,{
                std::span<const float>(checkpoint_prices),
                std::span<const float>(checkpoint_errors)});
        }
        progress.record_host_progress(offset+count);
    };
    if constexpr (rough_count==0U) {
        for(std::size_t offset=checkpoint.completed_prices();offset<result_count;
            offset+=batch_rows) {
            const auto count=std::min(batch_rows,result_count-offset);
            if(checkpoint.enabled()) batch_timer.start_batch();
            const unsigned blocks=blocks_for_count(count);
            monte_carlo::launch_cross_asset_batch_cuda<
                EquityPack,RoughPack,RatePolicy,Product,false,TeamSize>(
                    inputs,offset,count,0U,paths_per_price,
                    blocks,blocks,geometry.threads_per_block,
                    steps_per_day,dt,base_seed,nullptr,
                    device_moments.data(),nullptr,
                    blocks==1U ? device_prices.data() : nullptr,
                    blocks==1U ? device_errors.data() : nullptr);
            if(blocks>1U) {
                monte_carlo::finalize_price_batch_kernel
                    <<<static_cast<unsigned>(count),128U,
                        8U*sizeof(double)>>>(
                        device_moments.data(),blocks,paths_per_price,
                        offset,device_prices.data(),device_errors.data());
                check_cuda(cudaGetLastError(),
                    "cross-asset batch finalize");
            }
            if(checkpoint.enabled()) {
                batch_timer.finish_batch();
                commit_batch(offset,count);
            } else {
                progress.record_cuda_progress(offset+count);
            }
        }
    } else {
        const auto launch_chunk=[&](std::size_t result,std::size_t slot,std::size_t offset,std::size_t rows,
                                    cudaStream_t stream) {
            const auto product_index=
                row_plan.selectors[EquityPack::kAssetCount+rough_count+1U]
                    .select(static_cast<std::uint32_t>(result));
            const auto steps=products[product_index].maturity_days*steps_per_day;
            for(std::size_t j=1U;j<rows;++j) {
                const auto pi=row_plan.selectors[EquityPack::kAssetCount+rough_count+1U].select(result+j);
                if(products[pi].maturity_days*steps_per_day!=steps)
                    throw std::invalid_argument("cross-asset row batch must retain its native time grid");
            }
            auto* row_workspace=workspace.data()+slot*workspace_bytes;
            auto* row_volatility=device_volatility.data()
                +slot*volatility_values;
            auto* row_moments=device_moments.data()+slot*max_partials;
            {
                const auto count=std::min(chunk,
                    paths_per_price-offset);
                device_rough.template launch_volatilities<
                    dimension,EquityPack::kAssetCount>(
                        device_factors.data(),result,offset,count,dt,steps,
                        direct_tile_size,base_seed,row_workspace,
                        workspace_bytes,row_volatility,stream,
                        device_selectors.data(),
                        EquityPack::kAssetCount+rough_count+2U,rows,workspace_bytes,volatility_values);
                const auto blocks=static_cast<unsigned>((count+paths_per_block-1U)/paths_per_block);
                monte_carlo::launch_cross_asset_batch_cuda<
                    EquityPack,RoughPack,RatePolicy,Product,false,TeamSize>(
                        inputs,result,rows,offset,count,max_partials,blocks,path_threads,
                        steps_per_day,dt,base_seed,row_volatility,row_moments,nullptr,
                        nullptr,nullptr,stream,paths_per_team,(offset/chunk)*chunk_blocks,volatility_values);
            }
        };
        std::vector<cuda::Stream> streams;
        std::vector<cuda::Event> completed;
        if(rough_batch_size>1U) {
            streams.resize(rough_batch_size);
            completed.resize(rough_batch_size);
            for(auto& stream:streams)
                check_cuda(cudaStreamWaitEvent(stream.get(),start.get()),
                    "cross-asset rough batch start dependency");
        }
        // Keep the global result index: selectors and Philox both use it.
        // A completed prefix can therefore resume without changing a path.
        std::size_t progress_rows=128U;
        if(const char* configured=std::getenv(
            "AI_FACTORY_CROSS_ASSET_PROGRESS_ROWS"
        ); configured!=nullptr && *configured!='\0') {
            progress_rows=std::stoull(configured);
            if(progress_rows==0U || progress_rows>4096U)
                throw std::invalid_argument(
                    "Invalid cross-asset progress batch size");
        }
        std::size_t batch_begin=checkpoint.completed_prices();
        if(tracked_batches && batch_begin<result_count)
            batch_timer.start_batch();
        for(std::size_t result=batch_begin;result<result_count;) {
            const auto remaining=tracked_batches ? std::min(result_count-result,progress_rows-(result-batch_begin)) : result_count-result;
            std::vector<std::size_t> group_starts,group_sizes;
            std::size_t count=0U;
            for(std::size_t stream=0U;stream<rough_batch_size && count<remaining;++stream) {
                const auto product_slot=EquityPack::kAssetCount+rough_count+1U;
                const auto maturity=[&](std::size_t index) {
                    return products[row_plan.selectors[product_slot].select(static_cast<std::uint32_t>(index))].maturity_days;
                };
                std::size_t rows=1U;
                while(rows<launch_rows && count+rows<remaining && maturity(result+count+rows)==maturity(result+count)) ++rows;
                group_starts.push_back(count);group_sizes.push_back(rows);count+=rows;
            }
            for(std::size_t offset=0U;offset<paths_per_price;offset+=chunk)
                for(std::size_t group=0U;group<group_starts.size();++group)
                    launch_chunk(result+group_starts[group],group*launch_rows,offset,group_sizes[group],
                        rough_batch_size==1U ? nullptr : streams[group].get());
            for(std::size_t group=0U;group<group_starts.size();++group) {
                const auto stream=rough_batch_size==1U ? nullptr : streams[group].get();
                auto* moments=device_moments.data()+group*launch_rows*max_partials;
                monte_carlo::finalize_price_batch_kernel<<<static_cast<unsigned>(group_sizes[group]),128U,8U*sizeof(double),stream>>>(
                    moments,max_partials,paths_per_price,result+group_starts[group],device_prices.data(),device_errors.data());
                check_cuda(cudaGetLastError(),"cross-asset multirow finalize");
            }
            const auto next=result+count;
            result=next;
            if(tracked_batches && (next-batch_begin==progress_rows
                || next==result_count)) {
                if(rough_batch_size==1U) {
                    check_cuda(cudaDeviceSynchronize(),
                        "cross-asset rough batch completion");
                } else {
                    for(auto& stream:streams)
                        check_cuda(cudaStreamSynchronize(stream.get()),
                            "cross-asset rough batch completion");
                }
                batch_timer.finish_batch();
                commit_batch(batch_begin,next-batch_begin);
                batch_begin=next;
                if(next<result_count) batch_timer.start_batch();
            }
        }
        if(rough_batch_size>1U) {
            for(std::size_t slot=0U;slot<rough_batch_size;++slot) {
                check_cuda(cudaEventRecord(
                    completed[slot].get(),streams[slot].get()),
                    "cross-asset rough row completion");
                check_cuda(cudaStreamWaitEvent(0,completed[slot].get()),
                    "cross-asset rough batch completion");
            }
        }
    }
    check_cuda(cudaEventRecord(stop.get()),"cross-asset timer stop");
    check_cuda(cudaEventSynchronize(stop.get()),"cross-asset timer wait");
    float ms=0.0f;
    check_cuda(cudaEventElapsedTime(&ms,start.get(),stop.get()),
        "cross-asset elapsed");
    cuda::MonteCarloRun run;
    run.prices.resize(result_count);
    run.standard_errors.resize(result_count);
    device_prices.copy_to(run.prices.data());
    device_errors.copy_to(run.standard_errors.data());
    checkpoint.restore_prefix({
        std::span<float>(run.prices),
        std::span<float>(run.standard_errors)});
    run.kernel_seconds=checkpoint.enabled()
        || (rough_count>0U && progress.enabled())
        ? batch_timer.seconds() : static_cast<double>(ms)*1.0e-3;
    run.wall_seconds=std::chrono::duration<double>(
        std::chrono::steady_clock::now()-wall_start).count();
    progress.complete();
    return run;
}

}  // namespace ai_factory::workbench::offline::pricing
