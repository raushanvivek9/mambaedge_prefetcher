#include "mamba_ml.h"
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <deque>
#include <algorithm>
#include <cstdlib>

// =================================================
// Global model state
// =================================================
static torch::jit::script::Module ml_model;
static bool model_loaded = false;
static bool tried_load = false;

// =================================================
// IPEX Adaptive Throttle
// =================================================
enum PowerState { HIGH, MED, LOW };

static PowerState pf_state = HIGH;
static std::deque<int> acc_window;

static int below_count = 0;
static int above_count = 0;

static const int window_size = 10;
static const float throttle_thresh = 0.50f;
static const float recover_thresh  = 0.70f;
static const int throttle_pat = 3;
static const int recover_pat  = 5;

// =================================================
// Simulated Voltage States
// =================================================
enum VoltageState { BOOST, NORMAL, LOWV };

static VoltageState volt_state = NORMAL;

// =================================================
// Helpers
// =================================================
static const char* state_name()
{
    if (pf_state == HIGH) return "HIGH";
    if (pf_state == MED)  return "MED";
    return "LOW";
}

static const char* volt_name()
{
    if (volt_state == BOOST) return "BOOST";
    if (volt_state == NORMAL) return "NORMAL";
    return "LOWV";
}

// IPEX degree: lower = more aggressive
static int current_degree()
{
    if (pf_state == HIGH) return 2;
    if (pf_state == MED)  return 4;
    return 8;
}

// Voltage cap degree
static int voltage_degree()
{
    if (volt_state == BOOST) return 2;
    if (volt_state == NORMAL) return 4;
    return 8;
}

// Final degree = stricter of the two
static int final_degree()
{
    return std::max(current_degree(), voltage_degree());
}

// =================================================
// Update IPEX based on usefulness
// =================================================
static void update_ipex(bool useful)
{
    acc_window.push_back(useful ? 1 : 0);

    if ((int)acc_window.size() > window_size)
        acc_window.pop_front();

    float avg = 0.0f;
    for (int x : acc_window) avg += x;
    avg /= std::max(1, (int)acc_window.size());

    if (avg < throttle_thresh) {
        below_count++;
        above_count = 0;

        if (below_count >= throttle_pat) {
            if (pf_state == HIGH) pf_state = MED;
            else if (pf_state == MED) pf_state = LOW;

            below_count = 0;
        }
    }
    else if (avg > recover_thresh) {
        above_count++;
        below_count = 0;

        if (above_count >= recover_pat) {
            if (pf_state == LOW) pf_state = MED;
            else if (pf_state == MED) pf_state = HIGH;

            above_count = 0;
        }
    }
    else {
        below_count = 0;
        above_count = 0;
    }
}

// =================================================
// Simulate Real-Time Voltage Fluctuation
// =================================================
static void update_voltage(uint64_t access_count)
{
    // periodic phase change every 1M accesses
    uint64_t phase = (access_count / 1000000ULL) % 3ULL;

    if (phase == 0) volt_state = BOOST;
    else if (phase == 1) volt_state = NORMAL;
    else volt_state = LOWV;

    // optional random transient dips (2%)
    if ((std::rand() % 100) < 2)
        volt_state = LOWV;
}

// =================================================
// Load model once
// =================================================
static void load_model_once()
{
    if (tried_load) return;
    tried_load = true;

    try {
        ml_model = torch::jit::load("mambaedge_final_scripted.pt");
        ml_model.eval();
        model_loaded = true;
        std::cout << "[MAMBA_ML] Model loaded successfully\n";
    } catch (const c10::Error& e) {
        std::cerr << "[MAMBA_ML] Model load failed:\n";
        std::cerr << e.what() << std::endl;
    }
}

// =================================================
// Main Prefetch Logic
// =================================================
uint32_t mamba_ml::prefetcher_cache_operate(
    champsim::address addr,
    champsim::address ip,
    uint8_t cache_hit,
    bool useful_prefetch,
    access_type type,
    uint32_t metadata_in)
{
    load_model_once();

    if (!model_loaded)
        return metadata_in;

    static std::deque<int64_t> page_hist;
    static std::deque<int64_t> off_hist;

    static uint64_t access_count = 0;
    static uint64_t last_prefetch = 0;

    access_count++;

    // Update controllers
    update_ipex(useful_prefetch);
    update_voltage(access_count);

    uint64_t a = addr.to<uint64_t>();

    int64_t page = (a >> 12) % 512;
    int64_t off  = ((a & 4095ULL) >> 6) % 64;

    if (page_hist.size() == 32) {

        auto page_tensor = torch::zeros({1,32}, torch::kInt64);
        auto off_tensor  = torch::zeros({1,32}, torch::kInt64);

        for (int i = 0; i < 32; i++) {
            page_tensor.index_put_({0,i}, page_hist[i]);
            off_tensor.index_put_({0,i}, off_hist[i]);
        }

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(page_tensor);
        inputs.push_back(off_tensor);

        torch::NoGradGuard no_grad;
        auto output = ml_model.forward(inputs).toTuple();

        auto page_logits = output->elements()[0].toTensor();
        auto off_logits  = output->elements()[1].toTensor();

        int64_t pred_page = page_logits.argmax(1).item<int64_t>();
        int64_t pred_off  = off_logits.argmax(1).item<int64_t>();

        if (pred_off == 0)
            pred_off = (off + 1) % 64;

        auto off_prob = torch::softmax(off_logits, 1);
        float conf = off_prob.max().item<float>();

        int degree = final_degree();

        static int dbg = 0;
        if (dbg < 20) {
            std::cout << "[MAMBA_ML] "
                      << "curr=(" << page << "," << off << ") "
                      << "pred=(" << pred_page << "," << pred_off << ") "
                      << "conf=" << conf
                      << " ipex=" << state_name()
                      << " volt=" << volt_name()
                      << " deg=" << degree
                      << "\n";
            dbg++;
        }

        if (access_count % degree == 0) {

            if (conf > 0.90f &&
                pred_off > off &&
                (pred_off - off) <= 4) {

                uint64_t pred_addr =
                    (a & ~4095ULL) |
                    ((uint64_t)pred_off << 6);

                if (pred_addr != 0 &&
                    pred_addr != a &&
                    pred_addr != last_prefetch) {

                    bool issued =
                        prefetch_line(
                            champsim::address(pred_addr),
                            true,
                            metadata_in
                        );

                    if (issued)
                        last_prefetch = pred_addr;

                    static int pfp = 0;
                    if (pfp < 20) {
                        std::cout << "[MAMBA_ML] prefetch "
                                  << (issued ? "ISSUED " : "REJECTED ")
                                  << "0x" << std::hex << pred_addr
                                  << std::dec
                                  << " deg=" << degree
                                  << "\n";
                        pfp++;
                    }
                }
            }
        }
    }

    // Update history after inference
    page_hist.push_back(page);
    off_hist.push_back(off);

    if (page_hist.size() > 32) page_hist.pop_front();
    if (off_hist.size() > 32) off_hist.pop_front();

    return metadata_in;
}

// =================================================
// Cache Fill Hook
// =================================================
uint32_t mamba_ml::prefetcher_cache_fill(
    champsim::address addr,
    long set,
    long way,
    uint8_t prefetch,
    champsim::address evicted_addr,
    uint32_t metadata_in)
{
    return metadata_in;
}