#ifndef KESTEN_SIM_KESTENSIM_H
#define KESTEN_SIM_KESTENSIM_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <forward_list>
#include <chrono>

const static double second = 1000.0; // in ms
const static int N_e = 1600;
const static double eta_norm = 1.0;
const static double eta_targ = 12.5;
const static double w_min = 0.0363;
const static double w_max = 0.3244;
const static double p_conn_fraction = 0.08;
const static double p_inact = 0.1;
const static double T = 20*second;
const static double dt = 100.0; //ms
const static double dt_norm = 100.0; //ms
const static double dt_strct = 1000.0; //ms

bool do_norm(int step, int norm_steps) {
    return step % norm_steps == 0;
}

bool do_strct(int step, int strct_steps) {
    return step % strct_steps == 0;
}

bool syn_active(double w) {
    return w > 0;
}

double clamp(double w) {
    return std::clamp(w, w_min, w_max);
}

enum class StructuralPlasticityEventType {Destroy = 0, Create = 1};
struct StructuralPlasticityEvent {  // 1 create 0 destroy | t | i | j
    StructuralPlasticityEventType type;
    double t; // in seconds
    int i;
    int j;
};

std::ostream& operator<<(std::ostream& stream, const std::forward_list<StructuralPlasticityEvent>& events) {
    for (const auto& event : events) {
        stream << (int) event.type << " " << event.t << " " << event.i << " " << event.j << "\n";
    }
    return stream;
}


void kestensim() {
    std::mt19937 gen{193945};

    std::vector<std::vector<double>> w(N_e, std::vector<double>(N_e-1, 0.0));
    std::vector<double> xi_kesten(w.size());
    std::forward_list<StructuralPlasticityEvent> structual_events;

    std::normal_distribution<double> norm(0, sqrt(dt)); // Euler-Maruyama method

    const int n_available = w.size()*w[0].size();
    const int n_should_be_active = std::ceil(p_conn_fraction*n_available);

    int steps = std::ceil(T/dt);
    int norm_steps = std::ceil(dt_norm/dt);
    int strct_steps = std::ceil(dt_strct/dt);

    const double syn_kesten_mu_epsilon_1 = -0.032/second;
    const double syn_kesten_mu_eta = 0.003/second;
    const double syn_kesten_var_epsilon_1 = 0.0011/second ;
    const double syn_kesten_var_eta = 0.000028/second;

    std::uniform_real_distribution<double> unif(0.0, 1.0);
    for (auto& neuron_w : w) {
        for (auto& weight: neuron_w) {
            if (unif(gen) <= p_conn_fraction)
                weight = w_min+0.01*w_min; // add a bit to not prune synapses in first pruning event
        }
    }

    std::chrono::steady_clock::time_point t_begin = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point t_print = std::chrono::steady_clock::now();

    for (int step = 0; step < steps; step++) {
        // structural plasticity
        if (do_strct(step, strct_steps)) { // when="start"
            int n_active = 0;
            for (auto& neuron_w : w) {
                for (auto& w_single : neuron_w) {
                    if (syn_active(w_single)) {
                        n_active++;
                    }
                }
            }
            double p_insert = std::clamp((double)(n_should_be_active-n_active)/(double)(n_available-n_active),
                                         0.0, 1.0);
//            std::cout << "n_active " << n_active << std::endl;
//            std::cout << "fraction active " << (double)n_active/n_available << std::endl;
//            std::cout << "p_insert " << p_insert << std::endl;
//            std::cout << "=======" << std::endl;

            for (int i = 0; i < N_e; ++i) {
                for (int j = 0; j < N_e-1; ++j) {
                    auto& w_single = w[i][j];
                    if (syn_active(w_single)) { // prune?
                        bool should_stay_active = w_single > w_min || (w_single <= w_min && (unif(gen) > p_inact));
                        if (!should_stay_active) {
                            w_single = 0.0;
                            structual_events.emplace_front(
                                    StructuralPlasticityEventType::Destroy,
                                    ((double)step)/((double)steps) * T/second,
                                    i, j
                            );
                        }
                    } else { // create?
                        bool should_become_active = unif(gen) <= p_insert;
                        if (should_become_active) {
                            w_single = w_min;
                            structual_events.emplace_front(
                                    StructuralPlasticityEventType::Create,
                                    ((double)step)/((double)steps) * T/second,
                                    i, j
                            );
                        }
                    }
                }
            }
        }

        // kesten
        for (int i = 0; i < N_e; i++) {
            for (int j = 0; j < N_e-1; j++) {
                if (syn_active(w[i][j])) {
                    // using Euler-Maruyama method
                    w[i][j] = w[i][j]
                              + ((syn_kesten_mu_epsilon_1 * w[i][j] + syn_kesten_mu_eta)) * dt
                              + sqrt(syn_kesten_var_epsilon_1 * pow(w[i][j], 2) + syn_kesten_var_eta) * norm(gen);
                    w[i][j] = clamp(w[i][j]);
                }
            }
        }

        // normalization
        if (do_norm(step, norm_steps)) {
            for (int i = 0; i < N_e; i++) {
                double w_sum = 0;
                for (int j = 0; j < N_e-1; j++) {
                    w_sum += w[i][j];
                }
                for (int j = 0; j < N_e-1; j++) {
                    if (syn_active(w[i][j])) {
                        w[i][j] = clamp(w[i][j] * (1 + eta_norm * (eta_targ / w_sum - 1)));
                    }
                }
            }
        }

        std::chrono::steady_clock::time_point t_now = std::chrono::steady_clock::now();
        auto t_since_print = std::chrono::duration_cast<std::chrono::minutes>(t_now - t_print).count();
        auto t_since_begin = std::chrono::duration_cast<std::chrono::seconds>(t_now - t_begin).count();

        if (t_since_print >= 1) {
            std::cout << "step " << step << "/" << steps << " done in " << t_since_begin << " seconds" << std::endl;
            t_print = std::chrono::steady_clock::now();
        }
    }

    std::chrono::steady_clock::time_point t_now = std::chrono::steady_clock::now();
    auto t_since_begin = std::chrono::duration_cast<std::chrono::seconds>(t_now - t_begin).count();

    std::cout << steps << " steps " << " done in " << t_since_begin << " seconds" << std::endl;
    std::cout << "storing results..." << std::endl;


    std::ofstream output_file("./weights.txt");
    for (const auto& neuron_w : w) {
        for (const auto& weight: neuron_w) {
            if (weight > 0)
                output_file << weight << "\n";
        }
    }
    output_file.close();

    structual_events.reverse();
    std::ofstream turnover_file("./turnover.txt");
    turnover_file << structual_events;
    turnover_file.close();
}

#endif //KESTEN_SIM_KESTENSIM_H
