#include "kestensimulation.h"

namespace {
    bool do_norm(int step, int norm_steps) {
        return step % norm_steps == 0;
    }

    bool do_strct(int step, int strct_steps) {
        return step % strct_steps == 0;
    }

    bool syn_active(double w) {
        return w > 0;
    }
}

std::ostream& operator<<(std::ostream& stream, const std::forward_list<StructuralPlasticityEvent>& events)
{
    for (const auto& event : events) {
        stream << (int) event.type << " " << event.t << " " << event.i << " " << event.j << "\n";
    }
    return stream;
}

// TODO swap ij
// currently we do j->i but I think (check!) brian does i->j
KestenSimulation::KestenSimulation(const Parameters& p_)
        : p(p_)
        , t_begin()
        , t_print()
        , step(0)
        , steps(std::ceil(p.T/p.dt))
        , norm_steps(std::ceil(p.dt_norm/p.dt))
        , strct_steps(std::ceil(p.dt_strct/p.dt))
        , w(p.N_e, std::vector<double>(p.N_e-1, 0.0))
        , gen(p.seed)
        , unif(0.0, 1.0)
        , xi_kesten(0, sqrt(p.dt))
        , n_available(w.size()*w[0].size())
        , n_should_be_active(std::ceil(p.p_conn_fraction*n_available))
{
    // initialize weights
    for (auto& neuron_w : w) {
        for (auto& weight: neuron_w) {
            if (unif(gen) <= p.p_conn_fraction)
                weight = p.w_min+0.01*p.w_min; // add a bit to not prune synapses in first pruning event
        }
    }
}

bool KestenSimulation::hasNextStep() const
{
    return step < steps;
}

void KestenSimulation::doStep()
{
    if (!hasNextStep())
        return;

    t_begin = std::chrono::steady_clock::now();
    t_print = std::chrono::steady_clock::now();

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

        for (int i = 0; i < p.N_e; ++i) {
            for (int j = 0; j < p.N_e-1; ++j) {
                auto& w_single = w[i][j];
                if (syn_active(w_single)) { // prune?
                    bool should_stay_active = w_single > p.w_min || (w_single <= p.w_min && (unif(gen) > p.p_inact));
                    if (!should_stay_active) {
                        w_single = 0.0;
                        structual_events.emplace_front(
                                StructuralPlasticityEventType::Destroy,
                                ((double)step)/((double)steps) * p.T/second,
                                i, j
                        );
                    }
                } else { // create?
                    bool should_become_active = unif(gen) <= p_insert;
                    if (should_become_active) {
                        w_single = p.w_min;
                        structual_events.emplace_front(
                                StructuralPlasticityEventType::Create,
                                ((double)step)/((double)steps) * p.T/second,
                                i, j
                        );
                    }
                }
            }
        }
    }

    // kesten
    for (int i = 0; i < p.N_e; i++) {
        for (int j = 0; j < p.N_e-1; j++) {
            if (syn_active(w[i][j])) {
                // using Euler-Maruyama method
                w[i][j] = w[i][j]
                          + ((p.syn_kesten_mu_epsilon_1 * w[i][j] + p.syn_kesten_mu_eta)) * p.dt
                          + sqrt(p.syn_kesten_var_epsilon_1 * pow(w[i][j], 2) + p.syn_kesten_var_eta) * xi_kesten(gen);
                w[i][j] = std::clamp(w[i][j], p.w_min, p.w_max);
            }
        }
    }

    // normalization
    if (do_norm(step, norm_steps)) {
        for (int i = 0; i < p.N_e; i++) {
            double w_sum = 0;
            for (int j = 0; j < p.N_e-1; j++) {
                w_sum += w[i][j];
            }
            for (int j = 0; j < p.N_e-1; j++) {
                if (syn_active(w[i][j])) {
                    w[i][j] = std::clamp(w[i][j] * (1 + p.eta_norm * (p.eta_targ / w_sum - 1)), p.w_min, p.w_max);
                }
            }
        }
    }

    // print time passed every minute
    std::chrono::steady_clock::time_point t_now = std::chrono::steady_clock::now();
    auto t_since_print = std::chrono::duration_cast<std::chrono::minutes>(t_now - t_print).count();
    auto t_since_begin = std::chrono::duration_cast<std::chrono::seconds>(t_now - t_begin).count();

    if (t_since_print >= 1) {
        std::cout << "step " << step << "/" << steps << " done in " << t_since_begin << " seconds" << std::endl;
        t_print = std::chrono::steady_clock::now();
    }

    step++;
}

void KestenSimulation::saveResults()
{
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
