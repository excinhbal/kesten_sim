#include "kestensimulation.h"

namespace {
    bool do_norm(int step, int norm_steps) {
        return step % norm_steps == 0;
    }

    bool do_strct(int step, int strct_steps) {
        return step % strct_steps == 0;
    }
}

KestenStep::KestenStep(const Parameters& p_)
    : p(p_)
    , norm(0, 1)
{ }

void KestenStep::step(std::mt19937& gen, std::vector<std::vector<double>>& w)
{
    for (auto& neuron_w : w) {
        for (auto& w_: neuron_w) {
            // using Stochastic Heun method, scheme from Brian2 (see brian2/stateupdaters/explicit.py)
            // x - the variable
            // g(x,t) - part of the equation that is stochastic
            // f(x,t) - non-stochastic part
            // dW ~ Norm(0, sqrt(dt))
            //
            // x_support = x + g(x,t) * dW
            // g_support = g(x_support,t+dt)
            // x_new = x + dt*f(x,t) + .5*dW*(g(x,t)+g_support)

            double xi_kesten = sqrt(p.dt) * norm(gen);
            auto g = [this](const double w_) {
                return sqrt(p.syn_kesten_var_eta + p.syn_kesten_var_epsilon_1 * pow(w_, 2));
            };
            auto f = [this](const double w_) {
                return p.syn_kesten_mu_eta + p.syn_kesten_mu_epsilon_1 * w_;
            };
            double x_support = w_ + norm(gen) * g(w_);
            double g_support = g(x_support);
            w_ = w_ + p.dt * f(w_) + 0.5 * xi_kesten * (g(w_) + g_support);
            if (p.do_clamp_after_kesten)
                w_ = std::clamp(w_, p.w_min, p.w_max);
        }
    }
}

QuadStep::QuadStep(const QuadParameters& p_)
    : p(p_)
    , norm(0, 1)
{ }

void QuadStep::step(std::mt19937& gen, std::vector<std::vector<double>>& w)
{
    for (auto& neuron_w : w) {
        for (auto& w_: neuron_w) {
            w_ += p.mu_alpha + p.mu_beta_1*w_ + p.mu_gamma*pow(w_, 2)
                  + sqrt(p.var_alpha + p.var_beta_1*pow(w_, 2) + p.var_gamma*pow(w_, 4)) * norm(gen);
            if (p.do_clamp_after_kesten)
                w_ = std::clamp(w_, p.w_min, p.w_max);
        }
    }
}

template<typename P, typename L>
KestenSimulation<P, L>::KestenSimulation(const P& p_, NodeParameters nP_)
        : p(p_)
        , nP(nP_)
        , n_ownNeurons(nP.N_e.has_value() ? nP.N_e.value() : p.N_e)
        , n_potentiallyIncoming(p.N_e - 1)
        , t_begin()
        , t_print()
        , step(0)
        , steps(std::ceil(p.T/p.dt))
        , norm_steps(std::ceil(p.dt_norm/p.dt))
        , strct_steps(std::ceil(p.dt_strct/p.dt))
        , w(n_ownNeurons, std::vector<double>(n_potentiallyIncoming, 0.0))
        , is(n_ownNeurons, std::vector<unsigned short>(n_potentiallyIncoming, 0))
        , creation_times(n_ownNeurons, std::vector<StructuralPlasticityEvent*>(n_potentiallyIncoming, nullptr))
        , gen(p.seed + nP.seedOffset)
        , unif(0.0, 1.0)
        , n_available(p.N_e*w[0].size())
        , n_should_be_active(std::ceil(p.p_conn_fraction*n_available))
        , stepper(p)
{
    bool use_init_distribution = p.init_distribution_bins.size() > 0;
    if (use_init_distribution && p.init_distribution_bins.size() != p.init_distribution_density.size()+1) {
        std::cerr << "init_distribution_bins must have one more entry that density" << std::endl;
        exit(1);
    }
    const std::size_t center_count = use_init_distribution ? p.init_distribution_bins.size() - 1 : 0;
    std::vector<double> init_distribution_bins_center(center_count);

    if (use_init_distribution) {
        std::transform(p.init_distribution_bins.cbegin(), --p.init_distribution_bins.cend(),
                       ++p.init_distribution_bins.cbegin(),
                       init_distribution_bins_center.begin(), [](double a, double b) { return (a + b) / 2; });
    }
    std::discrete_distribution<> bin_center_dist(p.init_distribution_density.cbegin(), p.init_distribution_density.cend());

    int n_active_initially = 0;
    // initialize weights
    for (int j = 0; j < w.size(); ++j) { // TODO use fancy iterator
        w[j].resize(0); // this does not reduce capacity, but we will only iterate over the active elements
                        // in the beginning of the vector
                        // TODO save memory by reserving less - which could lead to reallocations
        is[j].resize(0);
        for (int i = 0; i < n_potentiallyIncoming; i++) {
            if (unif(gen) <= p.p_conn_fraction) {
                double w_init = p.w_min * 1.01; // add a bit to not prune synapses in first pruning event
                if (use_init_distribution)
                    w_init = init_distribution_bins_center[bin_center_dist(gen)];
                w[j].push_back(w_init);
                is[j].push_back(i);
            }
        }
        n_active_initially += is[j].size();
    }
    active_initial.reserve(n_active_initially);
    for (unsigned short j = 0; j < w.size(); ++j) {
        for (const unsigned short i : is[j]) {
            active_initial.emplace_back(i, j);
            // so that we can observe A and P types
            initial_structual_events.emplace_front(
                    StructuralPlasticityEventType::Create,
                    /* t= */ 0, i, nP.neuronOffset+j
            );
            creation_times[j][i] = &initial_structual_events.front();
        }
    }
    std::cout << std::endl;
}

template<typename P, typename L>
bool KestenSimulation<P, L>::hasNextStep() const
{
    return step < steps;
}

template<typename P, typename L>
void KestenSimulation<P, L>::doStep()
{
    if (!hasNextStep())
        return;

    if (step == 0) {
        t_begin = std::chrono::steady_clock::now();
        t_print = std::chrono::steady_clock::now();
    }

    // structural plasticity
    if (do_strct(step, strct_steps)) { // when="start"
        auto size_acc = [](const int& acc, const auto& neuron_w) { return acc + neuron_w.size(); };
        int n_active = std::accumulate(w.cbegin(), w.cend(), 0, size_acc);
        n_active = synchronizeActive(n_active);
        double p_insert = std::clamp((double)(n_should_be_active-n_active)/(double)(n_available-n_active),
                                     0.0, 1.0);
//            std::cout << "n_active " << n_active << std::endl;
//            std::cout << "fraction active " << (double)n_active/n_available << std::endl;
//            std::cout << "p_insert " << p_insert << std::endl;
//            std::cout << "=======" << std::endl;

        // Brian2 uses i->j which is not how our array is laid out
        const double w_prune = p.w_min;
        const int t = (int) ((double) step) / ((double) steps) * p.T / second;
        for (int j = 0; j < n_ownNeurons; ++j) {
            auto index_i = is[j].begin();
            for (int i = 0; i < n_potentiallyIncoming; ++i) {
                if (index_i != is[j].end() && *index_i == i) { // active -> prune?
                    auto& w_current = w[j][index_i - is[j].begin()];
                    bool should_stay_active = w_current > w_prune || (w_current <= w_prune && (unif(gen) > p.p_inact));
                    if (!should_stay_active) {
                        w[j].erase(w[j].begin() + (index_i - is[j].begin()));
                        is[j].erase(index_i);
                        structual_events.emplace_front(
                                StructuralPlasticityEventType::Destroy,
                                t,
                                i, nP.neuronOffset+j
                        );
                        auto creation_event = creation_times[j][i];
                        if (creation_event) {
                            observation_times.emplace_front(
                                    (std::uint32_t)creation_event->t, (std::uint32_t)t-creation_event->t
                            );
                            // std::cout << surv_time << std::endl;
                            creation_times[j][i] = nullptr;
                        } else {
                            std::cout << "no creation event for " << i << " " << j << " at " << t << std::endl;
                            exit(10);
                        }
                        // erase operation automatically makes index_i point at next element
                    } else {
                        index_i++;
                    }
                } else { // create?
                    bool should_become_active = unif(gen) <= p_insert;
                    if (should_become_active) {
                        w[j].push_back(p.w_min);
                        index_i = std::upper_bound(is[j].begin(), is[j].end(), i);
                        is[j].insert(index_i, i);
                        index_i++;
                        // index_i is now pointing at the next element after i, which is where it should point at
                        // for the next continuation of the for step
                        structual_events.emplace_front(
                                StructuralPlasticityEventType::Create,
                                t,
                                i, nP.neuronOffset+j
                        );
                        creation_times[j][i] = &structual_events.front();
                    }
                }
            }
        }
    }

    // kesten
    stepper.step(gen, w);

    // normalization
    if (p.do_norm && do_norm(step, norm_steps)) {
        for (int j = 0; j < n_ownNeurons; j++) {
            double w_sum = std::accumulate(w[j].cbegin(), w[j].cend(), 0.0);
            auto normalize = [w_sum, this](double& w_current) { return w_current * (1 + p.eta_norm * (p.eta_targ / w_sum - 1)); };
            auto clamp = [this](double& w_current) { return std::clamp(w_current, p.w_min, p.w_max); };
            std::transform(w[j].begin(), w[j].end(), w[j].begin(), normalize);
            std::transform(w[j].begin(), w[j].end(), w[j].begin(), clamp);
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

template<typename P, typename L>
void KestenSimulation<P, L>::afterLastStep()
{
    // observe A and C types
    const int t = (int) ((double) step) / ((double) steps) * p.T / second;
    for (unsigned short j = 0; j < w.size(); ++j) {
        for (const unsigned short i : is[j]) {
            auto creation_event = creation_times[j][i];
            if (creation_event) {
                observation_times.emplace_front(
                    (std::uint32_t)creation_event->t, (std::uint32_t)t-creation_event->t
                );
            }
        }
    }
}

template<typename P, typename L>
void KestenSimulation<P, L>::saveResults()
{
    std::chrono::steady_clock::time_point t_now = std::chrono::steady_clock::now();
    auto t_since_begin = std::chrono::duration_cast<std::chrono::seconds>(t_now - t_begin).count();
    std::cout << steps << " steps " << " done in " << t_since_begin << " seconds" << std::endl;

    std::cout << "storing results..." << std::endl;
    std::ofstream output_file("./weights.txt");
    for (const auto& neuron_w : w) {
        for (const auto& weight: neuron_w) {
            output_file << weight << "\n";
        }
    }
    output_file.close();

    structual_events.reverse();
    std::ofstream turnover_file("./turnover.txt");
    turnover_file << structual_events;
    turnover_file.close();

    std::ofstream initial_active_file("./initial_active.txt");
    initial_active_file << active_initial;
    initial_active_file.close();
}

template<typename P, typename L>
int KestenSimulation<P, L>::synchronizeActive(int n_active)
{
    return n_active;
}

template class KestenSimulation<Parameters, KestenStep>;
template class KestenSimulation<QuadParameters, QuadStep>;

std::ostream& operator<<(std::ostream& stream, const ObservationTime& event)
{
    stream << event.t_creation << " " << event.t_observation;
    return stream;
}
